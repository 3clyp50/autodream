"""Run with framework Python; no model calls or real memory writes."""
import asyncio
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_dream_coverage import dream


async def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(dream, "get_autodream_root", side_effect=lambda scope: str(root / scope)):
            assert dream.load_dream_stats("default") == {}
            run = {"started_at": "2026-09-30T12:00:00+00:00", "finished_at": "2026-09-30T12:00:05+00:00",
                   "status": "updated", "model_calls": 3, "input_tokens": 100, "output_tokens": 20,
                   "duration_seconds": 5, "sessions": 2}
            dream.record_dream_stats("default", run)
            dream.record_dream_stats("default", {**run, "status": "failed"})
            dream.record_dream_stats("default", {**run, "status": "noop"})
            totals = dream.load_dream_stats("default")
            assert totals["completed_dreams"] == 2 and totals["failed_dreams"] == 1
            assert totals["model_calls"] == 9
            assert totals["input_tokens"] == 300 and totals["output_tokens"] == 60
            assert totals["duration_seconds"] == 15
            assert totals["last_run"]["status"] == "noop"
            assert dream.load_dream_stats("projects/other") == {}
            dream.save_auto_dream_state("default", {"last_dream_at": "2026-09-01T12:00:00+00:00"})
            assert dream.load_dream_stats("default") == totals
            spec = importlib.util.spec_from_file_location(
                "autodream_stats_api", Path(__file__).resolve().parents[1] / "api/stats.py"
            )
            api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api)
            with patch.object(api, "auto_dream", dream), patch.object(dream, "get_memory_plugin_config", return_value={}):
                handler = api.Stats(None, None)
                assert handler.requires_auth() and handler.requires_csrf()
                assert (await handler.process({}, None))["stats"] == totals
                scoped = await handler.process({"project_name": "other", "agent_profile": "agent0"}, None)
                assert scoped["memory_scope"] == "projects/other" and scoped["stats"] == {}
                with patch.object(dream, "get_memory_plugin_config", return_value={"project_memory_isolation": False}):
                    shared = await handler.process({"project_name": "other", "agent_profile": "agent0"}, None)
                    assert shared["memory_scope"] == "default" and shared["stats"] == totals
                for value in ("../other", "/absolute", "..", "bad\\name", 5):
                    for key in ("project_name", "agent_profile"):
                        try:
                            await handler.process({key: value}, None)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("Unsafe scope accepted")
            path = root / "default/stats.json"
            path.write_text("broken JSON")
            try:
                dream.record_dream_stats("default", run)
            except json.JSONDecodeError:
                pass
            else:
                raise AssertionError("Corrupt statistics must not silently reset totals")
            assert path.read_text() == "broken JSON"
    print("PASS stats persistence, shared/isolated API scopes, auth/CSRF, invalid paths, and corrupt-data preservation")


if __name__ == "__main__":
    asyncio.run(main())
