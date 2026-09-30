"""From the framework root, using its Python runtime (no model or DB calls):
PYTHONPATH=. python usr/plugins/autodream/tests/test_dream_coverage.py
"""
import asyncio
import importlib.util
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from helpers.history import History
from langchain_core.documents import Document
from plugins._memory.helpers.memory import Memory

spec = importlib.util.spec_from_file_location(
    "autodream_coverage", Path(__file__).resolve().parents[1] / "helpers/auto_dream.py"
)
dream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dream
spec.loader.exec_module(dream)


async def check_coverage(session_count, vector_count, failure=None, stats_write_error=False):
    with TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        now = datetime.now(timezone.utc)
        previous = now - timedelta(days=1)
        stack.enter_context(patch.object(dream, "get_autodream_root", return_value=str(root / "memory")))
        stack.enter_context(patch.object(dream.files, "get_abs_path", side_effect=lambda p: str(root / p)))
        stack.enter_context(patch.object(History, "_get_max_embeds", return_value=0))
        stack.enter_context(patch.object(dream, "get_autodream_config", return_value={
            "enabled": True, "min_sessions": 1, "consolidate_every_n_dreams": 1,
        }))
        stack.enter_context(patch.object(dream, "get_memory_plugin_config", return_value={}))
        stack.enter_context(patch.object(dream, "initialize_agent"))
        stack.enter_context(patch.object(dream.tokens, "approximate_tokens", side_effect=lambda text: len(str(text))))
        if stats_write_error:
            stack.enter_context(patch.object(dream, "record_dream_stats", side_effect=OSError("stats unavailable")))
        errors = stack.enter_context(patch.object(dream.PrintStyle, "error"))
        context = stack.enter_context(patch.object(dream, "AgentContext"))
        dream.save_auto_dream_state("default", {"last_dream_at": previous.isoformat()})
        for i in range(session_count):
            history = History(agent=None)
            history.add_message(False, f"session-{i}")
            path = root / "usr/chats" / str(i) / "chat.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "id": str(i), "type": "user", "last_message": (now - timedelta(hours=1)).isoformat(),
                "agents": [{"history": history.serialize()}],
            }))
        docs = {
            str(i): Document(page_content=f"memory-{i}", metadata={
                "id": str(i), "timestamp": (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            }) for i in range(vector_count)
        }
        docs["old"] = Document(page_content="old", metadata={
            "id": "old", "timestamp": (previous - timedelta(seconds=1)).isoformat(),
        })
        docs["generated"] = Document(page_content="generated", metadata={"autodream_source": True})
        docs["knowledge"] = Document(page_content="knowledge", metadata={"knowledge_source": True})
        db = Mock()
        db.db.get_all_docs.return_value = docs
        stack.enter_context(patch.object(Memory, "get_by_subdir", new=AsyncMock(return_value=db)))
        related = stack.enter_context(patch.object(dream, "load_related_vector_memories", new=AsyncMock(return_value=[])))
        sync = stack.enter_context(patch.object(dream, "sync_autodream_vector_memory", new=AsyncMock(
            return_value={"file_count": 0, "doc_count": 0},
            side_effect=RuntimeError("sync failed") if failure == "sync" else None,
        )))
        batches = []
        consolidations = []
        query_sessions = []
        during_run = None

        async def model(system, message, **kwargs):
            nonlocal during_run
            if system == "memory.memories_query.sys.md":
                query_sessions.append(message["message"])
                return message["message"]
            if system == "autodream.consolidate.sys.md":
                consolidations.append(message)
                if failure == "consolidation":
                    return '{"changes": "invalid"}'
                return '{"changes": []}'
            assert system == "autodream.sys.md"
            batches.append(message)
            if len(batches) == 2:
                if failure == "batch":
                    return '[]'
                if failure == "invalid_changes":
                    return '{"changes": "invalid"}'
            during_run = datetime.now(timezone.utc)
            sessions = json.loads(message["recent_sessions"])
            assert related.await_args.kwargs["queries"] == [s["first_prompt"] for s in sessions]
            if len(batches) > 1:
                assert "Batch 1" in message["current_index"]
            return json.dumps({"changes": [{
                "action": "upsert", "path": f"batch-{len(batches)}.md",
                "title": f"Batch {len(batches)}", "content": "Durable fact",
                "source_context_ids": [s["context_id"] for s in sessions],
                "source_memory_ids": [v["id"] for v in json.loads(message["recent_vector_memories"])],
            }]})

        context.return_value.agent0.read_prompt.side_effect = lambda name, **kw: kw if kw else name
        context.return_value.agent0.call_utility_model.side_effect = model
        dream._RUNNING_SUBDIRS.add("default")
        dream._TASKS["default"] = Mock()
        await dream._run_auto_dream("source", None, "", "default")
        stats = dream.load_dream_stats("default")
        if session_count and not stats_write_error:
            assert stats.get("failed_dreams", 0) == int(bool(failure))
            assert stats.get("completed_dreams", 0) == int(not failure)
            assert stats["model_calls"] == context.return_value.agent0.call_utility_model.call_count
            assert stats["input_tokens"] > 0 and stats["output_tokens"] > 0
            assert stats["last_run"]["sessions"] == session_count
            assert stats["last_run"]["duration_seconds"] >= 0
        else:
            assert stats == {}
        state = dream.load_auto_dream_state("default")
        if failure:
            assert state["last_dream_at"] == previous.isoformat()
            assert errors.call_count == 1
            if failure == "batch":
                sync.assert_not_awaited()
        elif session_count:
            assert errors.call_count == int(stats_write_error), errors.call_args
            seen_sessions = [s["context_id"] for b in batches for s in json.loads(b["recent_sessions"])]
            seen_vectors = [v["id"] for b in batches for v in json.loads(b["recent_vector_memories"])]
            assert sorted(seen_sessions) == sorted(map(str, range(session_count)))
            assert sorted(seen_vectors) == sorted(map(str, range(vector_count)))
            assert len(query_sessions) == session_count
            assert all(len(json.loads(b["recent_sessions"])) <= 8 for b in batches)
            assert all(len(json.loads(b["recent_vector_memories"])) <= 16 for b in batches)
            assert state["last_session_count"] == session_count
            assert state["last_recent_vector_count"] == vector_count
            assert len(state["last_created_files"]) == len(batches)
            assert len(consolidations) == (1 if len(batches) > 1 else 0)
            checkpoint = dream.parse_iso_datetime(state["last_dream_at"])
            assert now <= checkpoint <= during_run
            path = root / "usr/chats/0/chat.json"
            payload = json.loads(path.read_text())
            payload["last_message"] = during_run.isoformat()
            path.write_text(json.dumps(payload))
            assert [s.context_id for s in dream.load_recent_sessions("default", checkpoint)] == ["0"]
            docs["boundary"] = Document(page_content="new", metadata={
                "id": "boundary", "timestamp": checkpoint.strftime("%Y-%m-%d %H:%M:%S"),
            })
            assert [d["id"] for d in await dream.load_recent_vector_memories("default", checkpoint)] == ["boundary"]
        else:
            assert not batches
            assert state["last_dream_at"] == previous.isoformat()
        if session_count:
            context.remove.assert_called_once()
        else:
            context.remove.assert_not_called()
        assert "default" not in dream._RUNNING_SUBDIRS
        assert "default" not in dream._TASKS


async def main():
    for sessions, vectors, failure in [
        (0, 0, None), (1, 0, None), (8, 16, None), (9, 17, None),
        (17, 1, None), (1, 49, None), (9, 17, "batch"), (9, 17, "sync"),
        (9, 17, "invalid_changes"), (9, 17, "consolidation"),
    ]:
        await check_coverage(sessions, vectors, failure)
        print(f"PASS sessions={sessions}, vectors={vectors}, failure={failure}")

    await check_coverage(1, 0, stats_write_error=True)
    print("PASS statistics write failure does not change dream completion")


if __name__ == "__main__":
    asyncio.run(main())
