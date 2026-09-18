"""Run with the framework Python: PYTHONPATH=. python usr/plugins/autodream/tests/test_dream_safety.py"""
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_dream_coverage import dream


def write_memory(root, name, content, **metadata):
    path = root / "memories" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"title": name, "description": "test", **metadata}
    path.write_text("---\n" + dream.yaml_helper.dumps(meta) + "---\n\n" + content, encoding="utf-8")
    return path


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


async def main():
    with TemporaryDirectory() as directory, patch.object(dream, "get_autodream_root", return_value=directory):
        root = Path(directory)
        detail = "Critical detail in the middle: never delete the raw observations."
        a = write_memory(root, "a.md", "start " * 600 + detail + " end" * 800,
                         source_context_ids=["chat-a"], source_first_prompts=["Keep observations"],
                         grounding="inferred", created_at="2020-01-01")
        b = write_memory(root, "b.md", "A different useful fact.",
                         source_memory_ids=[f"vector-{i}" for i in range(20)], grounding="grounded")
        write_memory(root, "hidden.md", "Not supplied to this pass")
        supplied, rendered = dream.prepare_memory_input("default")
        assert len(rendered) <= dream.MAX_EXISTING_MEMORY_INPUT_CHARS
        assert detail in rendered
        assert '"source_context_ids": ["chat-a"]' in rendered
        assert all(item.content == json.loads(rendered)[i]["content"] for i, item in enumerate(supplied))
        supplied = [item for item in supplied if item.file_name != "hidden.md"]
        with patch.object(dream, "MAX_EXISTING_MEMORY_INPUT_CHARS", 1000):
            small, _ = dream.prepare_memory_input("default")
            assert "a.md" not in {item.file_name for item in small}
        huge = write_memory(root, "huge.md", "x" * (dream.MAX_EXISTING_MEMORY_INPUT_CHARS + 1))
        selected, _ = dream.prepare_memory_input("default")
        assert huge.name not in {item.file_name for item in selected}
        print("PASS whole-file input, metadata, and oversized-file deferral")

        valid = {"action": "upsert", "path": "a.md", "title": "Updated", "content": detail}
        invalid_plans = [
            {}, {"changes": "not an array"}, {"changes": [None]},
            {"changes": [{**valid, "action": "erase"}]},
            {"changes": [{**valid, "path": "../outside.md"}]},
            {"changes": [{**valid, "path": "hidden.md"}]},
            {"changes": [{**valid, "path": "huge.md"}]},
            {"changes": [valid, valid]},
            {"changes": [valid, {**valid, "path": "b.md", "content": ""}]},
            {"changes": [{**valid, "source_context_ids": ["invented"]}]},
            {"changes": [{**valid, "source_memory_ids": ["invented"]}]},
            {"changes": [{**valid, "source_context_ids": "chat-a"}]},
            {"changes": [{**valid, "source_files": ["hidden.md"]}]},
            {"changes": [{**valid, "source_files": ["missing.md"]}]},
            {"changes": [{**valid, "grounding": "certain"}]},
            {"changes": [{**valid, "sourceFiles": ["a.md"]}]},
            {"changes": [{"action": "delete", "path": "b.md", "reason": "redundant"}]},
            {"changes": [{"action": "delete", "path": "b.md", "reason": "redundant", "replacement": "a.md"}]},
            {"changes": [valid, {"action": "delete", "path": "b.md", "reason": "redundant", "replacement": "a.md"}]},
            {"changes": [{**valid, "path": "new.md"}]},
        ]
        before = snapshot(root)
        for plan in invalid_plans:
            try:
                await dream.apply_auto_dream_plan("default", plan, 120, supplied)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid plan: {plan}")
            assert snapshot(root) == before, plan
        print(f"PASS {len(invalid_plans)} rejected plans leave all files unchanged")

        original_a = a.read_bytes()
        a.write_bytes(original_a + b"\nConcurrent edit")
        edited = snapshot(root)
        try:
            await dream.apply_auto_dream_plan("default", {"changes": [valid]}, 120, supplied)
        except ValueError as exc:
            assert "changed" in str(exc)
        else:
            raise AssertionError("Accepted a stale snapshot")
        assert snapshot(root) == edited
        a.write_bytes(original_a)
        link = root / "memories/link.md"
        link.symlink_to(a)
        assert "link.md" not in {item.file_name for item in dream.load_existing_memory_files("default")}
        try:
            await dream.apply_auto_dream_plan("default", {"changes": [{**valid, "path": "link.md"}]}, 120, supplied)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a symlink target")
        link.unlink()
        print("PASS stale snapshots and symlinks rejected")

        merged = {"action": "upsert", "path": "merged.md", "title": "Merged",
                  "content": detail + "\nA different useful fact.",
                  "source_files": ["a.md", "b.md"], "grounding": "grounded"}
        deletes = [{"action": "delete", "path": name, "reason": "Merged", "replacement": "merged.md"}
                   for name in ("a.md", "b.md")]
        plan = {"changes": [*deletes, merged]}
        original_b = b.read_bytes()
        real_write = dream.write_stream_atomic
        # An archive failure must happen before any active-file mutation.
        with patch.object(dream, "write_stream_atomic", side_effect=OSError("archive unavailable")):
            try:
                await dream.apply_auto_dream_plan("default", plan, 120, supplied)
            except OSError:
                pass
            else:
                raise AssertionError("Archive failure was ignored")
        assert a.read_bytes() == original_a and b.read_bytes() == original_b

        def fail_replacement(source, target, **kwargs):
            if Path(target).name == "merged.md":
                raise OSError("replacement failed")
            return real_write(source, target, **kwargs)

        with patch.object(dream, "write_stream_atomic", side_effect=fail_replacement):
            try:
                await dream.apply_auto_dream_plan("default", {"changes": [
                    {**merged, "path": "partial.md"}, *plan["changes"],
                ]}, 120, supplied)
            except OSError:
                pass
            else:
                raise AssertionError("Replacement failure was ignored")
        assert (root / "memories/partial.md").is_file()
        (root / "memories/partial.md").unlink()
        assert a.read_bytes() == original_a and b.read_bytes() == original_b
        for name, original in [("a.md", original_a), ("b.md", original_b)]:
            archive = root / "archive" / name / (hashlib.sha256(original).hexdigest() + ".md")
            assert archive.read_bytes() == original
        print("PASS archive/partial-write failures preserve originals and recoverable copies")

        result = await dream.apply_auto_dream_plan("default", plan, 120, supplied)
        assert sorted(result["deleted_files"]) == ["a.md", "b.md"]
        assert not a.exists() and not b.exists()
        current = root / "memories/merged.md"
        metadata, body = dream.parse_frontmatter(current.read_text())
        assert detail in body
        assert metadata["source_files"] == ["a.md", "b.md"]
        assert metadata["source_context_ids"] == ["chat-a"]
        assert len(metadata["source_memory_ids"]) == 20
        assert metadata["grounding"] == "inferred"
        assert metadata["source_first_prompts"] == ["Keep observations"]
        assert len([path for path in (root / "archive").rglob("*.md") if path.is_file()]) == 2
        loaded = dream.load_existing_memory_files("default")
        assert not {"a.md", "b.md"} & {item.file_name for item in loaded}
        assert "archive/" not in dream.read_memory_index("default")
        print("PASS reversed-order merge keeps complete content and uncapped provenance")

        for turn in range(3):
            supplied, _ = dream.prepare_memory_input("default")
            await dream.apply_auto_dream_plan("default", {"changes": [{
                "action": "upsert", "path": "merged.md", "title": "Merged",
                "content": body + f"\nUpdate {turn}", "grounding": "grounded",
            }]}, 120, supplied)
            meta, _ = dream.parse_frontmatter(current.read_text())
            assert meta["source_memory_ids"] == metadata["source_memory_ids"]
            assert meta["source_context_ids"] == metadata["source_context_ids"]
            assert meta["grounding"] == "inferred"
            assert meta["source_files"] == ["a.md", "b.md"]
        print("PASS provenance survives three rewrites without promotion to grounded")

        # Explicit paths remain stable; title changes never create title-suffixed duplicates.
        supplied, _ = dream.prepare_memory_input("default")
        new = {"action": "upsert", "path": "user-preference.promptinclude.md", "title": "Preference",
               "content": "Explicit user preference", "source_context_ids": ["new-chat"]}
        await dream.apply_auto_dream_plan("default", {"changes": [new]}, 120, supplied,
                                          source_context_ids={"new-chat"})
        supplied, _ = dream.prepare_memory_input("default")
        await dream.apply_auto_dream_plan("default", {"changes": [{**new, "title": "Renamed title"}]},
                                          120, supplied, source_context_ids={"new-chat"})
        assert (root / "memories/user-preference.promptinclude.md").is_file()
        assert not (root / "memories/renamed_title.md").exists()
        assert len(list((root / "memories").glob("*.promptinclude.md"))) == 1
        print("PASS validated current evidence and stable explicit paths")

        supplied, _ = dream.prepare_memory_input("default")
        result = await dream.apply_auto_dream_plan("default", {"changes": []}, 120, supplied)
        assert result["changed"] is False
        print("PASS explicit empty plan remains a valid no-op")


if __name__ == "__main__":
    asyncio.run(main())
