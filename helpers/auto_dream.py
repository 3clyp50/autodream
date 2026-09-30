from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import AgentContext, AgentContextType
from helpers import files, plugins, tokens
from helpers.defer import DeferredTask, THREAD_BACKGROUND
from helpers.dirty_json import DirtyJson
from helpers.file_transfers import write_stream_atomic
from helpers.print_style import PrintStyle
from helpers import yaml as yaml_helper
from initialize import initialize_agent
from langchain_core.documents import Document


PLUGIN_NAME = "autodream"
MEMORY_PLUGIN_NAME = "_memory"

AUTO_DREAM_DIR = "autodream"
AUTO_DREAM_MEMORIES_DIR = "memories"
AUTO_DREAM_INDEX_FILE = "MEMORY.md"
AUTO_DREAM_STATE_FILE = "state.json"
AUTO_DREAM_LOG_FILE = ".dream-log.md"
AUTO_DREAM_VECTOR_STATE_FILE = "vector_state.json"

MAX_RECENT_SESSIONS = 8
MAX_SESSION_CHARS = 4000
MAX_EXISTING_MEMORY_FILES = 24
MAX_EXISTING_MEMORY_INPUT_CHARS = 60000
MAX_RECENT_VECTOR_MEMORIES = 16
MAX_RECENT_VECTOR_MEMORY_CHARS = 700
MAX_RELATED_VECTOR_MEMORIES = 12
MAX_RELATED_VECTOR_MEMORY_CHARS = 700
MAX_VECTOR_QUERY_COUNT = 8
MAX_VECTOR_QUERY_CHARS = 220
MAX_ORPHAN_CANDIDATES = 4
MAX_INDEX_PROMPT_CHARS = 6000
AUTO_DREAM_LOG_MAX_ENTRIES = 40
RELATED_VECTOR_THRESHOLD = 0.58
RELATED_VECTOR_PER_QUERY = 4
MIN_ORPHAN_OVERLAP = 0.5

_RUNNING_SUBDIRS: set[str] = set()
_RUNNING_LOCK = threading.Lock()
_TASKS: dict[str, DeferredTask] = {}


@dataclass
class DreamMemoryFile:
    file_name: str
    title: str
    description: str
    updated_at: datetime | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


@dataclass
class DreamSession:
    context_id: str
    project_name: str | None
    agent_profile: str
    created_at: datetime
    last_message_at: datetime
    first_prompt: str
    transcript: str


def schedule_auto_dream(
    context_id: str,
    project_name: str | None,
    agent_profile: str,
    memory_subdir: str,
) -> bool:
    with _RUNNING_LOCK:
        if memory_subdir in _RUNNING_SUBDIRS:
            return False
        _RUNNING_SUBDIRS.add(memory_subdir)

    task = DeferredTask(thread_name=THREAD_BACKGROUND)
    task.start_task(
        _run_auto_dream,
        context_id=context_id,
        project_name=project_name,
        agent_profile=agent_profile,
        memory_subdir=memory_subdir,
    )
    _TASKS[memory_subdir] = task
    return True


async def _run_auto_dream(
    context_id: str,
    project_name: str | None,
    agent_profile: str,
    memory_subdir: str,
) -> None:
    background_context: AgentContext | None = None
    run_stats: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        config = get_autodream_config(project_name, agent_profile)
        memory_config = get_memory_plugin_config(project_name, agent_profile)
        if not memory_config.get("memory_memorize_enabled", True):
            return
        if not config.get("enabled"):
            return

        run_started_at = datetime.now(timezone.utc)
        state = load_auto_dream_state(memory_subdir)
        last_dream_at = parse_iso_datetime(state.get("last_dream_at"))
        recent_sessions = load_recent_sessions(memory_subdir, last_dream_at)

        if not should_run_auto_dream(
            last_dream_at=last_dream_at,
            recent_session_count=len(recent_sessions),
            min_hours=coerce_min_hours(config.get("min_hours")),
            min_sessions=coerce_min_sessions(config.get("min_sessions")),
        ):
            return

        started = time.monotonic()
        run_stats = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "sessions": len(recent_sessions),
        }
        background_context = AgentContext(
            config=initialize_agent(
                {"agent_profile": agent_profile} if agent_profile else None
            ),
            name="AutoDream",
            type=AgentContextType.BACKGROUND,
        )
        if project_name:
            from helpers import projects

            projects.activate_project(
                background_context.id,
                project_name,
                mark_dirty=False,
            )

        agent = background_context.agent0

        async def call_dream_model(system: str, message: str, **kwargs):
            run_stats["model_calls"] += 1
            run_stats["input_tokens"] += tokens.approximate_tokens(system) + tokens.approximate_tokens(message)
            response = await agent.call_utility_model(system=system, message=message, **kwargs)
            run_stats["output_tokens"] += tokens.approximate_tokens(response or "")
            return response

        recent_vector_memories = await load_recent_vector_memories(
            memory_subdir=memory_subdir,
            last_dream_at=last_dream_at,
        )
        memory_scope = describe_memory_scope(memory_subdir)
        orphan_candidates = find_orphan_candidates(memory_subdir)
        results = []
        related_vector_count = 0
        batch_count = max(
            (len(recent_sessions) + MAX_RECENT_SESSIONS - 1) // MAX_RECENT_SESSIONS,
            (len(recent_vector_memories) + MAX_RECENT_VECTOR_MEMORIES - 1)
            // MAX_RECENT_VECTOR_MEMORIES,
        )
        for batch in range(batch_count):
            session_batch = recent_sessions[
                batch * MAX_RECENT_SESSIONS : (batch + 1) * MAX_RECENT_SESSIONS
            ]
            vector_batch = recent_vector_memories[
                batch * MAX_RECENT_VECTOR_MEMORIES : (batch + 1) * MAX_RECENT_VECTOR_MEMORIES
            ]
            # Summarize long transcripts safely using the standard utility prompts
            system_sum = agent.read_prompt("fw.topic_summary.sys.md")
            for session in session_batch:
                if len(session.transcript) > MAX_SESSION_CHARS:
                    msg_sum = agent.read_prompt("fw.topic_summary.msg.md", content=session.transcript)
                    summary = await call_dream_model(system=system_sum, message=msg_sum)
                    if summary:
                        session.transcript = summary.strip()

            # Generate targeted semantic queries for related vector memories
            queries: list[str] = []
            system_query = agent.read_prompt("memory.memories_query.sys.md")
            for session in session_batch[:MAX_VECTOR_QUERY_COUNT]:
                msg_query = agent.read_prompt(
                    "memory.memories_query.msg.md",
                    history=session.transcript,
                    message=session.first_prompt
                )
                q = await call_dream_model(system=system_query, message=msg_query)
                q = (q or "").strip()
                if q and q != "-":
                    queries.append(q)

            existing_files, existing_memories = prepare_memory_input(memory_subdir)
            current_index = truncate_for_prompt(
                read_memory_index(memory_subdir),
                MAX_INDEX_PROMPT_CHARS,
            )
            related_vector_memories = await load_related_vector_memories(
                memory_subdir=memory_subdir,
                queries=queries,
                existing_memory_ids={
                    str(item.get("id", "")).strip()
                    for item in vector_batch
                    if str(item.get("id", "")).strip()
                },
            )

            system = agent.read_prompt("autodream.sys.md")
            message = agent.read_prompt(
                "autodream.msg.md",
                line_limit=int(config.get("line_limit", 120) or 120),
                memory_scope=json.dumps(memory_scope, ensure_ascii=False, indent=2),
                current_index=current_index or "_No existing index_",
                existing_memories=existing_memories,
                recent_sessions=json.dumps(
                    [
                        {
                            "context_id": session.context_id,
                            "project_name": session.project_name,
                            "agent_profile": session.agent_profile,
                            "created_at": serialize_datetime(session.created_at),
                            "last_message_at": serialize_datetime(session.last_message_at),
                            "first_prompt": session.first_prompt,
                            "transcript": session.transcript,
                        }
                        for session in session_batch
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                recent_vector_memories=json.dumps(
                    vector_batch,
                    ensure_ascii=False,
                    indent=2,
                ),
                related_vector_memories=json.dumps(
                    related_vector_memories,
                    ensure_ascii=False,
                    indent=2,
                ),
                orphan_candidates=json.dumps(
                    orphan_candidates,
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            response = await call_dream_model(
                system=system,
                message=message,
                background=True,
            )
            plan = DirtyJson.parse_string((response or "").strip())
            if not isinstance(plan, dict):
                raise ValueError("AutoDream model response was not a JSON object.")

            result = await apply_auto_dream_plan(
                memory_subdir=memory_subdir,
                plan=plan,
                existing_files=existing_files,
                source_context_ids={session.context_id for session in session_batch},
                source_memory_ids={
                    str(item.get("id", ""))
                    for item in [*vector_batch, *related_vector_memories]
                    if item.get("id")
                },
                line_limit=int(config.get("line_limit", 120) or 120),
                run_metadata={
                    "memory_scope": memory_scope,
                    "orphan_candidates": orphan_candidates,
                    "recent_session_count": len(session_batch),
                    "recent_vector_count": len(vector_batch),
                    "related_vector_count": len(related_vector_memories),
                    "phase": "learn"
                },
            )

            results.append(result)
            related_vector_count += len(related_vector_memories)

        result = {
            "summary": " | ".join(item["summary"] for item in results),
            "changed": any(item["changed"] for item in results),
            "memory_file_count": results[-1]["memory_file_count"],
            **{
                key: [name for item in results for name in item[key]]
                for key in ("created_files", "updated_files", "deleted_files")
            },
        }

        # Phase 2: Consolidate / Clean
        # Run periodically to prune redundancies and explicitly merge overlapping memories
        dreams_since_consolidation = int(state.get("dreams_since_consolidation", 0)) + 1
        consolidate_every = coerce_consolidate_every(config.get("consolidate_every_n_dreams"))
        
        consolidate_result = None
        if consolidate_every > 0 and dreams_since_consolidation >= consolidate_every:
            dreams_since_consolidation = 0
            existing_files_for_consolidation, consolidation_input = prepare_memory_input(memory_subdir)
            
            if len(existing_files_for_consolidation) > 1:
                try:
                    system_consolidate = agent.read_prompt("autodream.consolidate.sys.md")
                    message_consolidate = agent.read_prompt(
                        "autodream.consolidate.msg.md",
                        existing_memories=consolidation_input,
                        memory_scope=json.dumps(memory_scope, ensure_ascii=False, indent=2)
                    )

                    response_consolidate = await call_dream_model(
                        system=system_consolidate,
                        message=message_consolidate,
                        background=True,
                    )
                    plan_consolidate = DirtyJson.parse_string((response_consolidate or "").strip())
                    consolidate_result = await apply_auto_dream_plan(
                        memory_subdir=memory_subdir,
                        plan=plan_consolidate,
                        existing_files=existing_files_for_consolidation,
                        line_limit=int(config.get("line_limit", 120) or 120),
                        run_metadata={
                            "memory_scope": memory_scope,
                            "phase": "consolidation"
                        },
                    )
                except Exception as e:
                    raise ValueError(f"AutoDream consolidation failed: {e}") from e

        vector_sync = await sync_autodream_vector_memory(memory_subdir)
        
        final_summary = result["summary"]
        changed = result["changed"]
        if consolidate_result and consolidate_result["changed"]:
            final_summary += f" | Consolidation: {consolidate_result['summary']}"
            changed = True

        save_auto_dream_state(
            memory_subdir,
            {
                "schema_version": 2,
                "last_dream_at": run_started_at.isoformat(),
                "last_status": "updated" if changed else "noop",
                "last_summary": final_summary,
                "last_session_count": len(recent_sessions),
                "last_recent_vector_count": len(recent_vector_memories),
                "last_related_vector_count": related_vector_count,
                "memory_file_count": consolidate_result["memory_file_count"] if consolidate_result else result["memory_file_count"],
                "last_created_files": result["created_files"] + (consolidate_result["created_files"] if consolidate_result else []),
                "last_updated_files": result["updated_files"] + (consolidate_result["updated_files"] if consolidate_result else []),
                "last_deleted_files": result["deleted_files"] + (consolidate_result["deleted_files"] if consolidate_result else []),
                "last_orphan_candidates": [
                    item.get("memory_subdir", "") for item in orphan_candidates
                ],
                "last_log_path": get_autodream_log_path(memory_subdir),
                "vector_file_count": vector_sync["file_count"],
                "vector_doc_count": vector_sync["doc_count"],
                "dreams_since_consolidation": dreams_since_consolidation,
            },
        )

        run_stats["status"] = "updated" if changed else "noop"

        if changed and final_summary:
            context = AgentContext.get(context_id)
            if context:
                context.log.log(
                    type="util",
                    heading="AutoDream updated durable memory",
                    content=final_summary,
                    update_progress="none",
                )
    except Exception as exc:
        PrintStyle.error(f"AutoDream failed for '{memory_subdir}': {exc}")
    finally:
        if run_stats is not None:
            run_stats["finished_at"] = datetime.now(timezone.utc).isoformat()
            run_stats["duration_seconds"] = round(time.monotonic() - started, 2)
            try:
                record_dream_stats(memory_subdir, run_stats)
            except Exception as exc:
                PrintStyle.error(f"AutoDream could not save statistics for '{memory_subdir}': {exc}")
        with _RUNNING_LOCK:
            _RUNNING_SUBDIRS.discard(memory_subdir)
        _TASKS.pop(memory_subdir, None)
        if background_context:
            AgentContext.remove(background_context.id)


def load_dream_stats(memory_subdir: str) -> dict[str, Any]:
    path = Path(get_autodream_root(memory_subdir)) / "stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_dream_stats(memory_subdir: str, run: dict[str, Any]) -> None:
    stats = load_dream_stats(memory_subdir)
    stats.setdefault("tracking_since", run["started_at"])
    counter = "failed_dreams" if run["status"] == "failed" else "completed_dreams"
    stats[counter] = stats.get(counter, 0) + 1
    for key in ("model_calls", "input_tokens", "output_tokens", "duration_seconds"):
        stats[key] = stats.get(key, 0) + run[key]
    stats["last_run"] = run
    write_stream_atomic(
        io.BytesIO(json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8")),
        Path(get_autodream_root(memory_subdir)) / "stats.json",
    )


async def apply_auto_dream_plan(
    memory_subdir: str,
    plan: dict[str, Any],
    line_limit: int,
    existing_files: list[DreamMemoryFile],
    run_metadata: dict[str, Any] | None = None,
    source_context_ids: set[str] | None = None,
    source_memory_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, dict) or not isinstance(plan.get("changes"), list):
        raise ValueError("AutoDream plan must contain a changes array.")
    if plan.keys() - {"summary", "changes"}:
        raise ValueError("Unknown AutoDream plan fields.")
    if not isinstance(plan.get("summary", ""), str):
        raise ValueError("AutoDream summary must be a string.")

    memories_dir = Path(get_autodream_memories_dir(memory_subdir))
    supplied = {item.file_name: item for item in existing_files}
    source_keys = ("source_context_ids", "source_memory_ids", "source_first_prompts")
    allowed_ids = {
        "source_context_ids": set(source_context_ids or ()),
        "source_memory_ids": set(source_memory_ids or ()),
    }
    for key, values in allowed_ids.items():
        for item in existing_files:
            values.update(normalize_string_list(item.metadata.get(key, [])))

    changes: dict[str, dict[str, Any]] = {}
    originals: dict[str, bytes] = {}
    for change in plan["changes"]:
        if not isinstance(change, dict) or change.get("action") not in ("upsert", "delete"):
            raise ValueError("AutoDream changes must be upsert or delete objects.")
        allowed_fields = {"action", "path", "reason", "replacement"} if change["action"] == "delete" else {
            "action", "path", "title", "description", "content", "grounding", "source_files", *source_keys,
        }
        if change.keys() - allowed_fields:
            raise ValueError("Unknown AutoDream change fields.")
        name = change.get("path")
        if not isinstance(name, str) or not name or normalize_memory_filename(name) != name:
            raise ValueError("AutoDream paths must be normalized Markdown file names.")
        if name in changes:
            raise ValueError(f"Duplicate AutoDream target: {name}")
        path = memories_dir / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"AutoDream target is not a regular file: {name}")
        if path.exists() and name not in supplied:
            raise ValueError(f"AutoDream was not shown the complete file: {name}")

        if change["action"] == "upsert":
            for key in ("title", "content"):
                if not isinstance(change.get(key), str) or not change[key].strip():
                    raise ValueError(f"AutoDream upsert requires nonempty {key}: {name}")
            if not isinstance(change.get("description", ""), str):
                raise ValueError(f"AutoDream description must be a string: {name}")
            if change.get("grounding", "inferred") not in ("grounded", "inferred"):
                raise ValueError(f"Invalid AutoDream grounding: {name}")
            for key in (*source_keys, "source_files"):
                values = change.get(key, [])
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or not value.strip() for value in values
                ):
                    raise ValueError(f"AutoDream {key} must contain nonempty strings: {name}")
                if key in allowed_ids and not set(values) <= allowed_ids[key]:
                    raise ValueError(f"Unknown AutoDream {key}: {name}")
            sources = set(change.get("source_files", []))
            if name in supplied:
                sources.add(name)
            if not sources <= supplied.keys():
                raise ValueError(f"AutoDream merge references an unseen source: {name}")
            if not sources and not any(change.get(key) for key in allowed_ids):
                raise ValueError(f"AutoDream upsert requires source evidence: {name}")
        else:
            if name not in supplied:
                raise ValueError(f"AutoDream cannot delete an unseen file: {name}")
            if not isinstance(change.get("reason"), str) or not change["reason"].strip():
                raise ValueError(f"AutoDream deletion requires a reason: {name}")
            if not isinstance(change.get("replacement"), str):
                raise ValueError(f"AutoDream deletion requires a replacement: {name}")
            sources = {name}

        for source in sources:
            source_path = memories_dir / source
            if source_path.is_symlink():
                raise ValueError(f"AutoDream source is a symlink: {source}")
            original = source_path.read_bytes()
            if hashlib.sha256(original).hexdigest() != supplied[source].checksum:
                raise ValueError(f"AutoDream source changed after it was read: {source}")
            originals[source] = original
        changes[name] = change

    for name, change in changes.items():
        if change["action"] == "delete":
            replacement = changes.get(change["replacement"], {})
            if replacement.get("action") != "upsert" or name not in replacement.get("source_files", []):
                raise ValueError(f"AutoDream deletion must reference a planned merge: {name}")

    # Render the entire validated plan before creating backups or changing files.
    current_scope = (run_metadata or {}).get("memory_scope", {})
    rendered_changes: dict[str, str] = {}
    for name, change in changes.items():
        if change["action"] != "upsert":
            continue
        previous = supplied.get(name)
        sources = [supplied[source] for source in dict.fromkeys(
            ([name] if previous else []) + change.get("source_files", [])
        )]
        frontmatter = dict(previous.metadata) if previous else {}
        frontmatter.update({
            "title": change["title"].strip(),
            "description": collapse_single_line(change.get("description", frontmatter.get("description", ""))),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "memory_scope": memory_subdir,
            "grounding": change.get("grounding", frontmatter.get("grounding", "inferred")),
        })
        if any(item.metadata.get("grounding") == "inferred" for item in sources):
            frontmatter["grounding"] = "inferred"
        for key in source_keys:
            values = [value for item in sources for value in normalize_string_list(item.metadata.get(key, []))]
            values.extend(change.get(key, []))
            if key == "source_first_prompts":
                values = [normalize_source_prompt_snippet(value) for value in values]
            if values:
                frontmatter[key] = normalize_string_list(values)
        source_files = [
            source for item in sources
            for source in normalize_string_list(item.metadata.get("source_files", []))
        ]
        source_files.extend(item.file_name for item in sources if item.file_name != name)
        if source_files:
            frontmatter["source_files"] = normalize_string_list(source_files)
        for key in ("canonical_scope_name", "project_title"):
            value = current_scope.get("canonical_name" if key == "canonical_scope_name" else key)
            if value:
                frontmatter[key] = collapse_single_line(value)
        rendered_changes[name] = (
            "---\n" + yaml_helper.dumps(frontmatter).strip() + "\n---\n\n"
            + format_memory_body(frontmatter["title"], change["content"])
        )

    # Back up every original before the first mutation. Archives are not active memories.
    for name in changes.keys() & originals.keys():
        original = originals[name]
        archive = Path(get_autodream_root(memory_subdir)) / "archive" / name / (
            hashlib.sha256(original).hexdigest() + ".md"
        )
        write_stream_atomic(io.BytesIO(original), archive)

    created_files: list[str] = []
    updated_files: list[str] = []
    deleted_files: list[str] = []
    for name, rendered in rendered_changes.items():
        write_stream_atomic(io.BytesIO(rendered.encode("utf-8")), memories_dir / name)
        (updated_files if name in originals else created_files).append(name)
    # Publish replacements before removing their sources, regardless of model order.
    for name, change in changes.items():
        if change["action"] == "delete":
            (memories_dir / name).unlink()
            deleted_files.append(name)

    summary = plan.get("summary", "").strip()
    memory_files = load_existing_memory_files(memory_subdir)
    memory_index = render_memory_index(memory_files, line_limit=line_limit)
    write_stream_atomic(
        io.BytesIO(memory_index.encode("utf-8")), get_autodream_index_path(memory_subdir)
    )

    changed = bool(created_files or updated_files or deleted_files)
    if not summary:
        if changed:
            summary = build_change_summary(created_files, updated_files, deleted_files)
        else:
            summary = "AutoDream found no durable memory changes to apply."

    append_auto_dream_log(
        memory_subdir=memory_subdir,
        summary=summary,
        created_files=created_files,
        updated_files=updated_files,
        deleted_files=deleted_files,
        run_metadata=run_metadata or {},
    )

    return {
        "summary": summary,
        "changed": changed,
        "memory_file_count": len(memory_files),
        "created_files": created_files,
        "updated_files": updated_files,
        "deleted_files": deleted_files,
    }


async def sync_autodream_vector_memory(memory_subdir: str) -> dict[str, int]:
    from plugins._memory.helpers.memory import Memory

    db = await Memory.get_by_subdir(memory_subdir, preload_knowledge=False)
    vector_state = load_autodream_vector_state(memory_subdir)
    file_map = (
        vector_state.get("files", {}) if isinstance(vector_state.get("files"), dict) else {}
    )

    if not vector_state.get("initialized"):
        stale_ids = [
            str(doc.metadata.get("id", "")).strip()
            for doc in db.db.get_all_docs().values()
            if (doc.metadata or {}).get("autodream_source")
            and str(doc.metadata.get("id", "")).strip()
        ]
        if stale_ids:
            await db.delete_documents_by_ids(stale_ids)
        file_map = {}

    current_paths = {
        path.name: path
        for path in Path(get_autodream_memories_dir(memory_subdir)).glob("*.md")
        if path.is_file()
    }

    for file_name in list(file_map):
        if file_name in current_paths:
            continue
        old_ids = normalize_string_list(file_map[file_name].get("ids", []))
        if old_ids:
            await db.delete_documents_by_ids(old_ids)
        del file_map[file_name]

    for file_name, file_path in current_paths.items():
        text = file_path.read_text(encoding="utf-8")
        checksum = hashlib.md5(text.encode("utf-8")).hexdigest()
        tracked = file_map.get(file_name, {})
        tracked_ids = normalize_string_list(tracked.get("ids", []))
        if tracked.get("checksum") == checksum and tracked_ids:
            continue

        if tracked_ids:
            await db.delete_documents_by_ids(tracked_ids)

        frontmatter, body = parse_frontmatter(text)
        inserted_ids = await db.insert_documents(
            [build_autodream_document(memory_subdir, file_path, frontmatter, body)]
        )
        file_map[file_name] = {
            "checksum": checksum,
            "ids": inserted_ids,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    save_autodream_vector_state(
        memory_subdir,
        {
            "schema_version": 1,
            "initialized": True,
            "files": file_map,
        },
    )

    return {
        "file_count": len(file_map),
        "doc_count": sum(
            len(normalize_string_list(entry.get("ids", []))) for entry in file_map.values()
        ),
    }


async def load_recent_vector_memories(
    memory_subdir: str,
    last_dream_at: datetime | None,
) -> list[dict[str, Any]]:
    from plugins._memory.helpers.memory import Memory

    db = await Memory.get_by_subdir(memory_subdir, preload_knowledge=False)
    docs = list(db.db.get_all_docs().values())
    recent: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        if metadata.get("autodream_source"):
            continue
        if metadata.get("knowledge_source"):
            continue

        timestamp = parse_memory_timestamp(metadata.get("timestamp", ""))
        # Vector timestamps have second precision; replay the boundary second.
        if last_dream_at and timestamp and timestamp < last_dream_at.replace(microsecond=0):
            continue

        recent.append(
            {
                "id": metadata.get("id", ""),
                "area": metadata.get("area", ""),
                "timestamp": metadata.get("timestamp", ""),
                "source_file": metadata.get("source_file", ""),
                "content": truncate_for_prompt(
                    str(doc.page_content or ""), MAX_RECENT_VECTOR_MEMORY_CHARS
                ),
            }
        )

    recent.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return recent


async def load_related_vector_memories(
    memory_subdir: str,
    queries: list[str],
    existing_memory_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not queries:
        return []

    from plugins._memory.helpers.memory import Memory

    db = await Memory.get_by_subdir(memory_subdir, preload_knowledge=False)
    related: list[dict[str, Any]] = []
    seen_ids = set(existing_memory_ids or [])

    for query in queries:
        docs = await db.search_similarity_threshold(
            query=query,
            limit=RELATED_VECTOR_PER_QUERY,
            threshold=RELATED_VECTOR_THRESHOLD,
        )
        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("autodream_source") or metadata.get("knowledge_source"):
                continue

            doc_id = str(metadata.get("id", "") or "").strip()
            if not doc_id or doc_id in seen_ids:
                continue

            seen_ids.add(doc_id)
            related.append(
                {
                    "id": doc_id,
                    "area": metadata.get("area", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "matched_query": query,
                    "source_file": metadata.get("source_file", ""),
                    "content": truncate_for_prompt(
                        str(doc.page_content or ""),
                        MAX_RELATED_VECTOR_MEMORY_CHARS,
                    ),
                }
            )
            if len(related) >= MAX_RELATED_VECTOR_MEMORIES:
                return related

    return related


def coerce_consolidate_every(raw: Any) -> int:
    if raw is None:
        return 3
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return max(0, value)


def coerce_min_hours(raw: Any) -> float:
    if raw is None:
        return 8.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 8.0
    return max(0.0, value)


def coerce_min_sessions(raw: Any) -> int:
    if raw is None:
        return 3
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return max(0, value)


def should_run_auto_dream(
    last_dream_at: datetime | None,
    recent_session_count: int,
    min_hours: float,
    min_sessions: int,
) -> bool:
    if recent_session_count <= 0:
        return False
    if last_dream_at is None:
        return True

    hours_since = (datetime.now(timezone.utc) - last_dream_at).total_seconds() / 3600
    if min_sessions > 0 and recent_session_count >= min_sessions:
        return True
    if min_hours > 0 and hours_since >= min_hours:
        return True
    return False


def load_recent_sessions(
    target_memory_subdir: str,
    last_dream_at: datetime | None,
) -> list[DreamSession]:
    chats_root = Path(files.get_abs_path("usr/chats"))
    if not chats_root.exists():
        return []

    sessions: list[DreamSession] = []
    for chat_file in chats_root.glob("*/chat.json"):
        try:
            payload = json.loads(chat_file.read_text(encoding="utf-8"))
            if str(payload.get("type", "")).lower() != AgentContextType.USER.value:
                continue

            context_data = payload.get("data", {}) or {}
            project_name = context_data.get("project")
            agent_profile = str(context_data.get("agent_profile", "") or "")
            if (
                resolve_memory_subdir(
                    project_name=project_name,
                    agent_profile=agent_profile,
                )
                != target_memory_subdir
            ):
                continue

            last_message_at = parse_iso_datetime(payload.get("last_message"))
            if last_message_at is None:
                continue
            if last_dream_at and last_message_at <= last_dream_at:
                continue

            created_at = parse_iso_datetime(payload.get("created_at")) or last_message_at

            history_json = payload.get("agents", [{}])[0].get("history", "")
            if not history_json:
                continue

            from helpers.history import deserialize_history, output_text
            hist = deserialize_history(history_json, agent=None)
            outputs = hist.output()
            
            first_prompt = ""
            for out in outputs:
                if not out["ai"]:
                    first_prompt = str(out.get("content", ""))
                    break
            
            transcript = truncate_for_prompt(
                output_text(outputs),
                MAX_SESSION_CHARS * 15,
            )
            if not first_prompt and not transcript:
                continue

            sessions.append(
                DreamSession(
                    context_id=str(payload.get("id", "") or chat_file.parent.name),
                    project_name=project_name,
                    agent_profile=agent_profile,
                    created_at=created_at,
                    last_message_at=last_message_at,
                    first_prompt=first_prompt,
                    transcript=transcript,
                )
            )
        except Exception:
            continue

    sessions.sort(key=lambda item: item.last_message_at)
    return sessions


def load_existing_memory_files(memory_subdir: str) -> list[DreamMemoryFile]:
    memories_dir = Path(get_autodream_memories_dir(memory_subdir))
    if not memories_dir.exists():
        return []

    files_out: list[DreamMemoryFile] = []
    for path in memories_dir.glob("*.md"):
        try:
            if path.is_symlink():
                continue
            original = path.read_bytes()
            meta, body = parse_frontmatter(original.decode("utf-8"))
            title = str(meta.get("title", "") or path.stem).strip() or path.stem
            description = collapse_single_line(meta.get("description", ""))
            updated_at = parse_iso_datetime(meta.get("updated_at"))
            files_out.append(
                DreamMemoryFile(
                    file_name=path.name,
                    title=title,
                    description=description,
                    updated_at=updated_at,
                    content=body.strip(),
                    metadata=meta,
                    checksum=hashlib.sha256(original).hexdigest(),
                )
            )
        except Exception:
            continue

    files_out.sort(
        key=lambda item: (
            serialize_datetime(item.updated_at) if item.updated_at else "",
            item.title.lower(),
        ),
        reverse=True,
    )
    return files_out


def prepare_memory_input(memory_subdir: str) -> tuple[list[DreamMemoryFile], str]:
    selected: list[DreamMemoryFile] = []
    entries: list[str] = []
    remaining = MAX_EXISTING_MEMORY_INPUT_CHARS - 2
    for item in load_existing_memory_files(memory_subdir):
        entry = json.dumps({
            "path": item.file_name,
            "title": item.title,
            "description": item.description,
            "updated_at": serialize_datetime(item.updated_at),
            "metadata": item.metadata,
            "content": item.content,
        }, ensure_ascii=False, default=str)
        cost = len(entry) + (2 if entries else 0)
        if cost > remaining:
            continue
        selected.append(item)
        entries.append(entry)
        remaining -= cost
        if len(selected) >= MAX_EXISTING_MEMORY_FILES:
            break
    return selected, "[" + ",\n".join(entries) + "]"


def describe_memory_scope(memory_subdir: str) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "memory_subdir": memory_subdir,
        "autodream_root": get_autodream_root(memory_subdir),
        "canonical_name": memory_subdir.rsplit("/", 1)[-1] or memory_subdir or "default",
        "canonical_slug": slugify_text(memory_subdir.rsplit("/", 1)[-1] or memory_subdir or "default"),
    }

    if not memory_subdir.startswith("projects/"):
        return scope

    project_name = memory_subdir[9:]
    scope["project_name"] = project_name
    scope["canonical_name"] = project_name
    scope["canonical_slug"] = slugify_text(project_name)

    try:
        from helpers import projects

        scope["project_folder"] = projects.get_project_folder(project_name)
        project_data = projects.load_basic_project_data(project_name)
        project_title = collapse_single_line(project_data.get("title", ""))
        project_description = collapse_single_line(project_data.get("description", ""))
        if project_title:
            scope["project_title"] = project_title
        if project_description:
            scope["project_description"] = truncate_for_prompt(project_description, 200)
    except Exception:
        pass

    return scope


def find_orphan_candidates(memory_subdir: str) -> list[dict[str, Any]]:
    if not memory_subdir.startswith("projects/"):
        return []

    current_project_name = memory_subdir[9:]
    current_tokens = slug_tokens(current_project_name)
    if not current_tokens:
        return []

    try:
        from helpers import projects

        projects_root = Path(projects.get_projects_parent_folder())
    except Exception:
        return []

    if not projects_root.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue

        sibling_project_name = project_dir.name
        if sibling_project_name == current_project_name:
            continue

        sibling_tokens = slug_tokens(sibling_project_name)
        overlap = score_token_overlap(current_tokens, sibling_tokens)
        if overlap < MIN_ORPHAN_OVERLAP:
            continue

        memories_dir = Path(
            projects.get_project_meta(
                sibling_project_name,
                "memory",
                AUTO_DREAM_DIR,
                AUTO_DREAM_MEMORIES_DIR,
            )
        )
        if not memories_dir.exists():
            continue

        memory_files = [path for path in memories_dir.glob("*.md") if path.is_file()]
        if not memory_files:
            continue

        latest_update = max((path.stat().st_mtime for path in memory_files), default=0.0)
        candidates.append(
            {
                "memory_subdir": f"projects/{sibling_project_name}",
                "project_name": sibling_project_name,
                "autodream_root": str(memories_dir.parent),
                "memory_file_count": len(memory_files),
                "overlap_score": round(overlap, 2),
                "shared_tokens": sorted(current_tokens & sibling_tokens),
                "last_updated_at": (
                    datetime.fromtimestamp(latest_update, tz=timezone.utc).isoformat()
                    if latest_update
                    else ""
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item.get("overlap_score", 0),
            item.get("last_updated_at", ""),
        ),
        reverse=True,
    )
    return candidates[:MAX_ORPHAN_CANDIDATES]


def render_memory_index(
    memory_files: list[DreamMemoryFile],
    line_limit: int,
) -> str:
    lines = ["# Memory Index", ""]
    if not memory_files:
        lines.append("No durable memories indexed yet.")
        return "\n".join(lines) + "\n"

    max_entries = max(1, line_limit - 3)
    visible_entries = memory_files[:max_entries]
    hidden_entries = len(memory_files) - len(visible_entries)

    for item in visible_entries:
        title = collapse_single_line(item.title)
        description = collapse_single_line(item.description) or "Durable memory"
        lines.append(f"- [{title}](memories/{item.file_name}): {description}")

    if hidden_entries > 0:
        lines.append(
            f"- Additional memories hidden to respect the line limit: {hidden_entries}"
        )

    return "\n".join(lines) + "\n"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        end_index = -1
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = i
                break
        if end_index > 0:
            meta_text = "\n".join(lines[1:end_index])
            body_text = "\n".join(lines[end_index + 1 :]).strip()
            meta = yaml_helper.loads(meta_text) or {}
            if isinstance(meta, dict):
                return meta, body_text
    return {}, text.strip()


def save_auto_dream_state(memory_subdir: str, state: dict[str, Any]) -> None:
    Path(get_autodream_state_path(memory_subdir)).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(get_autodream_state_path(memory_subdir)).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_auto_dream_state(memory_subdir: str) -> dict[str, Any]:
    path = Path(get_autodream_state_path(memory_subdir))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_memory_index(memory_subdir: str) -> str:
    path = Path(get_autodream_index_path(memory_subdir))
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def append_auto_dream_log(
    memory_subdir: str,
    summary: str,
    created_files: list[str],
    updated_files: list[str],
    deleted_files: list[str],
    run_metadata: dict[str, Any],
) -> None:
    path = Path(get_autodream_log_path(memory_subdir))
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = render_auto_dream_log_entry(
        summary=summary,
        created_files=created_files,
        updated_files=updated_files,
        deleted_files=deleted_files,
        run_metadata=run_metadata,
    )

    existing_entries = parse_auto_dream_log_entries(
        path.read_text(encoding="utf-8") if path.exists() else ""
    )
    merged_entries = [entry, *existing_entries][:AUTO_DREAM_LOG_MAX_ENTRIES]
    header = "# AutoDream Log\n\nNewest runs first.\n\n"
    path.write_text(header + "\n\n".join(merged_entries).rstrip() + "\n", encoding="utf-8")


def render_auto_dream_log_entry(
    summary: str,
    created_files: list[str],
    updated_files: list[str],
    deleted_files: list[str],
    run_metadata: dict[str, Any],
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"## {timestamp}"]
    if summary:
        lines.append(f"- Summary: {collapse_single_line(summary)}")

    scope = run_metadata.get("memory_scope", {}) if isinstance(run_metadata, dict) else {}
    scope_name = collapse_single_line(scope.get("canonical_name", ""))
    if scope_name:
        lines.append(f"- Scope: {scope_name} ({scope.get('memory_subdir', '')})")

    phase = str(run_metadata.get("phase", "learn")).capitalize()
    lines.append(f"- Phase: {phase}")

    if phase == "Learn":
        lines.append(
            "- Inputs: "
            + ", ".join(
                [
                    f"{int(run_metadata.get('recent_session_count', 0) or 0)} sessions",
                    f"{int(run_metadata.get('recent_vector_count', 0) or 0)} recent vector memories",
                    f"{int(run_metadata.get('related_vector_count', 0) or 0)} related vector memories",
                ]
            )
        )

    lines.extend(render_log_file_line("Created", created_files))
    lines.extend(render_log_file_line("Updated", updated_files))
    lines.extend(render_log_file_line("Pruned", deleted_files))

    orphan_candidates = run_metadata.get("orphan_candidates", [])
    if isinstance(orphan_candidates, list) and orphan_candidates:
        parts = []
        for item in orphan_candidates[:MAX_ORPHAN_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("memory_subdir", "") or item.get("project_name", "")).strip()
            if not label:
                continue
            count = int(item.get("memory_file_count", 0) or 0)
            parts.append(f"{label} ({count} files)")
        if parts:
            lines.append(f"- Rename / orphan hints: {', '.join(parts)}")

    return "\n".join(lines).strip()


def parse_auto_dream_log_entries(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    entries: list[str] = []
    current: list[str] = []
    for line in stripped.splitlines():
        if line.startswith("# AutoDream Log"):
            continue
        if line.strip() == "Newest runs first.":
            continue
        if line.startswith("## "):
            if current:
                entries.append("\n".join(current).strip())
            current = [line]
            continue
        if current:
            current.append(line)

    if current:
        entries.append("\n".join(current).strip())
    return [entry for entry in entries if entry]


def render_log_file_line(label: str, file_names: list[str]) -> list[str]:
    if not file_names:
        return [f"- {label}: none"]
    return [f"- {label}: {', '.join(file_names)}"]


def build_autodream_document(
    memory_subdir: str,
    file_path: Path,
    frontmatter: dict[str, Any],
    body: str,
) -> Document:
    from plugins._memory.helpers.memory import Memory

    title = collapse_single_line(frontmatter.get("title", "")) or file_path.stem
    description = collapse_single_line(frontmatter.get("description", ""))
    content = body.strip() or f"# {title}\n"
    metadata: dict[str, Any] = {
        "area": Memory.Area.MAIN.value,
        "knowledge_source": False,
        "autodream_source": True,
        "autodream_file": True,
        "memory_scope": memory_subdir,
        "source_file": file_path.name,
        "source_path": str(file_path),
        "file_type": file_path.suffix.lstrip(".").lower(),
        "title": title,
    }
    if description:
        metadata["description"] = description

    for key in ["updated_at", "grounding", "canonical_scope_name", "project_title"]:
        value = collapse_single_line(frontmatter.get(key, ""))
        if value:
            metadata[key] = value

    for key in ["source_context_ids", "source_first_prompts", "source_memory_ids"]:
        values = normalize_string_list(frontmatter.get(key, []))
        if values:
            metadata[key] = values

    return Document(page_content=content, metadata=metadata)


def load_autodream_vector_state(memory_subdir: str) -> dict[str, Any]:
    path = Path(get_autodream_vector_state_path(memory_subdir))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_autodream_vector_state(memory_subdir: str, state: dict[str, Any]) -> None:
    path = Path(get_autodream_vector_state_path(memory_subdir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_autodream_config(
    project_name: str | None,
    agent_profile: str,
) -> dict[str, Any]:
    return (
        plugins.get_plugin_config(
            PLUGIN_NAME,
            project_name=project_name or "",
            agent_profile=agent_profile or "",
        )
        or {}
    )


def get_memory_plugin_config(
    project_name: str | None,
    agent_profile: str,
) -> dict[str, Any]:
    return (
        plugins.get_plugin_config(
            MEMORY_PLUGIN_NAME,
            project_name=project_name or "",
            agent_profile=agent_profile or "",
        )
        or {}
    )


def resolve_memory_subdir(project_name: str | None, agent_profile: str) -> str:
    config = get_memory_plugin_config(project_name, agent_profile)
    if project_name and config.get("project_memory_isolation", True):
        return f"projects/{project_name}"
    return config.get("agent_memory_subdir", "") or "default"


def get_autodream_root(memory_subdir: str) -> str:
    from plugins._memory.helpers.memory import abs_db_dir
    return files.get_abs_path(abs_db_dir(memory_subdir), AUTO_DREAM_DIR)


def get_autodream_memories_dir(memory_subdir: str) -> str:
    return str(Path(get_autodream_root(memory_subdir)) / AUTO_DREAM_MEMORIES_DIR)


def get_autodream_index_path(memory_subdir: str) -> str:
    return str(Path(get_autodream_root(memory_subdir)) / AUTO_DREAM_INDEX_FILE)


def get_autodream_state_path(memory_subdir: str) -> str:
    return str(Path(get_autodream_root(memory_subdir)) / AUTO_DREAM_STATE_FILE)


def get_autodream_log_path(memory_subdir: str) -> str:
    return str(Path(get_autodream_root(memory_subdir)) / AUTO_DREAM_LOG_FILE)


def get_autodream_vector_state_path(memory_subdir: str) -> str:
    return str(Path(get_autodream_root(memory_subdir)) / AUTO_DREAM_VECTOR_STATE_FILE)


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_memory_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return parse_iso_datetime(value)


def serialize_datetime(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat() if value else ""


def truncate_for_prompt(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head - 9
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()


def normalize_memory_filename(value: str) -> str:
    safe = files.safe_file_name(
        (value or "").replace("\\", "/").split("/")[-1].strip()
    )
    safe = safe.lower().strip(" ._") or "memory"
    if not safe.endswith(".md"):
        safe += ".md"
    return safe


def truncate_single_line(value: Any, max_chars: int) -> str:
    text = collapse_single_line(value)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()


def slugify_text(value: Any) -> str:
    normalized = normalize_memory_filename(str(value or "memory"))
    if normalized.endswith(".promptinclude.md"):
        return normalized[:-17]
    if normalized.endswith(".md"):
        return normalized[:-3]
    return normalized


def slug_tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1}


def score_token_overlap(current_tokens: set[str], sibling_tokens: set[str]) -> float:
    if not current_tokens or not sibling_tokens:
        return 0.0
    return len(current_tokens & sibling_tokens) / max(
        len(current_tokens),
        len(sibling_tokens),
    )


def normalize_source_prompt_snippet(text: str) -> str:
    """Strip JSON envelopes some models return instead of plain first-user text."""
    raw = str(text or "").strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                user_msg = parsed.get("user_message")
                if isinstance(user_msg, str) and user_msg.strip():
                    return collapse_single_line(user_msg)
        except Exception:
            pass
    return collapse_single_line(raw)


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = collapse_single_line(item)
        if text and text not in result:
            result.append(text)
    return result


def collapse_single_line(value: Any) -> str:
    return collapse_whitespace(value).replace("\n", " ")


def collapse_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def format_memory_body(title: str, content: str) -> str:
    stripped = str(content or "").strip()
    if not stripped:
        return f"# {title}\n"
    if stripped.startswith("#"):
        return stripped + ("\n" if not stripped.endswith("\n") else "")
    return f"# {title}\n\n{stripped}\n"


def build_change_summary(
    created_files: list[str],
    updated_files: list[str],
    deleted_files: list[str],
) -> str:
    parts: list[str] = []
    if created_files:
        parts.append(f"created {len(created_files)}")
    if updated_files:
        parts.append(f"updated {len(updated_files)}")
    if deleted_files:
        parts.append(f"pruned {len(deleted_files)}")
    if not parts:
        return "AutoDream found no durable memory changes to apply."
    return "AutoDream " + ", ".join(parts) + " durable memory files."
