from helpers.api import ApiHandler, Request, Response
from helpers.projects import validate_project_name
from usr.plugins.autodream.helpers import auto_dream


class Stats(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        scope = {}
        for key in ("project_name", "agent_profile"):
            value = input.get(key, "")
            if not isinstance(value, str) or "\\" in value or "\x00" in value:
                raise ValueError("Invalid statistics scope")
            scope[key] = validate_project_name(value) if value else ""
        memory_subdir = auto_dream.resolve_memory_subdir(**scope)
        state = auto_dream.load_auto_dream_state(memory_subdir)
        return {
            "memory_scope": memory_subdir,
            "running": memory_subdir in auto_dream._RUNNING_SUBDIRS,
            "stats": auto_dream.load_dream_stats(memory_subdir),
            "last_dream_at": state.get("last_dream_at"),
            "memory_file_count": state.get("memory_file_count"),
        }
