# AutoDream

AutoDream is a plugin that depends on the builtin `_memory` plugin of Agent Zero.

It performs a periodic reflective pass over recent sessions and recent memory fragments, consolidates what looks durable, writes scoped markdown memory files, regenerates a compact Claude Code-style `MEMORY.md` index, and syncs those durable files back into vector recall.

## Dependency Model

- `_memory` remains the source of vector storage, similarity search, and memory-subdirectory resolution.
- `AutoDream` owns the reflective pass, prompt contract, durable markdown files, `MEMORY.md` generation, and dream changelog.
- AutoDream keeps its code and settings under `usr/plugins/AutoDream/`, but stores durable memory inside each `_memory` scope so project-isolated memory continues to work as expected.

## Files Written

For each memory scope, AutoDream writes:

- `autodream/MEMORY.md` as a compact index
- `autodream/memories/*.md` as durable memory files
- `autodream/.dream-log.md` as a short changelog of each dream run
- `autodream/state.json` and `autodream/vector_state.json` as plugin bookkeeping

## Claude Code `MEMORY.md` Compatibility

AutoDream follows the same broad pattern as Claude Code's `MEMORY.md`, but per memory scope instead of assuming one global repository file.

- `MEMORY.md` is an index, not the memory itself.
- The real durable content lives in the sibling markdown files under `autodream/memories/`.
- Those durable files are synced into the `_memory` vector database so symbolic browsing and semantic recall stay aligned.

## Settings

AutoDream is off by default. Enable it in Settings and tune:

- `enabled`
- `min_hours`
- `min_sessions`
- `line_limit`
