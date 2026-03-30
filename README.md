# AutoDream

AutoDream is a plugin that depends on the builtin `_memory` plugin of Agent Zero.

Agent Zero's _memory plugin stores everything: every fragment, every session, but never forgets, never prioritises. Over time the vector database fills with duplicate fragments, stale context, and noise that dilutes semantic recall.

AutoDream gives the agent a sleep cycle. It periodically reviews recent sessions, merges what overlaps, promotes what matters, and lets the rest fade, exactly the way memory consolidation works during REM sleep. The result is a leaner, higher-signal long-term memory that actually improves with use instead of degrading under its own weight.

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

AutoDream is on by default. Tune its settings:

- `min_hours`
- `min_sessions`
- `line_limit`
- `consolidate_every_n_dreams`
