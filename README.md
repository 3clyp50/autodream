# AutoDream

AutoDream is a plugin that depends on the builtin `_memory` plugin of Agent Zero.

Agent Zero's _memory plugin stores everything: every fragment, every session, but never forgets, never prioritises. Over time the vector database fills with duplicate fragments, stale context, and noise that dilutes semantic recall.

AutoDream gives the agent a sleep cycle. It periodically reviews recent sessions, merges what overlaps, promotes what matters, and lets the rest fade, exactly the way memory consolidation works during REM sleep. The result is a leaner, higher-signal long-term memory that actually improves with use instead of degrading under its own weight.

## Dependency Model

- `_memory` remains the source of vector storage, similarity search, and memory-subdirectory resolution.
- `AutoDream` owns the reflective pass, prompt contract, durable markdown files, `MEMORY.md` generation, and dream changelog.
- AutoDream keeps its code and settings under `usr/plugins/autodream/`, but stores durable memory inside each `_memory` scope so project-isolated memory continues to work as expected.

## Dream Coverage

Each dream reviews all pending sessions and recent vector memories in batches of
up to 8 sessions and 16 vector memories. Semantic searches use the sessions in
that same batch. Durable files and the index are refreshed between batches;
consolidation runs once after the learning batches.

The checkpoint advances to the run's start time only after every batch and vector
sync succeed, so activity during a dream remains eligible for the next run. A
failed run, including invalid learning or consolidation output, keeps its previous
checkpoint and may replay already applied batches.
Vector memories from the checkpoint's second are replayed because their timestamps
have only second precision. Existing time/session trigger settings still apply.

This prevents future coverage gaps; it cannot identify sessions already skipped
by earlier versions.

## Safe Memory Updates

Learning and consolidation receive complete selected memory files and their
metadata, up to 24 files within a 60,000-character JSON content budget. Files
that do not fit are deferred, never truncated for replacement. The index may
mention more files, but only supplied, unchanged files may be updated, merged,
or deleted. New files must cite supplied session/vector IDs or source files.

The host validates the entire plan before any mutation. Each target has one
explicit normalized Markdown path. A deletion requires a replacement upsert in
the same plan that names the original in `source_files`. Replacements are written
before their sources are removed. Invalid plans fail the run instead of silently
advancing the checkpoint.

Updates preserve existing metadata; merges inherit source chat IDs, vector IDs,
first-prompt references, and source-file lineage without count-based truncation.
Inherited `inferred` memories stay inferred. Source IDs are checked against the
supplied evidence; this provides traceability, not proof that a generated claim
is true. A `.promptinclude.md` suffix classifies a rule; AutoDream does not itself
load those files into system instructions or authorize inferred instructions.

Before replacing or deleting a file, AutoDream saves its exact bytes under
`autodream/archive/<original-name>/<sha256>.md`. Archives are outside active
memory discovery, index generation, and AutoDream vector sync. They are retained
until manually removed and are not an implementation of a request to forget data.

Writes use Agent Zero's `helpers.file_transfers.write_stream_atomic`. Atomicity
is per file, not per merge or per dream: an I/O failure can leave an applied
subset, and there is no automatic rollback. Originals remain in the archive;
copy the selected archived version back to `autodream/memories/<original-name>`
to restore it. Remove an unwanted replacement separately if needed. A subsequent
successful dream rebuilds the index and synchronizes active files to the vector
store. Review active files before restoring a failed multi-file merge.

## Files Written

For each memory scope, AutoDream writes:

- `autodream/MEMORY.md` as a compact index
- `autodream/memories/*.md` as durable memory files
- `autodream/archive/<original-name>/<sha256>.md` as recoverable originals, excluded from active retrieval
- `autodream/.dream-log.md` as a short changelog of each dream run
- `autodream/state.json` and `autodream/vector_state.json` as plugin bookkeeping
- `autodream/stats.json` as cumulative dream statistics

## Claude Code `MEMORY.md` Compatibility

AutoDream follows the same broad pattern as Claude Code's `MEMORY.md`, but per memory scope instead of assuming one global repository file.

- `MEMORY.md` is an index, not the memory itself.
- The real durable content lives in the sibling markdown files under `autodream/memories/`.
- Those durable files are synced into the `_memory` vector database so symbolic browsing and semantic recall stay aligned.

## Dream Statistics

Open AutoDream's **Config** panel to see the Dream journal: completed dreams,
estimated input/output tokens, utility-model calls, last duration, latest result,
and the last successful memory-file count. Use **Refresh** to update the panel.
Project and agent-profile selectors use the same `_memory` scope as the dream
runner; profiles sharing a memory folder share statistics.

Totals begin with the first run after this update. Older last-dream information
remains visible, but historical totals are not reconstructed. Failed attempts are
counted separately, and their known text estimates and calls contribute to totals.
Token figures use Agent Zero's text-token estimator, not provider billing usage;
embeddings, hidden reasoning, provider retries and provider-specific message
formatting are excluded. Summary, query, learning and consolidation calls are
included.

Statistics are stored atomically in `autodream/stats.json`, separate from the
processing checkpoint and memory files. A statistics-file write failure is logged
without changing the outcome of a dream. The authenticated, CSRF-protected
`/api/plugins/autodream/stats` endpoint reads this data without starting a dream
or opening a vector database. The static night-sky graphic uses the active Agent
Zero theme and adds no animation, external assets or charting dependency.

## Settings

AutoDream is on by default. Tune its settings:

- `min_hours`
- `min_sessions`
- `line_limit`
- `consolidate_every_n_dreams`

## Verification

From the Agent Zero framework root, use its Python environment:

```bash
PYTHONPATH=. python usr/plugins/autodream/tests/test_dream_coverage.py
PYTHONPATH=. python usr/plugins/autodream/tests/test_dream_safety.py
PYTHONPATH=. python usr/plugins/autodream/tests/test_dream_stats.py
```

These checks use isolated temporary files and mocked model/database calls. They
cover batching, checkpoint failures, complete-file input, plan validation,
provenance, archives, and interrupted writes; they do not measure answer quality.
Implementation progress and remaining follow-ups are recorded in
[WORK_LEDGER.md](WORK_LEDGER.md).
