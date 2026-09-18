# AutoDream improvement ledger

## Scope

Implement the first research-backed safety patch on top of `7b897da`: complete
inputs for memory replacement/deletion, validation before any plan mutation,
recoverable originals, and provenance preservation. Keep the existing Markdown
and vector storage architecture. Research was completed with DeepAPI on 18 September 2026; principal sources:
[consolidation drift](https://arxiv.org/abs/2605.12978),
[LongMemEval](https://arxiv.org/abs/2410.10813), and
[MemTxn](https://arxiv.org/abs/2607.27834).

## Work

| Item | Status | Decision / evidence |
| --- | --- | --- |
| Trace learning, consolidation, file writes, and vector sync | Complete | Both model passes share `apply_auto_dream_plan`; validation belongs there. |
| Complete memory inputs | Complete | Select whole files within the existing approximate 60,000-character content budget; defer files that do not fit. Only supplied, unchanged files may be replaced/deleted. |
| Strict plans | Complete | Reject malformed actions, unsafe/duplicate targets, unsupported evidence IDs, and invalid merge dependencies before writing. |
| Preserve provenance | Complete | Include metadata in model input and inherit source references from updated/merged originals. |
| Recoverable originals | Complete | Archive originals outside active memory/index/vector discovery before replacing/deleting; use the framework atomic-write helper. |
| Prompt and documentation contracts | Complete | Replace mandatory aggressive pruning with conservative, evidence-preserving merge instructions. |
| Regression verification | Complete | Cover long-file details, rejected plans, provenance across merges, failure recovery, and the existing batching/checkpoint cases. |

## Follow-up work

Quality evaluation, relevant older-file selection, instruction-policy integration,
and incremental ingestion remain separate follow-ups. This patch does not claim
semantic verification, whole-run transactions, automatic rollback, or an evaluated
improvement in answer quality.

## Verification notes

- Initial safety run passed full-file selection, 20 invalid-plan cases, stale
  snapshots, symlink rejection, and archive/replacement failure handling.
- Corrected the archive-count assertion: directories named after original `.md`
  files also match `rglob("*.md")`; count only regular archive files.
- Explicit validated paths now retain their identity on retries. Removed the
  unused title-based filename allocation helpers.
- Malformed consolidation now fails the run rather than being swallowed before
  checkpoint advancement.

- All ten run/checkpoint scenarios passed in the framework runtime, including
  malformed learning and consolidation plans.
- Final combined run passed all ten run/checkpoint scenarios and all eight
  safety groups, including 20 invalid-plan cases, a partial-write failure after
  a first replacement was published, and lineage across three rewrites.
- The prompt no longer promises automatic enforcement of generated rule files.
  Instruction-policy integration itself remains a separate follow-up.

## Final verification

- Runtime: `agent-zero` at `localhost:32081`, framework Python
  `/opt/venv-a0/bin/python`, with the candidate copied under
  `/a0/tmp/autodream-safety/`. Models and vector DB calls were mocked; files used
  isolated temporary directories. AutoDream was not installed or enabled.
- Runnable checks: `tests/test_dream_coverage.py` and
  `tests/test_dream_safety.py`, using the commands in the README.
- Source/test syntax parsing and `git diff --check` passed. SHA-256 comparisons
  confirmed the tested helper and safety test match the repository files.
- No dependencies were installed and no real memories were changed.
- Version bumped from `1.0.5` to `1.0.6` for the safety patch. Commit and push
  to `main` authorized by the user. The quality evaluation and other follow-ups
  listed above have not been implemented or measured.
