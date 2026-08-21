# CPAM Reproduction Manifest

CPAM is preserved as a completed research record, while reusable code remains
active and singly maintained. The last commit containing the original,
pre-archive source layout is `24c59fa`. Commit `60a6309` records the initial
over-broad archive operation and should not be used as the architectural model.

## Historical evidence retained here

- `research_paper.md`, `literature_review.md`, and `future_work.md` preserve the
  intended report and follow-up work.
- `PLAN.md` preserves the chronological project plan.
- `results/` contains the generated JSON and plot artifacts used by the report.
- `README.md` records the historical commands and findings.

## Archived CPAM components

- proposal implementation: `src/methods/cpam/`
- CPAM-coupled historical runners: `src/experiments/`
- old compatibility modules for those runners: `src/no_backprop/`
- proposal and historical comparison tests: `tests/`
- exact declarative run settings: `configs/`

## Active shared components

- permanent comparison methods: repository-root `src/baselines/`
- datasets, streams, metrics, evaluation, checkpoints, and results:
  repository-root `src/continual_core/`
- neutral experiment framework: repository-root `src/experiments/`
- baseline and infrastructure tests: repository-root `tests/`

Run the active suite from the repository root with `python3 -m pytest`. The
archive contains only CPAM-specific or CPAM-coupled historical material, not
duplicate shared implementations. Use commit `24c59fa` when exact
reconstruction of the old monolithic layout is required.

The separated archive can also be checked against the active shared core and
baselines without installing anything:

```bash
PYTHONSAFEPATH=1 PYTHONPATH=archives/cpam/src:src \
  python3 -m pytest -c /dev/null --rootdir=archives/cpam \
  archives/cpam/tests/methods/cpam archives/cpam/tests/experiments
```

`PYTHONSAFEPATH=1` prevents the active repository root from shadowing the
archive's `experiments` package during this historical reproduction check.
