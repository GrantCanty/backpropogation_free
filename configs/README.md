# Active configurations

This directory contains only proposal-independent experiment configurations:

- `signal_mvp.json`: nonstationary scalar prediction with frozen, LMS, and RLS
  readouts;
- `delayed_mvp.json`: delayed association with fixed and eligibility reservoirs;
- `continual_mvp.json`: recurring-context classification controls.
- `solver_comparison.json`: full frozen-OS-ELM output-solver study;
- `solver_comparison_smoke.json`: fast structural version of that study.
- `solver_campaign_reference.json`: compact frozen parameters and provenance
  needed to run confirmatory campaigns without copying ignored result files.

The completed CPAM digit, drift, scaling, memory, predictive, and historical
systems-comparison configs are preserved under `archives/cpam/configs/`. New
proposals should add their own configs under a proposal-named subdirectory or
use a clearly neutral benchmark config here.

Each active JSON file declares its `benchmark` and can be executed directly:

```bash
PYTHONPATH=src python3 -m experiments --config configs/signal_mvp.json
```

Explicit command-line settings such as `--steps`, `--hidden-size`, and `--seed`
override values from the file.

For the solver comparison, `--output` names an artifact directory containing
`comparison.json`, one JSON file per held-out run, development trials, and
`REPORT.md`.
