# Continual Learning Research

This repository is the workspace for a new ground-up investigation of
continuous learning, predictive representations, and subquadratic alternatives
to exact recursive least squares.

The completed CPAM/no-backprop research narrative and generated result artifacts
are preserved under [`archives/cpam`](archives/cpam). Reusable source,
configuration-independent evaluation infrastructure, baselines, and tests
remain active at the repository root. CPAM-specific code, runners, tests, and
configs are retired under `archives/cpam`; shared code did not move with it.

OS-ELM, RLS, the analytic controls, stream machinery, metrics, resource
accounting, and their tests are active research infrastructure. New candidate
methods must depend on their public interfaces without making those components
proposal-specific.

## Reusable architecture

- `continual_core` owns protocols, datasets, streams, state locking, generic
  checkpoints, metrics, result schemas, and factory-driven evaluation.
- `baselines` owns independent LMS, RLS, approximate RLS, reservoir, OS-ELM,
  kernel RLS, RAN, ARTMAP, and backpropagation controls.
- `methods/<proposal>` is reserved for the next replaceable research idea.
- `experiments` contains neutral factory-driven runners and reusable synthetic
  stream benchmarks. Candidate methods receive dependencies through public
  factories.
- `archives/cpam` contains the retired proposal and its historical commands.

New experiments should inject `MethodSetup` learner factories through
`experiments.online_classification` and use the adapters in
`continual_core.evaluation`. Persistent state must be exposed through the
public state contract so locked evaluation, accounting, and checkpoints work
without any method-specific branches.

```bash
python3 -m pytest
PYTHONPATH=src python3 -m experiments --config configs/signal_mvp.json \
  --output results/signal.json --plot results/signal.png
PYTHONPATH=src python3 -m experiments --config configs/delayed_mvp.json
PYTHONPATH=src python3 -m experiments --config configs/continual_mvp.json
PYTHONPATH=src python3 -m experiments \
  --config configs/solver_comparison_smoke.json \
  --output results/solver_comparison_smoke
PYTHONPATH=src python3 scripts/run_solver_campaign.py \
  --stage confirmatory --confirm-seeds 50 \
  --output results/solver_campaign
```

The three-stage confirmatory, cross-dataset, and feature-width workflow is
documented in [`docs/SOLVER_CAMPAIGN.md`](docs/SOLVER_CAMPAIGN.md). Dataset
selection is explicit, and network-backed datasets require `--allow-download`.

Explicit CLI values override their corresponding JSON values. For example,
add `--steps 500 --seed 3` to run a smaller deterministic signal experiment.
