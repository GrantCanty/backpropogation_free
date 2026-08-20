# No-Backprop Continual Learning

This repository investigates recurrent systems that predict, act, and update
from a stream without automatic differentiation or backpropagation through
time. The core learner is NumPy-only and uses online readouts plus local
eligibility traces. A conventional PyTorch baseline is isolated under
`src/baselines` for controlled comparisons.

The experimental contract and milestone plan are in [PLAN.md](PLAN.md).

## What is implemented

- Strict `predict -> score -> learn` streaming protocol
- Fixed recurrent reservoir with LMS and RLS online readouts
- Local recurrent and input eligibility traces with fixed random feedback
- Surprise-gated recurrent plasticity
- Fast/slow readout weights with gradual consolidation
- Explicit NumPy checkpoints
- Nonstationary prediction, delayed association, and recurring-context
  classification benchmarks
- Bundled 8x8 handwritten-digit benchmark with matched shuffled,
  augmented-shuffled, and class-ordered streams
- Multi-seed replication and ablations
- Isolated truncated-BPTT systems baseline
- JSON results and optional plots

All current data is generated or bundled locally and handled deterministically.
Running the MVP does not download a dataset.

## Development

```bash
python3 -m pytest
PYTHONPATH=src python3 -m no_backprop signal --output results/signal.json
PYTHONPATH=src python3 -m no_backprop delayed --output results/delayed.json
PYTHONPATH=src python3 -m no_backprop continual --output results/continual.json
PYTHONPATH=src python3 -m no_backprop digits --output results/digits.json
PYTHONPATH=src python3 -m no_backprop replicate --output results/replication.json

# Conventional comparison, kept outside the core package
PYTHONPATH=src python3 -m baselines --output results/systems.json
PYTHONPATH=src python3 -m baselines --benchmark digits \
  --output results/digits-systems.json

# Optional local plots
PYTHONPATH=src python3 -m no_backprop plot \
  --input results/signal.json --output results/signal.png
```

The commands use `PYTHONPATH=src` so the project can run without modifying the
active Python environment. An editable installation is optional.

The digits command uses `sklearn.datasets.load_digits`, which ships inside an
existing scikit-learn installation. It does not access the network. Each image
is presented as eight row-events, and its label is used once after the final
row. The shuffled stream measures ordinary cumulative learning; the
augmented-shuffled stream adds deterministic translations and noise to training
images only; the class-ordered stream measures adaptation and catastrophic
forgetting. Every held-out pass explicitly verifies that model weights remain
unchanged.

## Architecture

```text
observation -> recurrent state -> prediction -> outcome
                    |                            |
                    +---- eligibility trace <---+ error/reward broadcast
                                  |
                           local weight update
```

The core stores only the current recurrent state, model parameters, and fixed-
size plasticity state. It never retains a sequence-length computation graph.
The experiment recorder may retain metrics for reporting; that memory is not
part of the deployed learner.

## Current evidence

The controlled MVP passes the five gates in `PLAN.md`:

- online LMS/RLS substantially outperform a frozen readout on the generated
  nonstationary signal
- eligibility-based recurrent updates improve mean delayed-association error
  across five seeds
- model state remains the same size before and after each stream
- tracked BPTT activation state grows with the unroll window
- fast/slow consolidation modestly improves recurring-context retention while
  slightly reducing average accuracy
- on bundled handwritten digits, RLS reaches about 91% held-out accuracy on a
  shuffled one-pass stream and retains about 89% after class-ordered exposure;
  LMS and the current fast/slow rule catastrophically forget the ordered stream
- on the same shuffled image stream, a parameter-matched BPTT/Adam RNN reaches
  about 89% held-out accuracy versus about 91% for RLS; deterministic
  augmentation raises BPTT to about 90% but lowers RLS to about 88%

These results validate the experimental machinery and the narrow MVP
hypotheses. They do **not** establish an advantage on real-world data or prove
that local learning generally outperforms backpropagation. See
[docs/RESULTS.md](docs/RESULTS.md) for measurements and limitations.
