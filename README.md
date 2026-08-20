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
- Diagonal/block RLS approximations and protected prototype memories
- Factor-free cumulative slow memory, residual fast representations, and a
  learned reliability ranker
- Single-path cumulative maturity networks with recruitable local neurons,
  tested with and without entropy-gated recruitment
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
- Lazy 8x8/28x28 stream-length and feature-width scaling benchmarks

All current data is generated or bundled locally and handled deterministically.
Running the MVP does not download a dataset.

## Development

```bash
python3 -m pytest
PYTHONPATH=src python3 -m no_backprop signal --output results/signal.json
PYTHONPATH=src python3 -m no_backprop delayed --output results/delayed.json
PYTHONPATH=src python3 -m no_backprop continual --output results/continual.json
PYTHONPATH=src python3 -m no_backprop digits --output results/digits.json
PYTHONPATH=src python3 -m no_backprop memory --output results/milestone6.json
PYTHONPATH=src python3 -m no_backprop cumulative-memory \
  --output results/cumulative-memory.json
PYTHONPATH=src python3 -m no_backprop scale --output results/scaling.json
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

The scale command does not allocate a 60,000-image dataset. It reuses one blank
image while executing the full predict/learn path, which isolates learner state
and runtime scaling. Its 28x28 case matches Fashion-MNIST's image dimensions and
60,000-example training-set length without downloading Fashion-MNIST.

The cumulative-memory command tests both factor-free branches. Every labeled
observation updates cumulative statistics with unit weight, no raw image is
retained, and no observation is aged out. The first architecture routes between
slow and residual memories. The maturity architectures instead use one
prediction path and a shared representation containing recruitable local
neurons; the matched entropy variant suppresses recruitment on high-entropy
startup mistakes. The leverage variant instead delays recruitment until an
error occurs at below-average RLS novelty, using a cumulative learned baseline
rather than a threshold. The probation variant holds a proposed key outside the
prediction path until a later nearby observation confirms its label, then
freezes the averaged center permanently. The adaptive key-value variants treat
the recurrent feature as a query, each local neuron center as a learned key,
its diagonal variance as the locality rule, and its output-weight column as a
value. Key updates use local cumulative statistics rather than gradients or a
backward pass.

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
- exact RLS state is constant across 60,000 images, but its quadratic feature
  state grows from 38 KB at width 65 to 2.05 MB at width 513; block RLS reduces
  that width-513 state to 104 KB while retaining 86% shuffled-digit accuracy
- the first factor-free fast/slow prototype proves exact cumulative inclusion
  without raw-sample storage, but currently trades about 3.4 points of better
  ordered-stream online accuracy for about 4.0 points of final retention versus
  its cumulative RLS slow-path baseline
- the single-path entropy maturity model averages 90.42% shuffled and 90.33%
  class-ordered final accuracy, modestly exceeding both its non-entropy ablation
  and the 90.08% no-discount RLS baseline; the three-seed differences are small
- adaptive key-value neurons raise ordered online accuracy from 76.83% to
  77.98% while leaving ordered final accuracy nearly unchanged (89.67% to
  89.75%); this costs 24% more state and roughly 6% more training time than the
  fixed-key control on the current vectorized CPU benchmark
- across 10 seeds, RLS-leverage-gated recruitment raises fixed-key ordered final
  accuracy from 90.25% to 90.78%; the nominal paired 95% interval is only +0.05
  to +1.01 points, so this is a modest result rather than definitive evidence
- adding probationary frozen keys to leverage gating raises shuffled and ordered
  online accuracy by 0.45 and 0.47 points across 10 seeds without a detectable
  final-accuracy or drift change; the dormant bank adds 12.6% state
- explicit top-k local responsibility does not improve the probation model:
  unnormalized top-4 is slightly slower with matched quality, winner-only loses
  ordered online accuracy, and normalized top-k catastrophically harms final
  retention by forcing weak matches to carry unit activation

These results validate the experimental machinery and the narrow MVP
hypotheses. They do **not** establish an advantage on real-world data or prove
that local learning generally outperforms backpropagation. See
[docs/RESULTS.md](docs/RESULTS.md) for measurements and limitations.

## Roadmap

The `memory` branch now tests factor-free complementary, single-path maturity,
and adaptive key-value representations. The next mechanism problem is reducing
stranded probation candidates and managing bounded capacity without sacrificing
the cumulative invariant.
JEPA-inspired predictive representations remain a later experiment; they are
not an I-JEPA reimplementation, and no automatic differentiation or backward
pass enters the core learner.
