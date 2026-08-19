# No-Backprop Continual Learning

This repository investigates recurrent systems that predict, act, and update
from a stream without automatic differentiation or backpropagation through
time. The core learner is NumPy-only and uses online readouts plus local
eligibility traces. A conventional PyTorch baseline is isolated under
`src/baselines` for controlled comparisons.

The experimental contract and milestone plan are in [PLAN.md](PLAN.md).

## Status

Implementation is in progress. The first vertical slice targets a fixed
reservoir with LMS and RLS readouts on a nonstationary prediction stream.

## Development

```bash
python3 -m pytest
python3 -m no_backprop --help
```
