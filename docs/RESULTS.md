# MVP Results

## Scope

These are controlled mechanism tests on deterministic, locally generated data.
No external dataset was downloaded. Measurements were produced on 20 August
2026 with Python 3.10, NumPy, and CPU execution. Values are useful for comparing
the implementations in this repository, not for general performance claims.

## Verification

The complete suite contains 23 tests covering:

- strict predict-before-learn ordering
- deterministic streams
- LMS, RLS, fast/slow, and eligibility updates
- checkpoint round trips
- constant learner state size
- absence of automatic differentiation in `src/no_backprop`
- isolation of the PyTorch baseline
- BPTT activation-memory scaling
- smoke experiments and multi-seed aggregation

Run it with:

```bash
python3 -m pytest
```

## Nonstationary signal

One scalar signal changes dynamics every 750 events. Predictions are scored
before the current target is used for an update.

| Learner | Prequential MSE | Tail MSE | State | Events/s |
|---|---:|---:|---:|---:|
| Frozen readout | 0.5130 | 0.5464 | 34.8 KB | 51.0k |
| LMS | 0.00254 | 0.00105 | 34.8 KB | 41.1k |
| RLS | 0.00110 | 0.00064 | 68.6 KB | 27.6k |

The primary result is that both online readouts adapt immediately and retain
exactly the same model-state size from the first event to the last.

## Delayed association

Each episode presents a random `-1` or `+1` cue, eight distractor events, and a
query. Feedback is available only at the query. Results aggregate seeds 3, 7,
11, 17, and 23 over 1,200 episodes per seed.

| Learner | Mean MSE | Tail MSE | Accuracy | Tail accuracy |
|---|---:|---:|---:|---:|
| Fixed recurrence + LMS | 0.3663 | 0.2051 | 90.17% | 95.8% |
| Eligibility recurrence + LMS | 0.2764 | 0.0694 | 93.33% | 100% |

Eligibility reduced mean MSE by 0.0899 on average. The improvement was positive
for all five seeds, but its size varied substantially. The plastic model stores
one recurrent eligibility value per recurrent weight, so its bounded state is
larger and its event throughput is lower.

## Recurring-context classification

Synthetic classes are transformed into contexts 0, 1, 2, and then 0 again.
The learner is not given a task-boundary signal. Results aggregate the same five
seeds over 3,200 events per seed.

| Learner | Accuracy | Tail accuracy | Return-context retention delta | MSE |
|---|---:|---:|---:|---:|
| Fixed recurrence | 93.81% | 96.83% | -0.1683 | 0.06363 |
| Eligibility | 93.86% | 96.83% | -0.1667 | 0.06360 |
| Surprise-gated eligibility | 93.84% | 96.83% | -0.1683 | 0.06361 |
| Fast/slow + gated eligibility | 93.12% | 96.67% | -0.1317 | 0.07004 |

Fast/slow weights improved the retention delta by 0.0367 on average across all
five seeds, at an average accuracy cost of 0.69 percentage points. Eligibility
and surprise gating alone made little difference on this benchmark. This is a
measured tradeoff, not a universal stability solution.

## BPTT systems comparison

A matched-width PyTorch tanh RNN uses Adam and truncated BPTT on the same local
signal. The online core uses float64; PyTorch uses float32. “Training state” is
model parameters, optimizer tensors, and unique non-parameter tensor storage
retained for the backward pass.

| Learner | Window | Prequential MSE | Training state | Events/s |
|---|---:|---:|---:|---:|
| LMS | n/a | 0.00254 | 34.8 KB | 41.1k |
| RLS | n/a | 0.00110 | 68.6 KB | 27.6k |
| BPTT | 8 | 0.0670 | 53.9 KB | 6.3k |
| BPTT | 32 | 0.0254 | 60.2 KB | 7.1k |
| BPTT | 128 | 0.0511 | 85.5 KB | 6.8k |

Unique saved non-parameter tensor storage grew from 2.4 KB at window 8 to 8.7
KB at window 32 and 34.0 KB at window 128. LMS state did not depend on stream
length. RLS illustrates an important counterexample: its inverse-correlation
matrix makes it use more state than short-window BPTT.

The online methods are unusually well suited to this simple signal, update
every event, and were not matched for optimizer tuning. Their accuracy and speed
advantage here must not be extrapolated to complex tasks.

## Acceptance gates

1. **Mechanical correctness:** passed; core code has no autograd dependency.
2. **Online learning:** passed; LMS and RLS beat the frozen control.
3. **Temporal credit:** passed narrowly; eligibility improved delayed MSE across
   all five tested seeds.
4. **Bounded memory:** passed for persistent learner state; BPTT retained state
   increased with window length.
5. **Continual stability:** passed as a measured tradeoff; fast/slow weights
   improved retention with a small accuracy cost.

## Limitations and next decision

- All tasks are synthetic and small.
- Hyperparameters have not undergone a formal search.
- The BPTT comparison is controlled but not equivalently optimized.
- Per-synapse eligibility scales quadratically in a dense recurrent layer.
- There is no adversarial update protection or production safety layer.
- Retention still declines when context zero returns.
- CPU throughput does not establish energy or accelerator efficiency.

The next milestone should select exactly one local domain—streaming audio/sensor
prediction, small control, or continual image classification—and define a
domain-specific success threshold before adding any downloaded data.
