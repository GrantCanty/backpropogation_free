# OS-ELM output-solver comparison

All held-out methods use frozen OS-ELM features and no backpropagation. 
Hyperparameters were selected on disjoint development seeds.

## Selected hyperparameters

- `lms`: `{'learning_rate': 0.1}`
- `exact_rls`: `{'regularization': 1.0}`
- `diagonal_rls`: `{'regularization': 1.0}`
- `block_rls`: `{'regularization': 1.0, 'block_size': 4}`
- `rpls`: `{'regularization': 1.0, 'components': 4}`
- `fd_ridge`: `{'regularization': 1.0, 'sketch_rank': 4}`

## Held-out means

Values are mean +/- sample standard deviation across held-out seeds.

| Method | Protocol | Online acc. | Final locked acc. | Final delta vs exact | Forgetting | State bytes | Update us |
|---|---|---:|---:|---:|---:|---:|---:|
| block_rls | class_ordered | 0.6750 +/- 0.0000 | 0.2200 +/- 0.0000 | -0.5600 +/- 0.0000 | 0.8667 +/- 0.0000 | 10200 | 125.5 +/- 0.0 |
| block_rls | shuffled_augmented | 0.2500 +/- 0.0000 | 0.6200 +/- 0.0000 | -0.0800 +/- 0.0000 | 0.0000 +/- 0.0000 | 10200 | 129.1 +/- 0.0 |
| diagonal_rls | class_ordered | 0.7750 +/- 0.0000 | 0.1000 +/- 0.0000 | -0.6800 +/- 0.0000 | 1.0000 +/- 0.0000 | 9816 | 31.2 +/- 0.0 |
| diagonal_rls | shuffled_augmented | 0.1000 +/- 0.0000 | 0.3800 +/- 0.0000 | -0.3200 +/- 0.0000 | 0.0400 +/- 0.0000 | 9816 | 31.2 +/- 0.0 |
| exact_rls | class_ordered | 0.3875 +/- 0.0000 | 0.7800 +/- 0.0000 | +0.0000 +/- 0.0000 | 0.1111 +/- 0.0000 | 11992 | 35.4 +/- 0.0 |
| exact_rls | shuffled_augmented | 0.3875 +/- 0.0000 | 0.7000 +/- 0.0000 | +0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 11992 | 35.8 +/- 0.0 |
| fd_ridge | class_ordered | 0.3625 +/- 0.0000 | 0.7600 +/- 0.0000 | -0.0200 +/- 0.0000 | 0.0889 +/- 0.0000 | 12160 | 109.8 +/- 0.0 |
| fd_ridge | shuffled_augmented | 0.4250 +/- 0.0000 | 0.6600 +/- 0.0000 | -0.0400 +/- 0.0000 | 0.0000 +/- 0.0000 | 12160 | 109.9 +/- 0.0 |
| lms | class_ordered | 0.4375 +/- 0.0000 | 0.1000 +/- 0.0000 | -0.6800 +/- 0.0000 | 1.0000 +/- 0.0000 | 9680 | 28.9 +/- 0.0 |
| lms | shuffled_augmented | 0.0750 +/- 0.0000 | 0.1400 +/- 0.0000 | -0.5600 +/- 0.0000 | 0.2000 +/- 0.0000 | 9680 | 23.7 +/- 0.0 |
| memory_matched_exact_rls | class_ordered | 0.3875 +/- 0.0000 | 0.7800 +/- 0.0000 | +0.0000 +/- 0.0000 | 0.1111 +/- 0.0000 | 11992 | 43.0 +/- 0.0 |
| memory_matched_exact_rls | shuffled_augmented | 0.3875 +/- 0.0000 | 0.7000 +/- 0.0000 | +0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 11992 | 41.1 +/- 0.0 |
| rpls | class_ordered | 0.3875 +/- 0.0000 | 0.8000 +/- 0.0000 | +0.0200 +/- 0.0000 | 0.0889 +/- 0.0000 | 13440 | 764.9 +/- 0.0 |
| rpls | shuffled_augmented | 0.3875 +/- 0.0000 | 0.7000 +/- 0.0000 | +0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 13440 | 1133.0 +/- 0.0 |

## Secondary metrics

| Method | Protocol | Sample-efficiency AUC | Adaptation delta | Backward transfer | Samples/s | Peak traced bytes |
|---|---|---:|---:|---:|---:|---:|
| block_rls | class_ordered | 0.8013 +/- 0.0000 | +0.9000 +/- 0.0000 | -0.8667 +/- 0.0000 | 1313.3 +/- 0.0 | 126156 +/- 0 |
| block_rls | shuffled_augmented | 0.1388 +/- 0.0000 | -0.1000 +/- 0.0000 | +0.4800 +/- 0.0000 | 1275.4 +/- 0.0 | 126082 +/- 0 |
| diagonal_rls | class_ordered | 0.8644 +/- 0.0000 | +0.9000 +/- 0.0000 | -1.0000 +/- 0.0000 | 1774.7 +/- 0.0 | 124154 +/- 0 |
| diagonal_rls | shuffled_augmented | 0.0538 +/- 0.0000 | -0.2000 +/- 0.0000 | +0.2800 +/- 0.0000 | 1765.2 +/- 0.0 | 124130 +/- 0 |
| exact_rls | class_ordered | 0.6144 +/- 0.0000 | +0.7000 +/- 0.0000 | -0.1111 +/- 0.0000 | 1787.4 +/- 0.0 | 130758 +/- 0 |
| exact_rls | shuffled_augmented | 0.2619 +/- 0.0000 | +0.0000 +/- 0.0000 | +0.4000 +/- 0.0000 | 1747.8 +/- 0.0 | 130695 +/- 0 |
| fd_ridge | class_ordered | 0.6081 +/- 0.0000 | +0.6000 +/- 0.0000 | -0.0444 +/- 0.0000 | 1391.0 +/- 0.0 | 128806 +/- 0 |
| fd_ridge | shuffled_augmented | 0.2806 +/- 0.0000 | +0.1000 +/- 0.0000 | +0.3400 +/- 0.0000 | 1384.9 +/- 0.0 | 128002 +/- 0 |
| lms | class_ordered | 0.5694 +/- 0.0000 | +0.9000 +/- 0.0000 | -1.0000 +/- 0.0000 | 1186.7 +/- 0.0 | 122530 +/- 0 |
| lms | shuffled_augmented | 0.0762 +/- 0.0000 | -0.1000 +/- 0.0000 | +0.0400 +/- 0.0000 | 1882.4 +/- 0.0 | 122898 +/- 0 |
| memory_matched_exact_rls | class_ordered | 0.6144 +/- 0.0000 | +0.7000 +/- 0.0000 | -0.1111 +/- 0.0000 | 1530.1 +/- 0.0 | 132255 +/- 0 |
| memory_matched_exact_rls | shuffled_augmented | 0.2619 +/- 0.0000 | +0.0000 +/- 0.0000 | +0.4000 +/- 0.0000 | 1517.6 +/- 0.0 | 135668 +/- 0 |
| rpls | class_ordered | 0.6144 +/- 0.0000 | +0.7000 +/- 0.0000 | -0.0889 +/- 0.0000 | 741.8 +/- 0.0 | 130295 +/- 0 |
| rpls | shuffled_augmented | 0.2619 +/- 0.0000 | +0.0000 +/- 0.0000 | +0.4000 +/- 0.0000 | 577.6 +/- 0.0 | 133812 +/- 0 |

## Descriptive winners

- Predictive quality: `rpls`.
- Under the declared reduced-state budget: `diagonal_rls`.
- At or below exact-RLS update time: `exact_rls`.

## Main findings

- Exact cumulative RLS is the strongest overall control. Its final locked accuracy is order-invariant because factor-one ridge sufficient statistics are cumulative.
- The memory-matched smaller exact RLS is the most important efficiency result: it gives up only about 0.4 percentage points of final accuracy while reducing total persistent state by about 17%. It strongly dominates the tested Frequent-Directions approximation at nearly the same state.
- RPLS approximately recovers exact-RLS final accuracy, but stores more state and is over two orders of magnitude slower per update in this NumPy implementation. It does not offer a useful trade-off here.
- Block RLS retains good shuffled accuracy with less state, but loses substantial class-ordered knowledge. Diagonal RLS and LMS look highly adaptive from ordered online accuracy while collapsing to chance-level final retention; online accuracy alone would therefore be misleading.
- The tested Frequent-Directions ridge method reduces state modestly but loses substantial accuracy and is slower than exact RLS. This implementation is a negative result, not evidence against every covariance-sketch design.

## Validation audit

All 70 held-out runs processed 2,794 updates. No run produced a non-finite persistent value, changed state during locked evaluation, or changed its declared persistent-state size. Development trials and the five held-out seeds are serialized separately.

These winners are benchmark-specific descriptive results, not a claim of universal or statistically definitive dominance. Raw development and held-out runs are retained alongside this report.

Peak memory is reported as traced Python/NumPy allocations. Per-method isolated peak process RSS was not reliably measurable inside this single-process paired runner and is recorded as null.
