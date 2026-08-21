# OS-ELM output-solver comparison

All held-out methods use frozen OS-ELM features and no backpropagation. 
Hyperparameters were selected on disjoint development seeds.

## Selected hyperparameters

- `lms`: `{'learning_rate': 0.3}`
- `exact_rls`: `{'regularization': 0.1}`
- `diagonal_rls`: `{'regularization': 1.0}`
- `block_rls`: `{'regularization': 1.0, 'block_size': 16}`
- `rpls`: `{'regularization': 0.1, 'components': 16}`
- `fd_ridge`: `{'regularization': 10.0, 'sketch_rank': 16}`

## Held-out means

Values are mean +/- sample standard deviation across held-out seeds.

| Method | Protocol | Online acc. | Final locked acc. | Final delta vs exact | Forgetting | State bytes | Update us |
|---|---|---:|---:|---:|---:|---:|---:|
| block_rls | class_ordered | 0.7913 +/- 0.0021 | 0.4505 +/- 0.0133 | -0.4710 +/- 0.0165 | 0.6061 +/- 0.0165 | 46680 | 134.1 +/- 2.6 |
| block_rls | shuffled_augmented | 0.6356 +/- 0.0086 | 0.9070 +/- 0.0112 | -0.0145 +/- 0.0124 | 0.0095 +/- 0.0062 | 46680 | 134.7 +/- 2.3 |
| diagonal_rls | class_ordered | 0.9442 +/- 0.0013 | 0.1000 +/- 0.0000 | -0.8215 +/- 0.0058 | 1.0000 +/- 0.0000 | 39000 | 29.4 +/- 0.3 |
| diagonal_rls | shuffled_augmented | 0.5824 +/- 0.0098 | 0.8850 +/- 0.0073 | -0.0365 +/- 0.0052 | 0.0050 +/- 0.0059 | 39000 | 29.4 +/- 0.1 |
| exact_rls | class_ordered | 0.5997 +/- 0.0070 | 0.9215 +/- 0.0058 | +0.0000 +/- 0.0000 | 0.0417 +/- 0.0088 | 72280 | 50.6 +/- 1.3 |
| exact_rls | shuffled_augmented | 0.6724 +/- 0.0089 | 0.9215 +/- 0.0058 | +0.0000 +/- 0.0000 | 0.0085 +/- 0.0055 | 72280 | 50.7 +/- 0.9 |
| fd_ridge | class_ordered | 0.5331 +/- 0.0098 | 0.8440 +/- 0.0122 | -0.0775 +/- 0.0073 | 0.0756 +/- 0.0120 | 60352 | 147.3 +/- 0.5 |
| fd_ridge | shuffled_augmented | 0.6015 +/- 0.0126 | 0.8455 +/- 0.0111 | -0.0760 +/- 0.0104 | 0.0375 +/- 0.0126 | 60352 | 147.5 +/- 0.5 |
| lms | class_ordered | 0.9904 +/- 0.0002 | 0.1000 +/- 0.0000 | -0.8215 +/- 0.0058 | 1.0000 +/- 0.0000 | 38480 | 23.8 +/- 0.5 |
| lms | shuffled_augmented | 0.4638 +/- 0.0124 | 0.7960 +/- 0.0574 | -0.1255 +/- 0.0561 | 0.0300 +/- 0.0346 | 38480 | 23.5 +/- 0.3 |
| memory_matched_exact_rls | class_ordered | 0.5908 +/- 0.0105 | 0.9175 +/- 0.0103 | -0.0040 +/- 0.0068 | 0.0439 +/- 0.0120 | 59672 | 49.4 +/- 0.2 |
| memory_matched_exact_rls | shuffled_augmented | 0.6649 +/- 0.0123 | 0.9175 +/- 0.0103 | -0.0040 +/- 0.0068 | 0.0030 +/- 0.0045 | 59672 | 49.2 +/- 0.2 |
| rpls | class_ordered | 0.5892 +/- 0.0062 | 0.9200 +/- 0.0035 | -0.0015 +/- 0.0052 | 0.0439 +/- 0.0080 | 77568 | 4809.4 +/- 10.3 |
| rpls | shuffled_augmented | 0.6626 +/- 0.0052 | 0.9200 +/- 0.0035 | -0.0015 +/- 0.0052 | 0.0040 +/- 0.0029 | 77568 | 8566.8 +/- 31.8 |

## Secondary metrics

| Method | Protocol | Sample-efficiency AUC | Adaptation delta | Backward transfer | Samples/s | Peak traced bytes |
|---|---|---:|---:|---:|---:|---:|
| block_rls | class_ordered | 0.8822 +/- 0.0023 | +0.7240 +/- 0.0337 | -0.6061 +/- 0.0165 | 4031.7 +/- 62.4 | 547862 +/- 0 |
| block_rls | shuffled_augmented | 0.5699 +/- 0.0064 | +0.0448 +/- 0.0613 | +0.0780 +/- 0.0303 | 4002.2 +/- 73.5 | 547862 +/- 0 |
| diagonal_rls | class_ordered | 0.9637 +/- 0.0010 | +0.5696 +/- 0.0104 | -1.0000 +/- 0.0000 | 7301.0 +/- 94.0 | 531254 +/- 0 |
| diagonal_rls | shuffled_augmented | 0.4731 +/- 0.0049 | +0.0512 +/- 0.0412 | +0.2185 +/- 0.0381 | 7308.6 +/- 30.0 | 531238 +/- 0 |
| exact_rls | class_ordered | 0.7738 +/- 0.0054 | +0.6288 +/- 0.0355 | -0.0372 +/- 0.0075 | 6264.3 +/- 40.3 | 665659 +/- 0 |
| exact_rls | shuffled_augmented | 0.6078 +/- 0.0088 | +0.0568 +/- 0.0502 | +0.0730 +/- 0.0021 | 6259.8 +/- 51.1 | 665633 +/- 14 |
| fd_ridge | class_ordered | 0.7182 +/- 0.0054 | +0.6328 +/- 0.0308 | -0.0661 +/- 0.0132 | 3830.8 +/- 14.9 | 581098 +/- 632 |
| fd_ridge | shuffled_augmented | 0.5520 +/- 0.0067 | +0.0336 +/- 0.0556 | +0.0365 +/- 0.0346 | 3835.3 +/- 16.3 | 581491 +/- 277 |
| lms | class_ordered | 0.9820 +/- 0.0002 | +0.1072 +/- 0.0018 | -1.0000 +/- 0.0000 | 7652.3 +/- 260.0 | 531819 +/- 6612 |
| lms | shuffled_augmented | 0.3592 +/- 0.0181 | +0.0424 +/- 0.0067 | +0.3960 +/- 0.1753 | 7583.8 +/- 285.2 | 529019 +/- 315 |
| memory_matched_exact_rls | class_ordered | 0.7690 +/- 0.0070 | +0.6280 +/- 0.0335 | -0.0417 +/- 0.0118 | 5429.7 +/- 15.4 | 633593 +/- 1106 |
| memory_matched_exact_rls | shuffled_augmented | 0.6031 +/- 0.0134 | +0.0552 +/- 0.0398 | +0.0695 +/- 0.0112 | 5439.9 +/- 31.8 | 633754 +/- 1537 |
| rpls | class_ordered | 0.7680 +/- 0.0048 | +0.6160 +/- 0.0273 | -0.0389 +/- 0.0094 | 202.7 +/- 0.5 | 638059 +/- 626 |
| rpls | shuffled_augmented | 0.6026 +/- 0.0076 | +0.0456 +/- 0.0503 | +0.0740 +/- 0.0072 | 115.0 +/- 0.4 | 637747 +/- 0 |

## Descriptive winners

- Predictive quality: `exact_rls`.
- Under the declared reduced-state budget: `block_rls`.
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
