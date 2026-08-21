# MVP Results

## Scope

These are controlled mechanism tests on deterministic, locally generated data.
No external dataset was downloaded. Measurements were produced on 20 August
2026 with Python 3.10, NumPy, and CPU execution. Values are useful for comparing
the implementations in this repository, not for general performance claims.

## Verification

The complete suite contains 54 tests covering:

- strict predict-before-learn ordering
- deterministic streams
- LMS, RLS, fast/slow, and eligibility updates
- checkpoint round trips
- constant learner state size
- absence of automatic differentiation in `src/no_backprop`
- isolation of the PyTorch baseline
- BPTT activation-memory scaling
- smoke experiments and multi-seed aggregation
- deterministic local digits splits and matched stream orderings
- deterministic augmentation and weight-locked evaluation for both learner
  families
- diagonal/block RLS updates, protected prototype state, and checkpoint recovery
- lazy blank-image streams and analytic feature-width projections
- exact equivalence between sequential cumulative memory and batch ridge
- factor-free residual compression, locked evaluation, and checkpoint recovery
- entropy calculation, matched cumulative maturity updates, evidence-based
  maturation, startup recruitment, locked evaluation, and checkpoint recovery

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

## Bundled 8x8 handwritten digits

The first domain-expansion experiment uses the 1,797-image dataset bundled with
scikit-learn; no data was downloaded. A deterministic stratified split holds out
40 examples per class, leaving 1,397 training images and 400 test images. Each
image is processed as eight sequential row-events. The prediction is scored and
the label is learned only after row eight.

The same split is presented in three ways:

- **Shuffled:** all classes are interleaved, testing ordinary cumulative
  learning and held-out generalization.
- **Shuffled augmented:** the original training images plus one deterministically
  translated/noisy copy are interleaved. Held-out images are never augmented.
- **Class ordered:** all examples of class 0 arrive, then class 1, and so on,
  testing immediate adaptation and retention of earlier classes.

Results below are mean ± sample standard deviation for seeds 7, 17, and 29
after one pass. They use no batches, epochs, replay, gradients, or BPTT.

| Stream | Learner | Online accuracy | Final test accuracy | Worst class | Mean forgetting |
|---|---|---:|---:|---:|---:|
| Shuffled | Frozen | 9.88% ± 0.00 | 10.00% ± 0.00 | 0.00% ± 0.00 | 0.00 ± 0.00 |
| Shuffled | LMS | 59.53% ± 1.47 | 67.33% ± 7.88 | 18.33% ± 23.63 | 0.229 ± 0.057 |
| Shuffled | RLS | 85.80% ± 0.79 | 90.83% ± 1.28 | 75.83% ± 9.46 | 0.017 ± 0.005 |
| Shuffled | Eligibility + LMS | 59.51% ± 1.54 | 67.50% ± 8.13 | 18.33% ± 23.63 | 0.227 ± 0.059 |
| Shuffled | Fast/slow + eligibility | 47.75% ± 2.25 | 58.42% ± 6.50 | 9.17% ± 15.88 | 0.314 ± 0.045 |
| Class ordered | Frozen | 9.88% ± 0.00 | 10.00% ± 0.00 | 0.00% ± 0.00 | 0.00 ± 0.00 |
| Class ordered | LMS | 97.40% ± 0.08 | 10.00% ± 0.00 | 0.00% ± 0.00 | 0.900 ± 0.000 |
| Class ordered | RLS | 83.85% ± 0.29 | 88.83% ± 1.13 | 70.83% ± 1.44 | 0.078 ± 0.010 |
| Class ordered | Eligibility + LMS | 97.40% ± 0.08 | 10.00% ± 0.00 | 0.00% ± 0.00 | 0.900 ± 0.000 |
| Class ordered | Fast/slow + eligibility | 97.47% ± 0.11 | 10.00% ± 0.00 | 0.00% ± 0.00 | 0.900 ± 0.000 |

The 97% online score in the ordered stream is not success: LMS quickly predicts
the one class currently arriving, then overwrites it when the class changes.
The held-out matrix exposes that behavior; final accuracy is chance and mean
forgetting is 0.90. This is precisely why both data orderings are required.

RLS is the clear first-domain baseline. Its inverse-correlation matrix preserves
information from earlier examples much better, at the cost of quadratic
auxiliary memory in the hidden feature count. Eligibility-based recurrent
plasticity changes little at the current learning rate, and the existing
fast/slow rule does not provide useful consolidation on this harder stream.
Those are negative but actionable findings: the next mechanism experiment
should target consolidation, not recurrent credit assignment.

### Matched BPTT comparison and augmentation

The conventional comparison is a parameter-matched tanh RNN with the same
64-unit hidden width and 5,322 model parameters. It sees each image as the same
eight-row sequence, predicts before its update, and performs one Adam/BPTT
update per image. This deliberately matches exposure count and batch size; it
is a comparison of learning rules, not a many-epoch offline accuracy ceiling.

Before every evaluation, the NumPy learner snapshots all learned arrays and the
PyTorch baseline snapshots every model parameter. The NumPy path withholds
feedback; the PyTorch path uses `eval()` and `no_grad()`. Both abort if any
learned value changes, and the NumPy path restores its transient activity and
eligibility traces afterward. Thus the 400 held-out images never train either
model or influence the next training event.

| Stream | Learner | Online accuracy | Final test accuracy | Worst class | Mean forgetting | Train images/s |
|---|---|---:|---:|---:|---:|---:|
| Shuffled | LMS | 59.53% ± 1.47 | 67.33% ± 7.88 | 18.33% | 0.229 | 7,673 |
| Shuffled | RLS | **85.80% ± 0.79** | **90.83% ± 1.28** | **75.83%** | **0.017** | 6,862 |
| Shuffled | Eligibility + LMS | 59.51% ± 1.54 | 67.50% ± 8.13 | 18.33% | 0.227 | 3,375 |
| Shuffled | Fast/slow | 47.75% ± 2.25 | 58.42% ± 6.50 | 9.17% | 0.314 | 3,239 |
| Shuffled | BPTT + Adam | 77.59% ± 2.11 | 88.92% ± 3.50 | 60.83% | 0.064 | 1,328 |
| Augmented | LMS | 40.07% ± 1.92 | 67.33% ± 9.83 | 16.67% | 0.238 | 7,384 |
| Augmented | RLS | **68.35% ± 1.08** | 88.25% ± 1.09 | 70.00% | **0.043** | 6,707 |
| Augmented | Eligibility + LMS | 40.01% ± 1.94 | 67.42% ± 9.64 | 15.83% | 0.237 | 3,429 |
| Augmented | Fast/slow | 32.61% ± 1.77 | 59.25% ± 8.13 | 8.33% | 0.314 | 3,159 |
| Augmented | BPTT + Adam | 62.15% ± 1.23 | **90.08% ± 0.58** | **75.83%** | 0.053 | 1,314 |

On the matched single-pass shuffled stream, RLS and BPTT are statistically
close in final accuracy; RLS is slightly higher in this three-seed sample and
about five times faster on this CPU. This throughput result is implementation-
and-hardware-specific, not an energy claim. BPTT's lower online accuracy but
strong final accuracy also shows that prequential and held-out measurements
answer different questions.

The augmentation is a real intervention, not an assumed improvement. It adds
translations and noise, doubles training exposure to 2,794 images, and reduces
online accuracy because examples are harder. It improves BPTT's mean held-out
accuracy and sharply reduces its seed variance, but slightly reduces RLS's mean
accuracy. The next augmentation study should separate translation from noise
and match total exposure with a repeated-unaugmented control.

## Milestone 6: scalable continual memory

Milestone 6 separates learning quality from systems scaling. Quality results use
the bundled digits split and aggregate seeds 7, 17, and 29. The systems run uses
synthetic blank images because accuracy is deliberately irrelevant there.

### Memory approximations

| Learner | State at width 65 | Shuffled | Repeated plain | Augmented | Class ordered |
|---|---:|---:|---:|---:|---:|
| LMS | 42.1 KB | 67.33% | 73.67% | 67.33% | 10.00% |
| Exact RLS | 75.1 KB | **90.83%** | **92.08%** | **88.25%** | **88.83%** |
| Diagonal RLS | 42.6 KB | 77.83% | 79.58% | 74.67% | 10.08% |
| Block RLS, width 16 | 50.1 KB | 86.00% | 88.33% | 82.25% | 47.17% |
| Class prototypes | 42.2 KB | 67.17% | 67.17% | 64.25% | 67.17% |
| Protected prototype + fast LMS | 47.2 KB | 45.58% | 40.50% | 42.42% | 10.00% |

Block RLS is the best approximation so far: it saves one third of exact RLS's
total model state and loses 4.8 percentage points shuffled. It does not preserve
enough cross-block information for the ordered stream. Diagonal RLS demonstrates
more sharply that correlations—not just per-feature learning rates—are central
to RLS's retention.

The prototype memory cannot match RLS generalization, but unlike LMS it does not
collapse under class ordering. The first protected fast/slow hybrid is a failed
intervention: its fast component dominates the stable prototype prediction.
Future work must gate or route fast memory rather than simply add it to the slow
prediction.

Repeating the original stream improves every adaptive linear readout more than
the chosen translation/noise augmentation. This confirms that the earlier
augmented comparison mixed augmentation effects with increased exposure.

### RLS forgetting factor

The heuristic effective history of exponentially weighted RLS is
`1 / (1 - factor)`. Factor 1.0 has no exponential forgetting.

| Factor | Approx. history | Shuffled | Ordered | Original retained after inversion | Inverted learned | Original after return | Inverted retained after return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | unlimited | 90.08% | **90.08%** | **84.42%** | 78.58% | 87.75% | 74.00% |
| 0.9999 | 10,000 | 90.17% | 89.92% | 83.33% | 80.17% | 88.08% | **74.50%** |
| 0.999 | 1,000 | **90.83%** | 88.83% | 73.33% | **86.08%** | **89.33%** | 68.00% |
| 0.99 | 100 | 88.58% | 37.08% | 8.42% | 84.50% | 88.17% | 7.67% |
| 0.95 | 20 | 70.92% | 12.83% | 9.75% | 67.08% | 66.08% | 10.58% |

There is no universally best factor. `1.0` and `0.9999` are stable memories;
`0.999` adapts more strongly while retaining useful old performance; `0.99`
behaves like a fast context-specific memory and almost completely overwrites the
previous context. `0.95` forgets too quickly even for this stream.

### Lazy 60,000-image scaling

Fashion-MNIST images are 28x28, with 60,000 training examples. The scaling
benchmark reuses one all-zero image—512 bytes at 8x8 or 6,272 bytes at 28x28—so
it exercises the learner without allocating or downloading a dataset.

| Learner | 8x8 images/s at 60k | 8x8 state | 28x28 images/s at 60k | 28x28 state | Bounded |
|---|---:|---:|---:|---:|---:|
| LMS | 8,458 | 42.1 KB | 2,521 | 52.1 KB | yes |
| Exact RLS | 7,318 | 75.1 KB | 2,419 | 85.1 KB | yes |
| Diagonal RLS | 8,049 | 42.6 KB | 2,489 | 52.6 KB | yes |
| Block RLS | 6,334 | 50.1 KB | 2,296 | 60.1 KB | yes |
| Prototypes | 4,428 | 42.2 KB | 1,283 | 52.2 KB | yes |
| Protected fast/slow | 3,861 | 47.2 KB | 1,162 | 57.2 KB | yes |

Throughput is nearly constant from 1,000 through 60,000 images for every
learner, and state size never changes. The 28x28 stream is about three times
slower because it performs 28 recurrent row steps instead of eight. At hidden
width 64, recurrent computation dominates enough that exact RLS is only about
4% slower than LMS in the 28x28 end-to-end run.

### Feature-width scaling

The readout-only benchmark isolates the dimension that makes RLS quadratic.

| Width | Exact RLS updates/s | Exact state | Diagonal updates/s | Diagonal state | Block updates/s | Block state |
|---:|---:|---:|---:|---:|---:|---:|
| 65 | 39,180 | 38.1 KB | 84,647 | 5.6 KB | 22,412 | 13.1 KB |
| 129 | 16,585 | 140.1 KB | 77,323 | 11.1 KB | 13,181 | 26.1 KB |
| 257 | 4,655 | 536.1 KB | 66,027 | 22.1 KB | 7,310 | 52.1 KB |
| 513 | 804 | 2.05 MB | 35,429 | 44.1 KB | 3,605 | 104.1 KB |

At a projected width of 4,097, exact RLS requires about 128.4 MB of float64
readout state, versus 0.34 MB for diagonal RLS and 0.81 MB for 16-wide block
RLS. These projections are analytic state counts; throughput was measured only
through width 513.

### CPU versus accelerator interpretation

The current LMS/RLS/BPTT throughput comparison uses tiny models, batch size one,
and CPU execution. It does not predict GPU ordering. BPTT can use accelerator
parallelism much more effectively with wider models, longer sequences, or
batches; batch-one tiny models may instead be dominated by launch and
synchronization overhead. LMS and RLS can also use accelerator matrix kernels,
especially at larger widths. A future GPU comparison must report both strict
per-event updates and a separately labeled batched-throughput mode.

## Acceptance gates

1. **Mechanical correctness:** passed; core code has no autograd dependency.
2. **Online learning:** passed; LMS and RLS beat the frozen control.
3. **Temporal credit:** passed narrowly; eligibility improved delayed MSE across
   all five tested seeds.
4. **Bounded memory:** passed for persistent learner state; BPTT retained state
   increased with window length.
5. **Continual stability:** passed as a measured tradeoff; fast/slow weights
   improved retention with a small accuracy cost.

## Factor-free cumulative representation memory

The `memory` branch replaces age-based discounting with complementary
representations:

- The **slow representation** is exact cumulative ridge regression. Every
  observation enters with weight one and remains in its sufficient statistics.
- Per-class semantic centroids provide a second stable representation.
- The **fast representation** receives only cases the slow path misclassifies,
  enforcing specialization. Errors with the same target/predicted-class pair
  are compressed into a cumulative centroid rather than retained as images.
- A cumulative ranker records which path was correct in contexts defined by the
  two proposed classes, slow confidence, and relative representational
  proximity. Its counts never decay.

The implementation contains no forgetting-factor or decay field and stores no
raw samples. A mechanical test confirms that its sequential slow weights equal
the closed-form batch ridge solution over all observations to numerical
precision. With ten classes, 65 readout features, and 16 rank bins, its state is
fixed at 731.8 KB regardless of stream length. The rank-resolution cost is
large but depends on classes, features, and bins—not the number of examples.

Results are means across seeds 7, 17, and 29:

| Learner | Shuffled final | Ordered online | Ordered final | Ordered forgetting | State |
|---|---:|---:|---:|---:|---:|
| Cumulative fast/slow memory | 89.75% | **80.43%** | 86.08% | 0.086 | 731.8 KB |
| RLS, no discount | **90.08%** | 77.00% | **90.08%** | **0.038** | 75.1 KB |
| Class prototypes | 67.17% | 80.12% | 67.17% | 0.147 | 42.2 KB |
| Prior protected fast/slow | 45.58% | 99.36% | 10.00% | 0.900 | 47.2 KB |

Every cumulative-memory run incorporated all 1,397 training observations into
the slow statistics and stored zero images. The shuffled stream created 53
active error representations on average, compressing 203 slow errors. The
ordered stream created 38 representations, compressing 321 errors. The ranker
selected fast memory for 1.8% of shuffled and 4.2% of ordered training
predictions.

The fast path therefore produces a real but unresolved tradeoff. It improves
immediate ordered-stream accuracy by 3.43 percentage points over its slow-path
baseline, while reducing final held-out retention by 4.00 points. Finer rank
contexts reduced the damage during development, but increased state and did not
eliminate it. This makes routing—not cumulative storage—the next bottleneck.

The original/inverted/original test supports the same conclusion:

| Learner | After inversion: original | After inversion: inverted | Final original | Final inverted |
|---|---:|---:|---:|---:|
| Cumulative fast/slow memory | 82.75% | 78.58% | 87.08% | 73.83% |
| RLS, no discount | **84.42%** | 78.58% | **87.75%** | **74.00%** |
| RLS, factor 0.999 | 73.33% | **86.08%** | 89.33% | 68.00% |

The discounted comparator adapts more aggressively by sacrificing the prior
domain. The cumulative systems keep both domains useful. The new fast path does
not yet improve on the no-discount slow baseline in this drift test, so the
experiment establishes the architecture and invariant but not a quality win.

Run one deterministic seed with:

```bash
PYTHONPATH=src python3 -m no_backprop cumulative-memory \
  --output results/cumulative-memory.json
```

### Single-path maturity network: entropy ablation

The next intervention removes expert routing. Both matched variants use one
prediction head over one shared representation:

```text
reservoir features + all active local neurons -> cumulative ridge prediction
```

The model reserves capacity for 32 radial neurons. A prediction error can
recruit a neuron centered on the current representation, provided it is not a
duplicate of an active center. Every active neuron participates continuously in
the same feature vector; there is no fast/slow winner. Recursive covariance
makes new directions highly plastic, while cumulative activation evidence makes
them progressively stable. All weights and evidence use unit-weight updates
without decay.

The entropy model is otherwise identical. It recruits on an error only when the
prediction's normalized categorical entropy is below the cumulative mean
entropy of correct predictions. Thus uniform startup errors train the shared
base but do not immediately consume structural capacity. This is a relative,
cumulative criterion, not an age window.

Results are means across seeds 7, 17, and 29, using localized neurons with RBF
width 0.05:

| Learner | Shuffled online | Shuffled final | Ordered online | Ordered final | Ordered forgetting | State |
|---|---:|---:|---:|---:|---:|---:|
| RLS, no discount | 85.45% | 90.08% | **77.00%** | 90.08% | **0.0383** | 75.1 KB |
| Routed cumulative memory | 85.25% | 89.75% | 80.43% | 86.08% | 0.0858 | 731.8 KB |
| Maturity, no entropy | 85.25% | 90.17% | 76.83% | 89.67% | 0.0408 | 135.4 KB |
| Maturity, entropy | **85.64%** | **90.42%** | 76.93% | **90.33%** | 0.0392 | 135.4 KB |

The entropy variant improves over its matched non-entropy control by 0.38
points shuffled-online, 0.25 shuffled-final, 0.10 ordered-online, and 0.67
ordered-final. It also finishes 0.33 points above no-discount RLS shuffled and
0.25 points above it ordered. These are small three-seed differences, not yet a
statistically established advantage.

The recruitment behavior is less ambiguous. The non-entropy model fills all 32
neurons in both protocols. Entropy recruits 22.3 neurons on average shuffled and
9.0 ordered, rejecting 176.7 and 312.7 errors respectively as high-entropy
recruitment events. Those observations still update the shared cumulative
model; only structural expansion is suppressed. No variant stores raw samples.

Localization is essential. With the earlier broad width of 0.2, the
non-entropy model reaches 97.85% ordered online accuracy but only 15.67% final
accuracy. Broad neurons adapt by activating across established classes. Entropy
reduces that failure to 84.99% online and 60.83% final, but cannot make broad
features stable. Narrow width 0.05 restores approximately 90% final retention.

### RLS-leverage recruitment gate

Entropy is an indirect novelty estimate computed from uncalibrated output
scores. The leverage variant instead uses the inverse correlation matrix that
already drives RLS. Before each update it computes

\[
h = x^\mathsf{T} P x, \qquad \tilde h = \frac{h}{1+h}.
\]

High leverage means the representation is unfamiliar. A prediction error may
recruit a neuron only when its normalized leverage is below the cumulative
pre-update mean. Thus an unfamiliar startup error first updates the shared
model; recruitment becomes possible if the region remains erroneous after it
becomes familiar. The mean includes every prior observation with unit weight,
so there is no hand-set threshold, forgetting factor, or age window.

The fixed-key comparison uses 10 matched seeds (3, 7, 11, 17, 23, 29, 37, 41,
47, and 53):

| Recruitment | Shuffled online | Shuffled final | Ordered online | Ordered final | Ordered forgetting |
|---|---:|---:|---:|---:|---:|
| Ungated | 85.11% | 90.78% | 76.90% | 90.25% | 4.13 points |
| Entropy | **85.32%** | 90.80% | 76.56% | 90.43% | 4.10 points |
| RLS leverage | 85.29% | **90.95%** | **77.24%** | **90.78%** | **3.73 points** |

Against the ungated control, leverage changes shuffled online by +0.18 points,
shuffled final by +0.18, ordered online by +0.34, ordered final by +0.53, and
ordered forgetting by -0.40. Paired 95% Student-t intervals respectively have
half-widths of 0.45, 0.32, 0.50, 0.48, and 0.45 points. Only ordered final
narrowly excludes zero, and these nominal intervals are not corrected for the
multiple comparisons. Entropy remains mixed at 10 seeds rather than becoming a
clear improvement.

Leverage rejects 5.1 early errors on average shuffled and 17.3 ordered, but
still fills all 32 neuron slots. Entropy rejects 172.8 and 320.4 errors and
finishes with 24.1 and 6.6 active neurons. Leverage therefore improves which
early observations become centers; it does not solve capacity allocation.
Every rejected observation still updates both the cumulative output model and
the leverage baseline.

On the 10-seed original/inverted/original benchmark, leverage finishes at
88.58% original and 75.88% inverted versus 88.20% and 75.93% for ungated
maturity. The +0.38-point original-domain change has a nominal paired interval
of approximately +0.02 to +0.74 points; inverted performance is unchanged.
This supports leverage as a modestly better recruitment default, not as a major
adaptation breakthrough.

### Probationary frozen-key experiment

Probation separates learning a key from using it. A leverage-qualified error
creates a dormant candidate that cannot affect prediction or RLS statistics. A
later nearby observation with the same label confirms the proposal; their
cumulative centroid becomes an active RBF key and is then immutable. The
confirming observation immediately trains the newly exposed coordinate. There
is one structural rule—proposal plus confirmation—but no promotion score,
decay, raw-sample buffer, or moving active basis.

An initial sanity design required the confirmation itself to be another error.
That stranded all 32 candidate slots and promoted only 7–18 keys on the three
original seeds. The final design accepts any later nearby same-label
observation as evidence that the proposed region persists. This rule was then
frozen before the 10-seed comparison.

| Model | Shuffled online | Shuffled final | Ordered online | Ordered final | Ordered forgetting | State |
|---|---:|---:|---:|---:|---:|---:|
| Leverage, immediate key | 85.29% | 90.95% | 77.24% | **90.78%** | **3.73 points** | 135.4 KB |
| Leverage + probation | **85.74%** | **90.98%** | **77.72%** | 90.58% | 3.98 points | 152.4 KB |

Probation improves shuffled online accuracy by 0.45 points with a nominal
paired 95% interval of +0.04 to +0.87, and ordered online by 0.47 points with an
interval of +0.24 to +0.71. Shuffled final changes by +0.03 points and ordered
final by -0.20; both intervals cross zero. Ordered forgetting changes by +0.25
points and is also inconclusive. Runtime is unchanged within measurement noise.

Both variants eventually activate all 32 keys. Probation creates 60.6
candidates on average shuffled and 45.0 ordered, finishing with 28.6 and 13.0
dormant candidates respectively. The shuffled candidate pool rejects 17.6
additional proposals on average; the ordered pool rejects none. Its fixed-size
candidate bank raises state by 12.6%, from 135.4 to 152.4 KB.

On original/inverted/original drift, probation changes performance after
inversion by -0.10 points original and +0.30 inverted. Final performance changes
by -0.13 points original and -0.03 inverted. Every paired interval crosses
zero, so there is no detected drift benefit or stability regression. The result
is specifically a modest improvement in online sample efficiency: averaging a
confirmed key is better than exposing the first qualifying error immediately.

### Local-responsibility experiment

The local-memory representation is made dynamically sparse by retaining only
the strongest key activations for each observation. The reservoir state and
base RLS features remain dense. Every policy is used from the start of training;
no model is trained dense and switched afterward. The comparison covers dense,
unnormalized top-4, top-2, winner-only, and normalized top-4/top-2 activation.

Ten-seed means are:

| Responsibility | Shuffled online | Shuffled final | Ordered online | Ordered final | Mean active keys (shuf./ord.) |
|---|---:|---:|---:|---:|---:|
| Dense probation | **85.74%** | **90.98%** | 77.72% | **90.58%** | 32.0 / 32.0 |
| Top-4 | 85.68% | 90.85% | 77.70% | 90.50% | 3.89 / 3.08 |
| Top-2 | 85.71% | 90.70% | 77.65% | 90.55% | 1.96 / 1.56 |
| Winner-only | 85.63% | 90.55% | 77.39% | 90.45% | 0.99 / 0.79 |
| Normalized top-4 | 83.71% | 89.85% | **86.30%** | 67.65% | 3.89 / 3.01 |
| Normalized top-2 | 82.91% | 88.78% | 85.40% | 66.23% | 1.96 / 1.55 |

Unnormalized sparsity does not improve quality. Top-4 differs from dense by
-0.06 points shuffled-online, -0.13 shuffled-final, -0.01 ordered-online, and
-0.08 ordered-final; all paired intervals include zero. Winner-only reduces
ordered-online accuracy by 0.32 points with a nominal paired interval of -0.58
to -0.06, and trends toward worse retention.

This NumPy implementation also receives no compute benefit from zeroing key
features: exact RLS still multiplies the full inverse-correlation matrix, while
top-k selection adds roughly 9–10% training time. Sparse responsibility becomes
a systems optimization only if paired with a sparse/block update rule.

Normalized local attention is a clear failure. It raises ordered online
accuracy by 8.58 points for top-4 and 7.69 for top-2, but lowers ordered final
accuracy by 22.93 and 24.35 points. When all matches are weak, normalization
still forces them to sum to one; class ordering then produces a large transient
response that overwhelms stable representation. No minimum-similarity threshold
was added after seeing this result.

Dense probation remains the foundation for the next experiment. The sparse
variants stay available as matched negative controls and as possible inputs to
later sparse-RLS scaling work.

### Candidate-capacity management

The unmanaged probation bank can strand one-observation proposals until all 32
candidate slots are occupied. Managed probation never replaces an active frozen
key. Under candidate-pool pressure it first reclaims a dormant center that the
current model now classifies correctly. If none is resolved, a new proposal may
replace the bank's most novel candidate only when the new proposal has lower
normalized RLS leverage. Otherwise it is rejected. All associated observations
remain in cumulative RLS statistics; only provisional structural summaries are
reused, with no age score or forgetting factor.

Ten-seed pool-size results are:

| Candidate policy | Shuffled online | Shuffled final | Ordered online | Ordered final | Active keys (shuf./ord.) | State |
|---|---:|---:|---:|---:|---:|---:|
| Unmanaged 32 | 85.74% | 90.98% | 77.72% | 90.58% | 32.0 / 32.0 | 152.4 KB |
| Managed 32 | **85.84%** | 90.93% | 77.72% | 90.58% | 32.0 / 32.0 | 152.7 KB |
| Managed 16 | 85.75% | **91.08%** | 77.72% | 90.58% | 32.0 / 32.0 | 144.1 KB |
| Managed 8 | 85.60% | 90.73% | **78.05%** | 90.50% | 24.8 / 32.0 | 139.8 KB |

Managed-16 is the useful operating point. Relative to unmanaged probation its
paired changes are +0.01 points shuffled-online, +0.10 shuffled-final, and
exactly zero for both ordered metrics; every interval includes zero. It reclaims
48.8 now-resolved candidates and makes 19.0 lower-novelty replacements on
average shuffled, reducing outright pool rejection from 17.6 events to 2.3.
Ordered data needs only 1.6 reclaims and no novelty replacements. State falls
5.5%, from 152.4 to 144.1 KB, and runtime is unchanged within noise.

Managed-8 is too restrictive: it activates only 24.8 keys on average shuffled
and increases forgetting by 0.63 points with a nominal paired interval of +0.17
to +1.08. Managed-32 removes rejection but adds state and slightly worsens mean
shuffled forgetting, so it offers no advantage over 16 slots.

On the 10-seed drift benchmark, managed-16 changes final original accuracy by
+0.10 points and final inverted accuracy by +0.13; all paired intervals cross
zero. Managed-16 therefore advances as the bounded candidate foundation for
the fast/slow-value experiment.

### Evidence-consolidated fast/slow values

Each confirmed key receives an optional fast output column in addition to its
ordinary cumulative RLS slow column. The prediction uses their sum, so there is
one path and no fast/slow router. Per-key scalar RLS state updates the fast
column from the residual left after the slow update; there is no learning-rate
parameter or decay.

In the consolidated variant, a key transfers its entire fast column into its
slow column whenever cumulative activation evidence doubles. The fast column is
then cleared and its scalar inverse correlation reset. Because the transfer is

\[
v_j^{slow} \leftarrow v_j^{slow}+v_j^{fast}, \qquad
v_j^{fast} \leftarrow 0,
\]

the total value and every prediction are exactly unchanged at consolidation.
All tests and experiment diagnostics report zero numerical prediction shift.
The non-consolidating fast-value model is the matched ablation.

Ten-seed results are:

| Value rule | Shuffled online | Shuffled final | Ordered online | Ordered final | Ordered forgetting | State |
|---|---:|---:|---:|---:|---:|---:|
| Managed-16 slow only | **85.75%** | **91.08%** | 77.72% | **90.58%** | **3.98 points** | 144.1 KB |
| + fast values | 85.58% | 90.73% | 79.52% | 88.70% | 6.80 points | 147.4 KB |
| + fast values and consolidation | 85.52% | 90.70% | **79.93%** | 87.30% | 8.28 points | 147.4 KB |

Fast values produce the intended plasticity but fail the stability gate. Against
slow-only, fast values gain 1.80 ordered-online points with a paired 95%
half-width of 0.71, but lose 1.88 final points with half-width 1.77 and add 2.83
forgetting points. Shuffled final also falls 0.35 points. State rises 2.3% and
training time rises roughly 9–13%.

Consolidation adds another 0.41 ordered-online points over fast-only, but loses
another 1.40 final points and adds 1.48 forgetting points; each paired interval
excludes zero. Although transfer itself is lossless, resetting local inverse
correlation makes subsequent residual updates aggressive again. Exact
consolidation therefore preserves current information but changes future
plasticity in a destabilizing way.

On drift, fast and consolidated models retain 0.78 and 0.83 fewer original
points immediately after inversion, while inverted accuracy changes by only
-0.30 and -0.35 points. Their larger recovery reflects having more to recover;
final original and inverted metrics match slow-only within uncertainty. Neither
fast-value variant advances to scaling, but both remain reproducible controls.

### Adaptive key-value neuron experiment

The attention-inspired variant keeps the same single output path but makes each
recruited local neuron an explicit key-value unit. The reservoir feature is the
query; the neuron's moving center is its key; a learned diagonal variance
defines an unnormalized Gaussian compatibility score; and the corresponding
column of the cumulative output matrix is its value. Activation-weighted
Welford updates learn keys and widths locally. Accumulated evidence makes center
movement progressively smaller, with no gradient, backward pass, raw-sample
buffer, forgetting factor, or prediction router.

An unconstrained width experiment increased ordered online adaptation but was
unstable: widths approaching 0.1 reduced seed-29 ordered final accuracy to
80.75%. The checked-in variant starts at width 0.05 and caps learned widths at
0.06. Three-seed means (seeds 7, 17, and 29) are:

| Learner | Shuffled online | Shuffled final | Ordered online | Ordered final | State |
|---|---:|---:|---:|---:|---:|
| Fixed keys, no entropy | **85.25%** | **90.17%** | 76.83% | 89.67% | 135.4 KB |
| Adaptive keys, no entropy | 85.18% | 89.92% | **77.98%** | **89.75%** | 168.1 KB |
| Fixed keys, entropy | **85.64%** | **90.42%** | 76.93% | **90.33%** | 135.4 KB |
| Adaptive keys, entropy | 85.30% | **90.42%** | **77.07%** | 90.00% | 168.1 KB |

The non-entropy adaptive model gains 1.15 points of ordered online accuracy
without a material mean final-retention change. It is not an overall quality
win: shuffled accuracy is slightly lower, entropy does not combine
constructively with key adaptation, state rises 24%, and training takes about
6% longer than the fixed-key non-entropy control after vectorizing the local key
updates (0.599 versus 0.568 seconds averaged over the six protocol/seed runs).

On original/inverted/original drift, adaptive keys average 5.25 points of
original-domain forgetting after inversion versus 5.33 for fixed keys, recover
4.17 versus 3.50 points on return, and finish at 88.33% original / 76.67%
inverted versus 88.67% / 75.75%. This is a small shift toward plasticity, not a
clear dominance result.

The central limitation is basis drift. Each observation remains represented in
cumulative correlation statistics, but those historical activations were
computed at earlier key positions. Without raw replay, moving a nonlinear key
cannot retroactively recompute them in the new basis. Maturity slows this drift
and the width cap bounds its reach; neither provides an exact correction.

The original/inverted/original drift benchmark is more favorable to the unified
representation:

| Learner | After inversion: original | After inversion: inverted | Final original | Final inverted |
|---|---:|---:|---:|---:|
| RLS, no discount | 84.42% | 78.58% | 87.75% | 74.00% |
| Maturity, no entropy | **85.17%** | 79.42% | **88.67%** | **75.75%** |
| Maturity, entropy | 85.08% | **79.75%** | **88.67%** | 74.75% |

Both maturity models retain more of the original domain while learning the
inverted representation better than no-discount RLS. Entropy is slightly better
immediately after inversion; the non-entropy model retains one more point of
inverted accuracy after the original domain returns.

The result supports a single-path expanding representation over explicit
routing, while the adaptive experiment shows that local receptive fields can be
learned without backpropagation. Entropy still comes from uncalibrated ridge
scores whose mean is near its maximum. Future work should address basis drift,
calibrate predictive uncertainty, and scale the neuron bank.

### Matched non-recurrent image frontends

The next experiment changes only image ingestion. All models expose 64 features
plus a bias coordinate to the same Managed-16 readout. The recurrent control
processes eight ordered rows. The pixel control flattens the complete 8x8 image.
The convolutional control applies four seeded, fixed, zero-mean orthogonal 3x3
filters, `tanh`, and 2x2 average pooling to produce exactly 64 features. Neither
spatial frontend has trainable encoder state, and every model receives one
label update per image. The design was fixed before the ten-seed comparison.

| Protocol / frontend | Online | Final | Forgetting | Images/s | State |
|---|---:|---:|---:|---:|---:|
| Shuffled / recurrent | 85.75% | 91.08% | 1.55 points | 2,342 | 144.1 KB |
| Shuffled / pixels | 90.69% | 93.53% | 2.00 points | **8,850** | **107.1 KB** |
| Shuffled / fixed conv | **91.10%** | **94.28%** | **1.50 points** | 5,309 | 107.4 KB |
| Augmented / recurrent | 68.67% | 88.65% | 2.10 points | 2,204 | 144.1 KB |
| Augmented / pixels | 66.54% | 91.48% | 2.35 points | **6,312** | **107.1 KB** |
| Augmented / fixed conv | **69.82%** | **92.90%** | **1.53 points** | 4,303 | 107.4 KB |
| Ordered / recurrent | 77.72% | 90.58% | 3.98 points | 2,547 | 144.1 KB |
| Ordered / pixels | **82.18%** | 93.55% | 3.05 points | **7,465** | **107.1 KB** |
| Ordered / fixed conv | 81.85% | **94.43%** | **2.75 points** | 4,783 | 107.4 KB |

Against recurrence, fixed convolution gains 5.35 ± 0.97 shuffled-online,
3.20 ± 1.10 shuffled-final, 1.15 ± 0.86 augmented-online, 4.25 ± 1.03
augmented-final, 4.14 ± 0.86 ordered-online, and 3.85 ± 1.12 ordered-final
percentage points. These are paired means and nominal 95% Student-t
half-widths across seeds 3, 7, 11, 17, 23, 29, 37, 41, 47, and 53. Every
accuracy interval excludes zero. The convolutional frontend also lowers
ordered forgetting by 1.23 ± 0.78 points.

The recurring-domain result reverses that ranking:

| Frontend | Original forgetting after inversion | Final original | Final inverted |
|---|---:|---:|---:|
| Recurrent | **4.33 points** | 88.55% | **75.98%** |
| Pixels | 65.65 points | **93.35%** | 0.73% |
| Fixed conv | 37.48 points | 89.33% | 10.58% |

After the original domain returns, pixels and fixed convolution retain 75.25 ±
2.25 and 65.40 ± 2.66 fewer inverted-domain points than recurrence. Removing
row recurrence is therefore an ordinary-classification and systems win, not yet
a complete continual-representation win. The spatial frontend advances to the
predictive-representation experiment because it preserves image locality and
parallelism; the recurrent model remains a required drift control until a
learned spatial representation recovers coexistence of both domains.

### Forward-only predictive representation

The first JEPA-inspired experiment keeps the fixed convolutional target space
and learns only a masked latent predictor. Its 4x4x4 feature map is divided
into four 2x2x4 quadrants. For each quadrant, the shared predictor receives the
other three quadrants, a one-hot target-position code, and a bias. Four
predictions are reassembled into a 64-coordinate representation for the same
Managed-16 classifier used by the controls. The classification prediction is
made before either the label update or the four self-supervised predictor
updates.

The predictor is exact cumulative RLS with forgetting factor 1.0. Every image
contributes four unit-weight updates, no raw example is retained, held-out
evaluation mutates no weights, and neither encoder nor classifier uses
backpropagation. This is a forward-only masked-prediction probe, not an I-JEPA
implementation: the target encoder is fixed and there is no learned target
network, stop-gradient operation, or exponential-moving-average copy.

| Protocol / representation | Online | Final | Forgetting | Images/s | State |
|---|---:|---:|---:|---:|---:|
| Shuffled / fixed conv | **91.10%** | **94.28%** | 1.50 points | **5,298** | **107.4 KB** |
| Shuffled / predictive | 86.44% | 93.00% | **1.45 points** | 2,584 | 153.2 KB |
| Augmented / fixed conv | **69.82%** | **92.90%** | 1.53 points | **4,360** | **107.4 KB** |
| Augmented / predictive | 65.82% | 91.65% | **1.45 points** | 2,333 | 153.2 KB |
| Ordered / fixed conv | 81.85% | **94.43%** | **2.75 points** | **4,736** | **107.4 KB** |
| Ordered / predictive | **84.27%** | 76.98% | 21.53 points | 2,740 | 153.2 KB |

Across the same ten paired seeds, predictive minus fixed convolution is +2.42
± 0.63 percentage points for ordered online accuracy, but -17.45 ± 1.81 for
ordered final accuracy and +18.78 ± 1.81 for forgetting. On shuffled and
augmented streams it loses 4.66 ± 0.81 and 4.01 ± 0.80 online points, then
finishes 1.28 ± 0.67 and 1.25 ± 0.79 points lower. The intervals are nominal
95% Student-t half-widths and are not corrected for multiple comparisons.

The self-supervised objective itself is working:

| Protocol | Initial target MSE | Final target MSE | Effective rank / 64 |
|---|---:|---:|---:|
| Shuffled | 0.1426 | **0.0280** | 8.37 |
| Augmented | 0.1426 | **0.0321** | 8.89 |
| Ordered | 0.1426 | **0.0280** | 8.37 |

The fixed-convolution control has effective rank 11.88 of 64. The predictor's
large loss reduction and nonzero rank rule out a constant-output failure,
although its lower rank shows additional compression. The downstream failure
is instead consistent with basis drift: cumulative classifier statistics bind
labels to predicted coordinates that continue moving as the predictor learns.
This is especially visible in class order, where the moving basis helps current
examples while invalidating older label associations.

The recurring-domain benchmark provides a weaker partial benefit but does not
solve coexistence:

| Representation | Original forgetting after inversion | Final original | Final inverted |
|---|---:|---:|---:|
| Recurrent control | **4.33 points** | **88.55%** | **75.98%** |
| Fixed convolution | 37.48 points | 89.33% | 10.58% |
| Predictive convolution | 59.20 points | 85.25% | 17.73% |

Predictive coordinates retain 7.15 ± 4.49 more inverted-domain points than
fixed convolution, but lose 4.08 ± 2.02 original-domain points and forget
21.73 ± 4.09 more original-domain points during inversion. They remain 58.25
points behind recurrence on final inverted accuracy. The experiment therefore
fails the stability and efficiency gates and remains a diagnostic control. A
follow-up should preserve downstream coordinate identity through frozen,
append-only, or explicitly consolidated predictive features before adding a
learned target encoder.

### Contrast-polarity geometry ablation

The inversion result above does not imply that recurrent state is inherently
better at changing information. Reservoir state is reset before every image,
so it cannot carry experience between examples. The likely confound is the
fixed convolution's geometry. Its kernels have zero mean and its activation is
odd, making interior responses approximately satisfy
`tanh(k * (1 - x)) = -tanh(k * x)`. The same digit and its inversion therefore
become opposing inputs to one cumulative classifier.

Two fixed 64-coordinate controls test that explanation. Absolute convolution
applies four filters and retains only response magnitudes. Signed-magnitude
convolution applies two filters to produce 32 signed coordinates and
concatenates their 32 magnitudes. The latter preserves polarity information
while giving each example a shared even component. It has no mixing coefficient
or tuned threshold: signed and magnitude channels receive equal representation
width. Both use the exact Managed-16 readout, one update per image, locked
evaluation, and no gradients or stored examples.

| Encoder | Original/inverted cosine | Per-seed cosine gap from recurrence |
|---|---:|---:|
| Recurrent reservoir | 0.231 | -- |
| Fixed convolution | -0.473 | 0.704 |
| Signed-magnitude convolution | **0.127** | **0.145** |
| Absolute convolution | 0.724 | 0.492 |

These are ten-seed means. The recurrent cosine is seed-sensitive (sample
standard deviation 0.204), whereas signed-magnitude is more stable (0.044).
Signed-magnitude is the direct answer to the geometry question: it moves
inversion pairs much closer to the reservoir's relationship at the same width.
Absolute convolution targets a different outcome, making the two domains
similar rather than merely non-opposing.

| Protocol / encoder | Online | Final | Forgetting |
|---|---:|---:|---:|
| Shuffled / fixed conv | 91.10% | 94.28% | 1.50 points |
| Shuffled / signed-magnitude | **91.65%** | **94.68%** | 1.50 points |
| Shuffled / absolute | 91.62% | 94.63% | **1.30 points** |
| Augmented / fixed conv | 69.82% | **92.90%** | **1.53 points** |
| Augmented / signed-magnitude | 73.18% | 92.40% | 2.08 points |
| Augmented / absolute | **73.52%** | 92.53% | 1.63 points |
| Ordered / fixed conv | 81.85% | 94.43% | 2.75 points |
| Ordered / signed-magnitude | 84.25% | **94.58%** | 2.75 points |
| Ordered / absolute | **84.94%** | **94.58%** | **2.63 points** |

Relative to fixed convolution, absolute and signed-magnitude improve augmented
online accuracy by 3.70 ± 1.46 and 3.36 ± 1.18 percentage points and ordered
online accuracy by 3.09 ± 0.95 and 2.40 ± 0.92 points. Their ordinary final
accuracy changes are small and every nominal 95% interval includes zero.

The recurring-domain result changes completely:

| Encoder | Original forgetting after inversion | Final original | Final inverted |
|---|---:|---:|---:|
| Fixed convolution | 37.48 points | 89.33% | 10.58% |
| Recurrent reservoir | 4.33 points | 88.55% | 75.98% |
| Signed-magnitude convolution | 8.95 points | 90.32% | 78.78% |
| Absolute convolution | **3.33 points** | **92.38%** | **88.88%** |

Against fixed convolution, absolute responses reduce original forgetting by
34.15 ± 3.39 points and improve final inverted accuracy by 78.30 ± 2.96.
Signed-magnitude reduces forgetting by 28.53 ± 2.77 points and improves final
inverted accuracy by 68.20 ± 3.59. Absolute convolution also finishes 12.90 ±
2.38 inverted-domain points above recurrence; signed-magnitude finishes 2.80 ±
3.39 above it, an inconclusive difference.

This establishes that the previous 65-point recurrence advantage was primarily
a polarity-representation failure. Absolute convolution wins because inversion
becomes approximately a nuisance symmetry: at the feature level, much of the
"domain change" disappears. That is correct for this benchmark because labels
are invariant to contrast polarity, but it would be harmful wherever polarity
changes meaning. Signed-magnitude is the more informative general control
because it preserves both sign and shared energy. Neither fixed transform
solves arbitrary context shifts or the predictive model's moving-basis problem;
future drift tests must include transformations not removable by a hand-chosen
symmetry.

### Recurring transformation drift suite

The broader suite promotes signed-magnitude convolution to the primary spatial
baseline. It retains the recurrent reservoir as a sequential control, ordinary
convolution as the sign-interference ablation, and absolute convolution as an
invariance diagnostic. Six deterministic transformations preserve the supplied
labels: inversion, contrast scaling toward 0.5, Gaussian noise with standard
deviation 0.16, a centered 2x2 occlusion, a one-pixel down-right translation,
and a 20% alternating-column background.

For each seed and model, the original domain is trained exactly once. That
post-training learner is copied into six branches, each of which follows
`original -> transformed -> original`. Consequently, every transformation for
a model begins with byte-identical learned statistics. All models expose 64
coordinates to the same 32-mature/16-candidate Managed readout, receive one
label update per image, retain no raw samples, and remain locked during every
evaluation. Results use the same ten paired seeds as the frontend experiments.

| Encoder | Mean original forgetting | Final original | Final transformed | Worst transformed | Final joint |
|---|---:|---:|---:|---:|---:|
| Signed-magnitude baseline | 2.32 points | 93.68% | 89.40% | 77.93% | 91.54% |
| Recurrent reservoir | **1.63 points** | 90.39% | 85.13% | 75.63% | 87.76% |
| Fixed convolution | 7.29 points | 93.34% | 78.25% | 10.75% | 85.80% |
| Absolute convolution | 1.68 points | **93.80%** | **91.30%** | **85.00%** | **92.55%** |

The table first averages each seed across transformations and then averages the
ten seed-level values. Relative to recurrence, signed-magnitude gains 3.29 ±
0.57 points final-original, 4.28 ± 0.95 final-transformed, and 3.78 ± 0.65
final-joint accuracy. Recurrence reduces mean original forgetting by 0.69 ±
0.44 points, but starts and finishes at lower accuracy. Relative to ordinary
convolution, signed-magnitude gains 11.15 ± 0.86 transformed and 5.74 ± 0.69
joint points; the inversion failure accounts for most of that difference.

| Transformation | Signed-magnitude | Recurrent | Fixed conv | Absolute conv |
|---|---:|---:|---:|---:|
| Inversion | 78.28% | 75.63% | 10.75% | **88.88%** |
| Low contrast | 95.03% | 88.00% | **96.33%** | 95.63% |
| Gaussian noise | 90.33% | 86.23% | **90.98%** | 90.58% |
| Center occlusion | 92.45% | 87.25% | **92.78%** | 92.50% |
| Translation | **85.23%** | 83.83% | 83.45% | 85.03% |
| Striped background | 95.13% | 89.85% | **95.25%** | 95.23% |

These are final transformed-domain accuracies after the original domain
returns. Signed-magnitude is 1.40 ± 1.18 points above recurrence on translation,
and 4.10--7.03 points above it on noise, occlusion, low contrast, and stripes;
all four latter intervals exclude zero. Its 2.65 ± 3.24 inversion advantage is
inconclusive. Against fixed convolution, signed-magnitude clearly wins
inversion and translation, while fixed convolution is 1.30 ± 0.77 points
better under low contrast and is statistically tied on the remaining three.

Absolute convolution averages 1.90 ± 0.97 transformed and 1.01 ± 0.62 joint
points above signed-magnitude. The per-transformation comparison localizes that
gain: absolute is 10.60 ± 2.92 points better on inversion, while every one of
its other final-transformed differences from signed-magnitude includes zero.
Thus its aggregate advantage is the expected reward for encoding the exact
polarity symmetry, not evidence of a generally richer representation.

The suite supports signed-magnitude as the main spatial baseline. It preserves
the information destroyed by absolute responses, substantially exceeds
recurrence across several unrelated appearance shifts, and avoids the fixed
convolution's catastrophic inversion conflict. Translation remains the hardest
non-polarity transformation at 85.23%, making spatial displacement a useful
target for the next learned representation. The suite still covers
label-preserving covariate shifts, not changing label semantics or real-world
open-ended drift.

### Stable-backbone predictive-surprise recruitment

This experiment removes the first predictive model's moving-coordinate
confound. Classification always uses the fixed 64-coordinate signed-magnitude
map. In parallel, a cumulative RLS predictor masks and predicts each 2x2x4
quadrant. Its pre-update image MSE, divided by cumulative historical predictor
MSE, is used only to rank unresolved probationary candidates when the bounded
16-slot bank is full. It cannot change active coordinates, logits, or the
single label update. There is no forgetting factor below 1, raw-example replay,
backpropagation, or absolute surprise threshold.

Four matched variants use ten paired seeds: the original signed-magnitude
baseline, a predictor that is trained but unused, aligned current-image
surprise, and a one-image-lagged surprise control. The unused predictor is
bit-for-bit equivalent in every reported accuracy metric, isolating the cost of
prediction from the effect of its routing signal.

| Stream | Metric | Baseline | Aligned surprise | Paired change |
|---|---|---:|---:|---:|
| Shuffled | Online | 91.65% | 91.65% | +0.00 points |
| Shuffled | Final | 94.68% | 94.68% | +0.00 points |
| Augmented | Online | 73.18% | 73.13% | -0.05 points |
| Augmented | Final | 92.40% | 92.30% | -0.10 points |
| Class ordered | Online | 84.25% | 84.26% | +0.01 points |
| Class ordered | Final | 94.58% | 94.68% | +0.10 points |

The predictor itself learns: held-out masked-target MSE falls from 0.1351 to
0.0304 on shuffled and ordered streams and to 0.0346 with augmentation. The
signal also materially changes bounded memory management. For example, aligned
surprise averages 113.3 candidate replacements and 440.9 rejections on the
augmented stream. Therefore the near-zero classification result is not caused
by a disconnected signal.

| Drift aggregate | Baseline | Aligned surprise | Paired change (mean ± seed SD) |
|---|---:|---:|---:|
| Final original | 93.68% | 93.58% | -0.10 ± 0.20 points |
| Final transformed | 89.40% | 88.93% | -0.48 ± 0.25 points |
| Final joint | 91.54% | 91.25% | -0.29 ± 0.14 points |
| Worst transformed | 77.93% | 77.03% | -0.90 ± 1.28 points |

Aligned surprise improves only Gaussian-noise final transformed accuracy
(+0.10 ± 0.39 points) and is neutral-to-negative on the other five shifts,
including -0.63 ± 1.00 points on translation. The lagged control is closer to
baseline than aligned surprise on aggregate final transformed accuracy
(-0.14 ± 0.27 points), which is evidence against the current scalar structural
prediction error being a useful recruitment objective.

The mechanism costs 156.9 KB versus 109.8 KB for baseline and processes roughly
2.3--2.5 thousand rather than 4.4--5.2 thousand training images/second on this
CPU. The stable-anchor architecture passes its engineering contract, but this
specific global surprise ranking does not pass the quality/cost gate. A next
attempt should test spatially local residual structure or frozen predictive
coordinates, rather than tune this scalar ranking post hoc.

### Recurring-regime associative-memory capstone

The memory capstone isolates the selected memory rule from representation
changes. Every model receives the same fixed 64-coordinate signed-magnitude
features and the same per-seed stream:

```text
original -> inversion -> translation -> center occlusion
         -> original return -> inversion return
```

After every phase, all four domains are evaluated without learning. The primary
metrics separate first-shift online adaptability, accuracy retained immediately
before a recurring domain returns, online accuracy during that return, and
final mean accuracy over all domains. Results use ten paired seeds, one label
update per image, no raw replay, and fixed-size learner state.

The expanded RLS factor curve shows the expected stability/plasticity exchange:

| RLS factor | Approx. effective history | First-shift online | Pre-return | Return online | Final mean |
|---:|---:|---:|---:|---:|---:|
| 1.0 | all | 68.57% | **73.29%** | 79.77% | **77.51%** |
| 0.99999 | 100,000 | 68.76% | 73.13% | 79.75% | 77.52% |
| 0.9999 | 10,000 | 70.21% | 70.53% | 79.71% | 77.26% |
| 0.999 | 1,000 | 80.10% | 45.06% | 84.35% | 58.71% |
| 0.995 | 200 | **87.54%** | 31.06% | **89.61%** | 29.74% |
| 0.99 | 100 | 86.42% | 23.04% | 67.72% | 25.85% |
| 0.98 | 50 | 61.05% | 16.29% | 38.31% | 18.88% |
| 0.95 | 20 | 17.67% | 10.53% | 20.14% | 11.20% |

Factors through 0.995 initially buy responsiveness by heavily discounting older
evidence. More aggressive factors become unstable and eventually lose both
adaptability and retention; their high across-seed variation is retained rather
than filtered from the result.

All managed models use the same 16-slot probation bank. The capacity below is
permanent mature-neuron capacity; therefore `Managed 32` is the previously
selected Managed-16 architecture (32 mature plus 16 candidate slots).

| Model | First-shift online | Pre-return | Return online | Final mean | State | Images/s |
|---|---:|---:|---:|---:|---:|---:|
| RLS factor 1 | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | 9,893 |
| Immediate maturity 32 | 69.09% | 73.94% | 80.90% | 77.93% | 100.9 KB | 6,297 |
| Managed 8 | 68.78% | 73.68% | 80.52% | 77.81% | 62.0 KB | 6,653 |
| Managed 16 | 69.06% | 74.21% | 81.21% | 78.36% | 76.9 KB | 6,115 |
| Managed 32 | 69.67% | 75.49% | 82.23% | 79.08% | 109.8 KB | 5,335 |
| Managed 64 | **70.22%** | **75.84%** | **82.68%** | **80.36%** | 187.9 KB | 4,403 |

Relative to factor-1 RLS, Managed 32 gains 1.10 ± 0.48 points first-shift,
2.20 ± 1.67 pre-return, 2.46 ± 1.19 return-online, and 1.56 ± 0.50 final mean;
these are paired means with nominal 95% Student-t half-widths, and all exclude
zero. Probation itself matters: Managed 32 exceeds immediate maturity by 0.58 ±
0.36, 1.55 ± 1.38, 1.34 ± 0.95, and 1.14 ± 0.36 points on those same metrics.

The sharpest matched-adaptability comparison is Managed 64 against factor
0.9999 RLS. Their first-shift online accuracies differ by only +0.005 ± 0.87
points, while managed memory gains 5.31 ± 2.18 points pre-return, 2.97 ± 1.52
return-online, and 3.10 ± 1.26 final mean. This places the associative memory
above the observed RLS discount frontier rather than merely choosing a different
point on it.

Capacity is not free. Capacities 8 and 16 fill in every seed, on average during
phases 1.4 and 2.3. Capacity 32 fills in nine of ten seeds and capacity 64 in
five of ten. On return, Managed 32 reactivates on average 14.8 existing neurons
for original images and 11.0 for inversion, while every previously mature center
remains bitwise fixed. Managed 32 uses 2.81 times factor-1 RLS state and trains
46% slower on this NumPy CPU; Managed 64 uses 4.80 times state and trains 55%
slower.

The result supports a narrow but useful claim: bounded probationary associative
memory improves both responsiveness and recurring-domain retention over
cumulative RLS without deliberately discounting any observation. It does not
establish unbounded lifelong capacity, large-image scaling, or superiority to
online backpropagation. Managed 32 is the balanced reference configuration;
Managed 64 demonstrates the higher-cost quality frontier.

### Matched continuous backpropagation comparison

The earlier BPTT study compared recurrent architectures on shuffled images; it
did not explain whether forward-only learning was justified for the selected
recurring-memory system. This final comparison holds the stable
signed-magnitude frontend, six-phase stream, train/test splits, event order,
batch size one, one update per image, and absence of replay fixed. It adds two
ordinary continuously trained backpropagation controls:

- a linear softmax classifier with Adam;
- a 64-unit tanh MLP with Adam, using float64 like the local learners.

Learning rates are selected independently for each architecture from
`0.0001, 0.0003, 0.001, 0.003, 0.01` on development seeds 2, 5, and 13. These
are disjoint from the ten test seeds. The preregistered selection score weights
first-shift online accuracy, pre-return retention, return online accuracy, and
final mean accuracy equally. It selects 0.01 for the linear model and 0.003 for
the MLP. Test results are not used for optimizer selection.

| Model | First-shift online | Pre-return | Return online | Final mean | Persistent state | Images/s |
|---|---:|---:|---:|---:|---:|---:|
| RLS factor 1 | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | **12,203** |
| Managed 32 | 69.67% | 75.49% | 82.23% | 79.08% | 109.8 KB | 6,795 |
| Managed 64 | 70.22% | **75.84%** | 82.68% | **80.36%** | 187.9 KB | 5,432 |
| Online Adam linear | 73.91% | 38.48% | 81.70% | 41.60% | **15.6 KB** | 4,752 |
| Online Adam MLP-64 | **75.50%** | 44.78% | **88.97%** | 65.94% | 115.5 KB | 3,501 |

The backpropagation models confirm that gradients and continuous learning are
compatible: both update after every individual event, and the MLP learns new
and returning domains fastest. The cost is severe interference. Relative to
Managed 32, the state-near MLP gains 5.83 ± 2.10 points first-shift and 6.73 ±
2.03 return-online, but loses 30.71 ± 2.32 points immediately before return and
13.14 ± 5.31 final mean. Values are paired means with nominal 95% Student-t
half-widths; every difference excludes zero.

The MLP stores 115.5 KB persistently—5.2% more than Managed 32—and its maximum
tracked saved activations add 1.1 KB. Managed 32 processes 1.94 times as many
CPU training images per second in this batch-one float64 comparison. The small
linear model uses only 15.6 KB plus 0.6 KB of saved activations, but its final
mean accuracy is 37.48 ± 4.96 points below Managed 32. These are implementation-
and-CPU-specific measurements, not GPU or energy claims.

This supports choosing forward-only cumulative memory for the project's strict
no-replay recurring-regime objective: it gives up some immediate plasticity for
substantially stronger persistence and final joint performance, while the
state-near Adam alternative is slower here. The comparison does not isolate
gradient use inside an otherwise identical architecture, nor cover replay,
EWC-style regularization, expandable backpropagation networks, or other
specialized continual-learning methods. Therefore the defensible conclusion is
that this forward-only design beats ordinary matched online Adam under the
tested contract—not that backpropagation is inherently unsuitable for
continual learning.

### Closest analytic and constructive baselines

The closest-prior-art audit identified four missing forward-only controls. They
now use the same fixed signed-magnitude frontend, six-phase recurring stream,
event order, one update per image, locked evaluation, and no raw replay as the
capstone. Each method is preallocated to the largest structural capacity that
fits under either Managed-32's 109,800-byte state ceiling or Managed-64's
187,880-byte ceiling:

- ALD kernel RLS admits sufficiently novel feature vectors to an RBF dictionary
  and performs reduced KRLS updates when the dictionary is full; it never
  evicts a mature dictionary element.
- The direct-link Resource-Allocating Network fits a normalized-LMS linear/RBF
  output and recruits fixed centers when both error and feature distance are
  large.
- fast-learning Fuzzy ARTMAP uses fixed complement coding, vigilance, and match
  tracking to create label-consistent categories.
- Online Sequential ELM uses fixed random tanh features and an event-wise
  regularized RLS output update.

Method parameters are selected at the smaller state ceiling on development
seeds 2, 5, and 13 using the equal mean of the four displayed quality metrics.
Those choices are frozen before paired evaluation on test seeds 3, 7, 11, 17,
23, 29, 37, 41, 47, and 53. The selected parameters are KRLS width 0.2 and ALD
tolerance 0.01; RAN learning rate 0.1, error threshold 0.75, and distance
threshold 0.05; ARTMAP vigilance 0.5 and choice 0.001; and OS-ELM input scale
1.0 and ridge regularization 1.0.

| Model | Capacity | First-shift online | Pre-return | Return online | Final mean | State | Images/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| RLS factor 1 | — | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | **9,057** |
| Managed 32 | 32 mature + 16 candidate | 69.67% | 75.49% | 82.23% | 79.08% | 109.8 KB | 5,099 |
| ALD-KRLS, 32 budget | 66 vectors | 62.97% | 61.39% | 69.62% | 74.09% | 109.5 KB | 7,026 |
| Fuzzy ARTMAP, 32 budget | 104 categories | 76.05% | 81.26% | **96.27%** | 79.31% | 109.2 KB | 5,048 |
| OS-ELM, 32 budget | 84 hidden units | 73.12% | **82.19%** | 86.72% | **82.13%** | 109.1 KB | 7,267 |
| RAN, 32 budget | 171 neurons | **78.51%** | 66.66% | 90.91% | 64.58% | 109.3 KB | 5,605 |
| Managed 64 | 64 mature + 16 candidate | 70.22% | 75.84% | 82.68% | 80.36% | 187.9 KB | 4,124 |
| ALD-KRLS, 64 budget | 91 vectors | 69.64% | 70.60% | 78.25% | 80.43% | 187.3 KB | 6,245 |
| Fuzzy ARTMAP, 64 budget | 179 categories | 76.06% | 81.26% | 96.28% | 79.29% | 187.8 KB | 5,020 |
| OS-ELM, 64 budget | 119 hidden units | 78.14% | **87.08%** | 90.41% | **86.27%** | 187.8 KB | 6,428 |
| RAN, 64 budget | 300 neurons | **83.74%** | 86.55% | **98.27%** | 75.54% | 187.8 KB | 4,645 |

OS-ELM is the strongest balanced baseline and changes the project's empirical
conclusion. Relative to the state-matched Managed 32 model, OS-ELM-32 gains
3.45 ± 1.37 points first-shift, 6.70 ± 1.46 pre-return, 4.48 ± 1.58 return-
online, and 3.06 ± 1.24 final mean. At the larger budget, its gains over
Managed 64 are 7.93 ± 1.55, 11.24 ± 2.09, 7.73 ± 1.96, and 5.91 ± 1.45 points.
These are paired means with nominal 95% Student-t half-widths and all exclude
zero.

The other methods expose useful tradeoffs. Fuzzy ARTMAP-32 gains 14.03 ± 1.99
points in return-online accuracy over Managed 32 while its final difference is
only +0.23 ± 3.68 points. RAN-64 is the most plastic method, gaining 13.53 ±
2.13 first-shift and 15.59 ± 2.46 return-online points over Managed 64, but it
finishes 4.81 ± 1.86 points lower. ALD-KRLS-64 is statistically unresolved
against Managed 64 on the four primary metrics and the smaller dictionary is
clearly inadequate.

Both KRLS dictionaries and both RANs fill their budgets; rejected admissions
continue to update existing outputs. ARTMAP creates 100 categories in the
inspected seed and therefore receives no benefit from its larger 179-category
allocation. All state ceilings and locked-evaluation checks pass. CPU
throughput is implementation-specific, but OS-ELM also processes more images
per second than the corresponding managed model in this NumPy run.

The defensible conclusion is no longer that CPAM is the best tested
forward-only memory. A simpler fixed random representation with cumulative RLS
is stronger on this small stationary-label transformation suite. CPAM remains
interesting as an interpretable frozen-key/probation mechanism, but a paper
claim now needs a setting where structural allocation provides something
OS-ELM cannot, such as class growth, heterogeneous local novelty, strict
capacity saturation, or a learned nonstationary representation.

## Limitations and next decision

- All tasks are synthetic and small.
- Hyperparameters have not undergone a formal search.
- The original BPTT comparison is controlled but not equivalently optimized.
  The final online-Adam comparison tunes learning rates on disjoint development
  seeds, but does not tune architectures or test specialized backprop-based
  continual-learning algorithms.
- Per-synapse eligibility scales quadratically in a dense recurrent layer.
- There is no adversarial update protection or production safety layer.
- Retention still declines when context zero returns.
- Some earlier image results are only three seeds on a small bundled dataset.
  The leverage, responsibility, capacity, value, frontend, predictive,
  polarity, and drift-suite experiments use 10 seeds, but their intervals are
  not corrected for multiple comparisons, and hyperparameters were not
  formally tuned.
- The augmentation doubles exposure; it has not yet been compared with simply
  repeating the original stream for the same number of updates.
- Masking removes the target latent coordinates, but neighboring fixed
  convolution coordinates have overlapping pixel receptive fields; this probe
  does not establish pixel-level target isolation.
- Predictive target loss falls substantially, but its continuously changing
  output basis is incompatible with the current cumulative downstream
  statistics. A successful predictive objective is not by itself a stable
  continual representation.
- Stable-backbone predictive surprise avoids coordinate drift, but a single
  normalized image-level prediction error changes candidate allocation without
  improving ordinary accuracy and slightly worsens aggregate drift retention.
- The predictive probe retains a fixed target encoder. A learned target
  encoder is deliberately deferred until downstream coordinate stability is
  addressed.
- The original/inverted/original benchmark preserves labels under contrast
  reversal. Absolute convolution is intentionally tailored to that symmetry;
  its success must not be generalized to transformations that change semantic
  information.
- The drift suite broadens label-preserving appearance changes but still does
  not test changing label semantics, new classes, adversarial corruptions, or
  transformations whose labels require human review. Occlusion and translation
  can make some 8x8 digits intrinsically ambiguous.
- RLS's strong retention uses quadratic state, so it is not yet a scalable
  answer to continual learning.
- The closest-analytic comparison finds that state-matched OS-ELM dominates
  CPAM on all four primary recurring-regime metrics. This substantially weakens
  a general performance claim for CPAM on the current benchmark.
- ALD-KRLS stores observed feature vectors, RAN stores recruited centers, and
  Fuzzy ARTMAP stores learned category prototypes. They retain no raw images,
  but their representation-state assumptions differ from fixed random OS-ELM
  and should not be described as identical memory semantics.
- Blank-image scaling validates systems behavior, not accuracy or numerical
  behavior on a diverse 60,000-image stream.
- CPU throughput does not establish energy or accelerator efficiency.
- The first factor-free router uses a fixed-resolution cumulative context table;
  it is factor-free but consumes about ten times the state of exact RLS here.
- Fast-memory routing improves immediate ordered adaptation but still damages
  final retention, and does not beat cumulative RLS on the drift benchmark.
- The maturity model preallocates 32 neuron slots. The fixed-key control uses a
  hand-set RBF width; the capstone shows that 32 slots fill in nine of ten seeds,
  while increasing capacity improves quality at quadratic-RLS state cost. The
  adaptive variant still requires safety bounds and computes entropy from
  uncalibrated scores.
- The threshold-free leverage gate delays poor recruitment but still fills all
  32 slots, so it does not address long-run capacity allocation.
- Probation improves online accuracy without improving final accuracy. Its
  candidate bank adds 12.6% state and can accumulate stranded proposals.
- Dense local responsibility remains best. Top-k adds selection overhead without
  reducing exact-RLS work, while normalization causes severe ordered forgetting.
- Evidence-managed 16-slot probation reduces candidate state and rejection with
  matched quality; an 8-slot bank prevents enough shuffled keys from maturing.
- Fast values increase immediate ordered plasticity but significantly worsen
  retention; evidence-doubling consolidation is algebraically lossless yet
  further destabilizes subsequent learning.
- Ordinary fixed convolution improves conventional accuracy and CPU time but
  fails under inversion. Signed-magnitude is now the primary spatial baseline;
  recurrence remains a sequential control and absolute convolution a
  transformation-specific invariance diagnostic.
- Adaptive keys introduce nonlinear basis drift, add 24% state and about 6%
  NumPy CPU time, and have only a small three-seed adaptation gain.
- Past observations have no counterfactual activations for neurons recruited
  later; narrow locality currently protects those historical regions without
  replay, but does not provide a formal non-interference guarantee.

The recurring-regime and matched online-Adam capstones are the finish line for
this first bounded-memory project. They establish a positive
stability/plasticity result, justify the forward-only choice under the tested
no-replay contract, and measure its state and throughput cost. Larger datasets,
mature-capacity expansion, specialized backprop continual-learning methods,
self-supervised context discovery, recommendation, and reinforcement learning
are separate follow-up scopes.
