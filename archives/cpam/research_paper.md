# Research Paper Working Outline

## Working identity

**Title:** Beyond Forgetting Factors: Cumulative Probationary Associative
Memory for Forward-Only Continual Learning

**Method:** Cumulative Probationary Associative Memory (CPAM)

**Paper type:** Venue-neutral full research article. Adapt to a specific
conference or journal only after the evidence is complete and a venue is
selected.

**Central question:** Can a bounded, replay-free learner improve the
stability-plasticity tradeoff of cumulative recursive least squares by adding
confirmed, immutable local representations, without deliberately discounting
past observations or using backpropagation during the stream?

**Central answer supported today:** On a ten-seed, six-phase recurring 8x8 digit
stream, CPAM improves adaptation, pre-return retention, return learning, and
final mean accuracy over cumulative RLS. At matched first-shift adaptation it
also improves retention and final performance over discounted RLS. It retains
substantially more than ordinary online Adam but uses more state than plain RLS
and adapts less quickly than an Adam MLP.

**Evidence still required:** standard 28x28 and natural-image datasets,
class-incremental tests, close analytic baselines, specialized continual
backpropagation, memory-matched replay, formal tuning, corrected statistics,
and accelerator/system measurements.

## Claim discipline

Use these formulations:

- “forward-only online update” or “backpropagation-free during the continual
  stream,” not “training-free”;
- “does not deliberately discount observations,” not “never forgets”;
- “stores cumulative sufficient statistics and bounded learned centers, not raw
  samples,” not “stores all information”;
- “beats ordinary matched online Adam under the tested no-replay contract,” not
  “backpropagation is unsuitable for continual learning”;
- “moves above the measured RLS forgetting-factor frontier on the current
  benchmark,” not “solves the stability-plasticity dilemma”;
- “bounded at configured capacity,” not “unbounded lifelong memory”;
- “CPU implementation result,” not a general speed, energy, or GPU claim.

The final abstract and title must be revised if the expanded experiments do not
support the cross-dataset version of the central claim.

## Abstract: near-final skeleton

Continual learners must incorporate new observations while preserving behavior
learned from earlier, potentially recurring regimes. Conventional online
gradient updates can adapt rapidly but may interfere with earlier knowledge,
whereas recursive least squares (RLS) with unit weighting preserves cumulative
evidence but can respond slowly to distribution change. Forgetting factors
increase responsiveness by explicitly discounting older observations, creating
a fixed stability-plasticity tradeoff.

We introduce Cumulative Probationary Associative Memory (CPAM), a replay-free,
forward-only online classifier that augments a cumulative RLS readout with
bounded local features. Prediction errors in familiar regions propose dormant
features using RLS leverage. A feature affects predictions only after a later,
nearby observation with the same label confirms it; its center is then frozen
permanently. A bounded candidate bank reclaims resolved proposals while never
replacing mature features. Every labeled observation updates the shared RLS
statistics with unit weight, and no raw example, forgetting factor, backward
pass, or optimizer is used during the stream.

On the current six-phase recurring-domain handwritten-digit benchmark, CPAM-32
improves over cumulative RLS by **1.10 points** in first-shift online accuracy,
**2.20 points** in pre-return retention, **2.46 points** during return learning,
and **1.56 points** in final mean accuracy across ten paired seeds. CPAM-64
matches the first-shift accuracy of RLS with factor 0.9999 (70.22% versus
70.21%) while improving pre-return retention by **5.31 points** and final mean
accuracy by **3.10 points**. A state-near online Adam MLP adapts faster but ends
**13.14 points** below CPAM-32 in final mean accuracy and **30.71 points** below
it immediately before domain return. CPAM-32 uses 2.81 times the persistent
state of cumulative RLS and is slower than RLS, exposing a quality-capacity
tradeoff rather than a free improvement.

**Awaiting final-work replacement paragraph:** Report results across MNIST,
Fashion-MNIST, CIFAR-10, domain- and class-incremental protocols, close analytic
continual learners, specialized gradient methods, and memory-matched replay.
State whether CPAM consistently remains above the RLS frontier and identify the
settings in which it does not.

## 1. Introduction

### 1.1 Motivation

Begin with deployed learning systems rather than a human-brain analogy. Many
systems receive a nonstationary stream after deployment: recommendations track
changing preferences, sensors encounter changing environments, and embodied
agents revisit earlier conditions. A static model requires periodic retraining;
an online model can update continuously, but the current observation may
interfere with earlier behavior.

Define the desired operating contract:

```text
observe x_t -> predict y_hat_t -> receive y_t -> update once -> continue
```

There are no epochs, required task labels, replayed raw observations, or pauses
for batch retraining. Evaluation predicts without updating.

### 1.2 The specific stability-plasticity problem

Explain why neither original baseline is sufficient:

- LMS is cheap and reactive but a constant step size continually overwrites old
  evidence.
- Cumulative RLS is sample efficient and retains all observations in its
  sufficient statistics, but old evidence increasingly dominates, reducing
  responsiveness.
- An RLS forgetting factor restores plasticity by geometrically reducing the
  influence of earlier samples. The correct factor depends on the unknown rate
  of change, and a smaller factor deliberately sacrifices retention.

The project therefore asks whether *representation growth* can respond to
persistent error while the shared estimator remains cumulative.

### 1.3 Proposed idea

Introduce CPAM in one paragraph. It combines a stable fixed representation, a
single cumulative RLS prediction path, and a bounded bank of local RBF features.
Errors do not immediately create active features. Low normalized leverage marks
a repeatedly observed/familiar region; an error there proposes a dormant
candidate. A second nearby observation with the same target confirms the region,
averages the two feature vectors into a center, and permanently freezes it. The
new coordinate then participates in the same cumulative readout.

Emphasize that there is no fast/slow router and no separate “winner” prediction.
The base features and all mature local activations form one feature vector and
one readout.

### 1.4 Contributions

Provisional contribution list:

1. A bounded probationary associative feature mechanism that uses cumulative
   RLS leverage to delay structural recruitment until an error persists in a
   familiar region.
2. A confirmation-and-freezing rule that prevents provisional or moving centers
   from contaminating the cumulative prediction basis.
3. An evidence-managed candidate bank that reuses provisional capacity without
   replacing mature features or discounting observations.
4. A recurring-domain evaluation that separates initial adaptation, retention
   before return, relearning/reactivation during return, final joint accuracy,
   and resource costs.
5. Empirical evidence that CPAM lies above the observed RLS forgetting-factor
   frontier, paired with the negative result that state-matched OS-ELM is
   stronger on every primary metric in the current benchmark.
6. **Awaiting:** multi-dataset and class-incremental evidence, specialized
   continual-learning baselines, and memory-matched replay.

Do not claim that each component is individually unprecedented. Resource
allocation, frozen features, analytic online learning, and stability-plasticity
mechanisms all have substantial prior literature. The proposed novelty is the
specific cumulative/probationary design and its measured frontier behavior.

## 2. Background and related work

This section should be a synthesis, not a chronological list. The detailed
source-by-source material and novelty matrix are maintained in
[literature_review.md](literature_review.md).

### 2.1 Continual learning and catastrophic interference

Cover the stability-plasticity dilemma, catastrophic forgetting, evaluation
settings, and the difference among task-, domain-, and class-incremental
learning. Distinguish online/prequential learning from phase-wise training.

Position major strategy families:

- rehearsal and experience replay;
- regularization and parameter-importance methods such as EWC and Synaptic
  Intelligence;
- distillation methods such as Learning without Forgetting;
- parameter isolation and expandable architectures;
- prototype, kernel, and analytic methods.

CPAM belongs primarily to the last two groups: it expands a bounded stable
feature basis and updates an analytic readout.

Starting sources include Kirkpatrick et al., “Overcoming catastrophic forgetting
in neural networks”; Lopez-Paz and Ranzato, “Gradient Episodic Memory”; and the
standard primary papers for every implemented baseline. Add recent online
continual-learning evaluation papers through the final literature cutoff.

### 2.2 Adaptive filters and analytic continual learning

Introduce LMS and RLS as sequential estimators. Explain that factor-1 RLS
recursively maintains the ridge solution for a fixed feature basis using
sufficient statistics, whereas factor less than one exponentially discounts
history.

Discuss online sequential extreme learning machines and kernel adaptive
filters. Closely compare CPAM with:

- [ACIL: Analytic Class-Incremental Learning with Absolute Memorization and
  Privacy Protection](https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html);
- [GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class
  Incremental Tasks](https://openaccess.thecvf.com/content/CVPR2023/html/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.html);
- [RanPAC: Random Projections and Pre-trained Models for Continual
  Learning](https://papers.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html).

Critical distinction to verify: CPAM changes a bounded feature basis online via
probationary local recruitment while keeping confirmed centers immutable and
using cumulative unit-weight output statistics. Determine whether an earlier
analytic/kernel method already contains this exact combination.

### 2.3 Constructive and resource-allocating networks

Discuss Platt's resource-allocating network, growing RBF networks, adaptive
resonance theory, growing neural gas, and constructive networks. Explicitly cite
[Cascade-Correlation](https://papers.nips.cc/paper_files/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html),
which adds hidden units and freezes their incoming weights.

Comparison dimensions:

| Question | CPAM answer |
|---|---|
| What triggers a proposal? | A classification error, distinctness, and below-mean normalized RLS leverage |
| Does one unusual point become permanent? | No; it enters a dormant candidate bank |
| What confirms it? | A later nearby observation with the same label |
| Do dormant candidates affect prediction? | No |
| Do mature centers move or get replaced? | No |
| How is the output trained? | Unit-weight cumulative RLS |
| Are raw observations replayed? | No |
| Is structural capacity bounded? | Yes, separately for mature and candidate centers |

### 2.4 Backpropagation-free and local learning

Briefly survey feedback alignment, direct feedback alignment, target propagation,
local losses, equilibrium methods, and Forward-Forward. Avoid implying CPAM
solves the same problem. CPAM does not train an arbitrary deep representation
without gradients; it uses fixed features plus an analytic expanding readout.

The reason to evaluate it is operational and empirical: one-pass updates, no
autograd graph or optimizer during the stream, no raw replay, and potentially
different retention behavior. Whether those properties improve time or memory
depends on feature width and the quadratic RLS state.

### 2.5 Stable and predictive representations

Explain the tension revealed by this project: cumulative statistics assume a
stable coordinate system. The JEPA-inspired masked predictor learned to predict
held-out latent regions, but its moving representation invalidated downstream
historical statistics and harmed final retention.

Reference [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)
only as inspiration and clearly state that the repository's experiment is not
an implementation of I-JEPA. The negative experiment belongs in the supplement
and motivates future work on frozen, append-only, or explicitly consolidated
representations.

## 3. Problem formulation

### 3.1 Stream and information assumptions

At time `t`, the learner receives features `x_t`, emits scores and a prediction,
then receives a one-hot label `y_t` and updates exactly once. Define the stream
as non-IID and potentially recurring. Domain identity and change points are not
provided to the learner. Task identity is unavailable at inference.

State explicitly:

- supervised feedback is currently immediate;
- there is no raw-example replay;
- learner state is bounded at fixed configured capacity;
- all evaluation passes are locked;
- the encoder is fixed during the online memory experiment;
- pretrained-feature experiments, if used, are not end-to-end backprop-free.

### 3.2 Metrics

Define prequential accuracy as correctness measured before updating on the
current event. Define phase-end locked accuracy separately.

For recurring domain stream `D_0, D_1, ..., D_k, D_0, D_1`, define:

- **first-shift online:** mean prequential accuracy on first appearances of
  shifted domains;
- **pre-return retention:** accuracy on a domain immediately before it returns;
- **return online:** prequential accuracy while recurring domains are observed
  again;
- **final mean:** mean locked accuracy over all domains after the stream;
- **forgetting:** decline from a domain's best prior locked accuracy to its
  later locked accuracy.

For class-incremental experiments, additionally define average accuracy over
seen classes, backward transfer, average forgetting, and last-phase accuracy.

### 3.3 Resource accounting

Persistent state includes weights, inverse-correlation matrices, centers,
candidate metadata, optimizer moments, and replay buffers as applicable. Peak
training memory additionally includes saved activations and transient tensors.
Report prediction and update latency separately. State whether measurements are
CPU or GPU, dtype, thread count, batch size, and hardware.

## 4. Method

### 4.1 Stable input representation

The current grayscale encoder applies two fixed 3x3 spatial filters, pools the
response maps, and concatenates signed responses with their magnitudes to form
64 coordinates. A bias coordinate yields the 65-coordinate base feature vector
used by the readout. The signed component retains polarity information; the
magnitude component reduces sensitivity to polarity changes without making the
representation completely inversion invariant.

Include a compact encoder ablation in the main paper. Absolute convolution is a
useful invariance diagnostic but should not be the default because polarity can
carry semantic information in other tasks.

**Awaiting:** exact RGB fixed encoder and optional frozen pretrained encoder.

### 4.2 Cumulative RLS readout

Let `f_t` be the concatenation of base features and all mature local-neuron
activities. Inactive preallocated coordinates are zero. With output matrix
`W_t`, inverse correlation `P_t`, one-hot target `y_t`, and forgetting factor
`lambda`, prediction and update are:

```text
y_hat_t = W_(t-1) f_t
q_t     = P_(t-1) f_t
k_t     = q_t / (lambda + f_t^T q_t)
W_t     = W_(t-1) + (y_t - y_hat_t) k_t^T
P_t     = (P_(t-1) - k_t f_t^T P_(t-1)) / lambda
```

CPAM fixes `lambda = 1`. Initialize `P_0 = I / alpha`, where `alpha` is ridge
regularization, and initialize output weights to zero. The implementation
symmetrizes `P` after each update to limit floating-point asymmetry.

For a fixed feature basis, factor-1 RLS retains all observations in cumulative
sufficient statistics. Because CPAM exposes new coordinates over time, do not
claim exact equivalence to a retrospective batch kernel model: earlier samples
have no counterfactual activation in later-recruited coordinates.

### 4.3 Local mature features

For mature center `c_j` and fixed width `sigma`, define:

```text
a_j(x) = exp(-mean((x - c_j)^2) / (2 sigma^2))
```

The complete prediction uses one path:

```text
f(x) = [x, a_1(x), ..., a_M(x)]
y_hat = W f(x)
```

There are no separate fast and slow predictions and no router deciding which
network wins. All mature local features contribute to the same cumulative
readout. Existing dense responsibility is retained because top-k activation did
not improve quality or reduce exact-RLS computation.

### 4.4 Leverage-gated proposal

Before updating on an event, compute RLS leverage:

```text
h_t       = f_t^T P_(t-1) f_t
h_t_norm  = h_t / (1 + h_t)
```

High leverage indicates an unfamiliar feature direction. A misclassified event
may propose a local center only when its normalized leverage is below the
cumulative pre-update mean leverage, the feature is sufficiently distant from
existing mature centers, and mature capacity remains. This has no learned
threshold, forgetting factor, or age window. It intentionally waits for the
shared model to consider a region familiar before structural recruitment.

### 4.5 Probation and immutable maturity

A qualifying event does not immediately alter prediction. It creates a dormant
candidate containing a center summary, count, label, and proposal leverage. A
later observation confirms the candidate when it has the same label and lies
within the fixed distance threshold. The candidate centroid is updated with
that observation, promoted to the next mature slot, and frozen permanently.
The confirming observation immediately trains the newly exposed coordinate.

The candidate therefore contains a bounded learned summary of observations,
not a replayable raw-sample history. Dormant candidates never enter prediction
or RLS statistics, avoiding a moving feature coordinate.

### 4.6 Bounded candidate management

When a candidate slot is required:

1. Use an empty slot if one exists.
2. Otherwise, evaluate dormant candidates with the current model. Reclaim the
   correctly classified candidate with the largest output margin.
3. If none is resolved, identify the candidate with greatest stored normalized
   leverage. Replace it only when the new proposal has lower normalized
   leverage; otherwise reject the proposal.

This policy never replaces a mature center. Reclaiming a dormant structural
proposal does not remove its observations from cumulative RLS statistics; it
only reuses provisional feature capacity.

The counterintuitive preference for lower leverage should be explained: a
candidate in a familiar but persistently erroneous region is more justified as
a discriminative local feature than a one-off point in an unfamiliar region.

### 4.7 Complexity and limitations

Let `d` be base feature width, `M` mature capacity, `C` candidate capacity, and
`K` output classes.

- Exact RLS state and update cost scale quadratically with `d + M` because of
  `P`; output weights scale as `K(d + M)`.
- Mature-center storage scales as `Md`.
- Candidate storage and scanning scale as `Cd` and `C`, respectively.
- State is independent of stream length at fixed capacities, but not necessarily
  small.
- Increasing mature capacity improves quality in current results and increases
  both state and update cost.
- The method has no formal guarantee of non-interference for features recruited
  after earlier samples were processed.

## 5. Experimental design

### 5.1 Current dataset and split

Describe the bundled scikit-learn digits data, train/test construction,
`test_per_class = 40`, deterministic seeds, one update per training image, and
the fact that no dataset download is required. Clarify that current images are
8x8.

### 5.2 Current recurring-domain capstone

Use ten paired seeds: 3, 7, 11, 17, 23, 29, 37, 41, 47, and 53. Train in six
phases:

```text
original -> inversion -> one-pixel translation -> 2x2 center occlusion
         -> original return -> inversion return
```

After every phase, evaluate original, inversion, translation, and occlusion
test sets without learning. Compare RLS factors 1, 0.99999, 0.9999, 0.999,
0.995, 0.99, 0.98, and 0.95; immediate maturity; and CPAM mature capacities 8,
16, 32, and 64 with 16 candidate slots.

### 5.3 Matched online-backprop experiment

Hold encoder, stream, splits, event order, batch size one, one update per image,
no replay, and float64 fixed. Compare linear softmax and tanh MLP-64 models using
Adam. Select learning rates independently on development seeds 2, 5, and 13
from 0.0001, 0.0003, 0.001, 0.003, and 0.01. The selected rates are 0.01 for the
linear model and 0.003 for the MLP. Do not use test seeds for tuning.

### 5.4 Expanded submission experiments

**Awaiting from `future_work.md`:**

- MNIST and Fashion-MNIST recurring-domain results;
- MNIST and Fashion-MNIST 5x2 class-incremental results;
- CIFAR-10 domain- and class-incremental results;
- controlled fixed-feature and frozen-pretrained-feature tracks;
- OS-ELM, ACIL/RanPAC-style, and kernel analytic comparisons;
- EWC, Synaptic Intelligence, distillation, Experience Replay, and DER++;
- matched-memory replay budgets;
- long-stream capacity saturation and numerical stability;
- CPU/GPU time and peak-memory measurements.

### 5.5 Statistical analysis

Current intervals are nominal paired Student-t intervals and are not corrected
for multiple comparisons. The final paper must preregister primary metrics,
separate development and test seeds, report paired effect sizes, correct
secondary families, and use a cross-dataset repeated-measures or hierarchical
analysis for the overall claim.

## 6. Results: completed progression

This section should tell a causal experimental story, not reproduce every
repository milestone.

### 6.1 LMS and RLS establish the starting tradeoff

Summarize the early experiments: LMS provides inexpensive constant-memory
updates but poor retention; RLS is substantially more sample efficient and
stable at the cost of a dense inverse-correlation matrix. The factor sweep
demonstrates why selecting a global recency scale is unsatisfactory.

Current capstone factor curve:

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

Interpretation: moderate discounting improves immediate adaptation while
degrading retained and final performance. Aggressive discounting eventually
damages both. This curve, rather than factor 1 alone, is CPAM's primary analytic
baseline.

### 6.2 Why a single routed fast/slow memory was rejected

Briefly report that early factor-free routed memories increased reactivity but
lost stability and used far more state. Routing confidence did not reliably
identify which memory should control a prediction. This motivated a single
prediction path with representational recruitment rather than competing
experts.

Keep detailed routed-memory tables in the supplement.

### 6.3 From immediate recruitment to probation

Present the ablation progression:

- localized RBF features are necessary; broad features cause cross-class
  interference;
- entropy gating reduces recruitment but yields small/mixed quality changes;
- leverage is a more direct familiarity signal and modestly improves final
  performance;
- immediate recruitment can permanently encode an early mistake;
- probation requires repeated same-label local evidence before a key becomes
  active and immutable;
- managed candidate capacity prevents unresolved proposals from permanently
  occupying the candidate bank.

Key probation result versus immediate leverage recruitment:

| Model | Shuffled online | Shuffled final | Ordered online | Ordered final | State |
|---|---:|---:|---:|---:|---:|
| Immediate leverage key | 85.29% | 90.95% | 77.24% | **90.78%** | 135.4 KB |
| Leverage + probation | **85.74%** | **90.98%** | **77.72%** | 90.58% | 152.4 KB |

Probation primarily improves online sample efficiency; it does not by itself
establish a final-retention gain. The capstone later shows a clearer effect when
recurring regimes and capacity are evaluated together.

### 6.4 Candidate management bounds provisional state

Report that a managed 16-slot candidate bank matched the 32-slot unmanaged
bank while reducing candidate state and rejections. Managed-8 was too small.
Candidate management did not create a large direct accuracy gain; its role is
to make probationary state bounded and usable.

### 6.5 Stable signed+magnitude features remove a representation confound

The fixed convolutional frontend improved ordinary accuracy and CPU throughput
over recurrent row ingestion but initially failed badly under contrast
inversion. Absolute responses made inversion nearly invariant and performed
best on that label-preserving benchmark, but this invariance can be invalid
when polarity is meaningful. Signed+magnitude features preserve sign while
adding magnitude information and became the main stable baseline.

Across the six-shift drift suite, signed+magnitude finished with 89.40% mean
transformed accuracy and 93.68% mean original accuracy, versus 85.13% and
90.39% for recurrence. It improved mean joint accuracy by 3.78 +/- 0.65 points.

This belongs in the main paper because CPAM's memory results depend on the
stable representation. Keep the full frontend and polarity tables in the
supplement.

### 6.6 CPAM moves above the measured RLS frontier

Main current table:

| Model | First-shift online | Pre-return | Return online | Final mean | State | Images/s |
|---|---:|---:|---:|---:|---:|---:|
| RLS factor 1 | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | 9,893* |
| Immediate maturity 32 | 69.09% | 73.94% | 80.90% | 77.93% | 100.9 KB | 6,297* |
| CPAM-8 | 68.78% | 73.68% | 80.52% | 77.81% | 62.0 KB | 6,653* |
| CPAM-16 | 69.06% | 74.21% | 81.21% | 78.36% | 76.9 KB | 6,115* |
| CPAM-32 | 69.67% | 75.49% | 82.23% | 79.08% | 109.8 KB | 5,335* |
| CPAM-64 | **70.22%** | **75.84%** | **82.68%** | **80.36%** | 187.9 KB | 4,403* |

`*` These throughput values come from the capstone-only run. When comparing
against Adam in the same table, use the matched-comparison measurements instead.

CPAM-32 versus cumulative RLS:

- first-shift: +1.10 +/- 0.48 points;
- pre-return: +2.20 +/- 1.67 points;
- return online: +2.46 +/- 1.19 points;
- final mean: +1.56 +/- 0.50 points.

CPAM-32 versus immediate maturity:

- first-shift: +0.58 +/- 0.36 points;
- pre-return: +1.55 +/- 1.38 points;
- return online: +1.34 +/- 0.95 points;
- final mean: +1.14 +/- 0.36 points.

CPAM-64 versus factor-0.9999 RLS has essentially equal first-shift performance
(+0.005 +/- 0.87 points), with +5.31 +/- 2.18 pre-return, +2.97 +/- 1.52
return-online, and +3.10 +/- 1.26 final mean. This is the strongest current
matched-adaptability result.

Capacity is not free. CPAM-32 uses 2.81 times cumulative-RLS state and is 46%
slower in the capstone-only NumPy CPU measurement; CPAM-64 uses 4.80 times the
state and is 55% slower. Capacities 8 and 16 fill in every seed, 32 fills in
nine of ten, and 64 fills in five of ten. During returns, CPAM-32 reactivates an
average of 14.8 existing neurons for original images and 11.0 for inversion,
while all mature centers remain bitwise fixed.

### 6.7 Online Adam adapts faster but retains less

Use the matched-comparison throughput values here:

| Model | First-shift online | Pre-return | Return online | Final mean | Persistent state | Images/s |
|---|---:|---:|---:|---:|---:|---:|
| RLS factor 1 | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | **12,203** |
| CPAM-32 | 69.67% | **75.49%** | 82.23% | **79.08%** | 109.8 KB | 6,795 |
| CPAM-64 | 70.22% | 75.84% | 82.68% | 80.36% | 187.9 KB | 5,432 |
| Online Adam linear | 73.91% | 38.48% | 81.70% | 41.60% | **15.6 KB** | 4,752 |
| Online Adam MLP-64 | **75.50%** | 44.78% | **88.97%** | 65.94% | 115.5 KB | 3,501 |

Relative to CPAM-32, the MLP gains 5.83 +/- 2.10 points on first-shift and
6.73 +/- 2.03 on return-online accuracy, but loses 30.71 +/- 2.32 points on
pre-return retention and 13.14 +/- 5.31 on final mean. Its persistent state is
5.2% larger, and CPAM-32 processes 1.94 times as many images per second in this
batch-one, float64, single-CPU-thread implementation.

Interpretation: gradients are fully compatible with continuous learning and
the MLP is more plastic. The ordinary no-replay optimizer suffers much stronger
interference. This does not predict the outcome against replay, EWC, SI,
distillation, alternative architectures, batching, or GPU execution.

### 6.8 Awaiting expanded results

Reserve subsections for:

1. Cross-dataset recurring-domain frontier.
2. Class-incremental retention and backward transfer.
3. Analytic continual-learning competitors.
4. Specialized gradient continual-learning and memory-matched replay.
5. Controlled versus pretrained representation tracks.
6. Long-stream saturation and numerical stability.
7. Accuracy-state-latency Pareto analysis on CPU and GPU.

For every pending subsection, report failures and dataset heterogeneity. Do not
replace dataset-level outcomes with a favorable grand average.

## 7. Ablations and negative results

### Main-paper ablations

Keep these in the main paper because they establish causal support:

- full RLS factor curve;
- cumulative RLS versus immediate maturity;
- leverage gate;
- probation versus immediate maturity;
- unmanaged versus managed candidate capacity;
- mature-capacity curve;
- compact signed, absolute, and signed+magnitude encoder comparison.

### Supplementary experiments

Report these as useful negative or secondary results:

- Entropy gating reduced structural recruitment, but accuracy changes were
  small and mixed.
- Sparse top-k responsibility did not improve quality and did not reduce dense
  exact-RLS computation. Normalizing weak activations caused severe forgetting.
- Fast per-key values increased plasticity but damaged retention; algebraically
  lossless evidence-doubling transfers still destabilized later learning.
- Adaptive key movement created basis drift, added state and time, and produced
  only a small preliminary adaptation gain.
- The forward-only masked predictive representation reduced target MSE but its
  moving coordinates substantially harmed final retention.
- Stable-backbone scalar predictive surprise changed candidate allocation but
  did not improve ordinary results and slightly worsened drift performance.
- Absolute convolution performed exceptionally under polarity inversion because
  the transformation preserves the digit label; it is not appropriate when
  contrast polarity is semantic.

These outcomes explain why the final mechanism is simpler than the full set of
ideas explored. They also reduce publication bias and prevent future work from
repeating known failures.

## 8. Discussion

### 8.1 What CPAM appears to achieve

CPAM changes adaptation structurally rather than globally discounting the past.
Confirmed local features can activate when a recurring region returns, while
their immutable centers preserve a stable coordinate system. Unit-weight RLS
continues to aggregate every labeled event. The result is a modest but
consistent improvement over cumulative RLS and a better retention/adaptation
combination than the tested factor curve.

### 8.2 Why probation matters

Early errors are common while a model is learning. Immediate permanent
recruitment can turn an incidental startup error into lasting structure.
Probation treats one error as a proposal and repeated local same-label evidence
as confirmation. This is not equivalent to keeping a replay buffer: the
candidate is a bounded structural summary, does not participate in prediction,
and is discarded after promotion or reclamation.

### 8.3 Why “factor free” does not mean “memory free”

CPAM removes exponential age discounting, not resource costs. It pays for
retention with a dense inverse-correlation matrix, frozen centers, and bounded
candidate metadata. The capacity sweep shows a quality-state tradeoff. The
scientific question is whether this tradeoff is preferable to recency
discounting, replay, optimizer state, or parameter protection in a given
deployment—not whether memory is free.

### 8.4 Why backpropagation-free remains relevant but secondary

The primary contribution is continuous memory behavior. Backpropagation-free
updates are relevant because the stream requires an update after every event
without an autograd graph, backward pass, or replay buffer. However, exact RLS
has quadratic state and computation, so absence of gradients does not guarantee
lower cost. The paper must make resource claims from measurements, not from the
learning-rule label.

### 8.5 Representation stability

The predictive experiments reveal a core constraint: a downstream cumulative
estimator interprets its statistics in a coordinate system. If an upstream
representation moves, historical sufficient statistics become stale. CPAM's
frozen mature centers are one response to that issue. Future learned
representations will need frozen targets, append-only coordinates, explicit
transport/consolidation of statistics, or replay-like correction.

### 8.6 Practical operating region

Discuss where CPAM may be attractive if expanded results support it:

- streams where old regimes recur;
- raw-example retention is undesirable;
- updates must occur one event at a time;
- feature width and memory capacity remain moderate;
- a stable fixed or pretrained representation is available.

It is less attractive when immediate adaptation dominates retention, replay is
cheap and allowed, feature width makes exact RLS infeasible, or end-to-end
representation change is essential.

## 9. Limitations

Current limitations that must remain until evidence resolves them:

- only one small dataset currently supports the main claim;
- shifts are synthetic and label preserving;
- there is no current class-incremental result;
- no specialized continual-backprop or replay method has been tested;
- close analytic/constructive prior art has not yet been exhaustively audited;
- hyperparameters were not selected under a complete formal search;
- current intervals are nominal and uncorrected;
- exact RLS scales quadratically with expanded feature width;
- the candidate and mature capacities impose a finite structural lifetime;
- earlier observations lack activations for features recruited later;
- fixed-width RBF locality and distance thresholds are hand configured;
- CPU NumPy/PyTorch throughput does not establish GPU speed or energy use;
- no adversarial update defense or production safety layer exists;
- supervised labels arrive immediately; recommendation and reinforcement
  feedback are not evaluated;
- the successful encoder encodes domain-specific invariances and is not learned
  end to end.

Update this list rather than deleting it wholesale when expanded experiments
arrive. A limitation is resolved only by direct evidence.

## 10. Conclusion: draft

This work studies continual adaptation without replay, epochs, gradient
backpropagation, or deliberate temporal discounting. Cumulative RLS preserves
unit-weight evidence but adapts slowly; a forgetting factor improves plasticity
by intentionally reducing the influence of the past. CPAM instead allocates
bounded local structure only after a persistent error is observed in a familiar
region. Dormant candidates require confirmation, mature centers remain fixed,
and every event continues to update a single cumulative readout.

On the current recurring-domain digit study, this mechanism improves adaptation
and retention over cumulative RLS and lies above the measured forgetting-factor
frontier. It also retains substantially more than ordinary matched online Adam,
although the gradient model adapts faster and CPAM pays significant state and
compute costs. Crucially, a state-matched OS-ELM improves all four primary
metrics at both tested memory ceilings. Fuzzy ARTMAP and RAN also reveal that
much faster online adaptation is available with different retention tradeoffs.
The current evidence therefore does not support CPAM as the strongest general
forward-only learner; it supports the narrower mechanism claim and identifies
the conditions its next evaluation must isolate.

**Awaiting final conclusion:** Insert the multi-dataset outcome, specialized
continual-learning and replay comparisons, resource frontier, and explicit
failure cases. A positive paper now requires a setting where structural
recruitment has a measured benefit that fixed random OS-ELM does not provide.

## 11. Planned main-paper display items

1. Figure: CPAM architecture and candidate lifecycle.
2. Figure: six-phase recurring stream and evaluation protocol.
3. Figure: RLS factor adaptation-retention frontier with CPAM capacities.
4. Figure: prequential learning curves around first shifts and returns.
5. Figure: cross-dataset paired effects with confidence intervals.
6. Figure: accuracy versus state and update latency.
7. Table: dataset, stream, information, and representation assumptions.
8. Table: primary quality and resource comparison.
9. Table: mechanism ablations.
10. Supplement: full per-dataset/per-seed results and negative experiments.

## 12. Reproducibility and artifact checklist

- Link every reported number to a result artifact and analysis command.
- Publish train/test construction, event order, transformations, and class order.
- Publish development/test seeds and complete tuning grids.
- Record git commit, environment, dtype, hardware, thread count, and device.
- Include state-accounting definitions and memory-budget calculations.
- Test predict-before-update ordering and locked evaluation.
- Test bitwise immutability of mature centers.
- Test bounded state across stream lengths.
- Publish checkpoints or deterministic regeneration commands.
- Include data licenses and explicit download instructions.
- State which experiments use pretrained or backpropagation-trained components.
- Include a research ethics, broader impacts, and compute statement appropriate
  to the eventual venue.

## 13. Authorship and submission placeholders

- Authors: `[AWAITING AUTHOR LIST]`
- Affiliations: `[AWAITING AFFILIATIONS]`
- Corresponding author: `[AWAITING CONTACT]`
- Funding: `[AWAITING FUNDING STATEMENT]`
- Conflicts of interest: `[AWAITING DISCLOSURE]`
- Code/data availability URL: `[AWAITING ARCHIVAL REPOSITORY]`
- Target venue: `[AWAITING RESULTS AND VENUE SELECTION]`
- Literature cutoff date: `[SET IMMEDIATELY BEFORE SUBMISSION]`
