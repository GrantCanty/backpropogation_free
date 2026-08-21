# CPAM: Future Work Toward a Journal-Level Paper

Last updated: 2026-08-21

This document is the execution roadmap for turning the current bounded-memory
continual-learning study into a defensible journal or major-conference paper.
It is intentionally stricter than the repository's original milestone plan.
The existing work is a positive proof of concept; it is not yet evidence of
generality or state of the art.

The working method name is **Cumulative Probationary Associative Memory
(CPAM)**. The working paper title is:

> Beyond Forgetting Factors: Cumulative Probationary Associative Memory for
> Forward-Only Continual Learning

## 1. Current evidence and claim boundary

The completed experiment uses the bundled scikit-learn 8x8 handwritten-digit
dataset, a fixed 64-coordinate signed+magnitude convolutional representation,
one label update per image, no raw replay, and locked evaluation. Its recurring
six-phase stream is:

```text
original -> inversion -> translation -> center occlusion
         -> original return -> inversion return
```

Across ten paired test seeds, the most important current results are:

| Model | First-shift online | Pre-return | Return online | Final mean | Persistent state | Images/s |
|---|---:|---:|---:|---:|---:|---:|
| RLS, factor 1 | 68.57% | 73.29% | 79.77% | 77.51% | 39.1 KB | 12,203 |
| CPAM-32 | 69.67% | 75.49% | 82.23% | 79.08% | 109.8 KB | 6,795 |
| CPAM-64 | 70.22% | 75.84% | 82.68% | 80.36% | 187.9 KB | 5,432 |
| Online Adam linear | 73.91% | 38.48% | 81.70% | 41.60% | 15.6 KB | 4,752 |
| Online Adam MLP-64 | 75.50% | 44.78% | 88.97% | 65.94% | 115.5 KB | 3,501 |

CPAM-32 improves over cumulative RLS by 1.10 +/- 0.48 points on first-shift
online accuracy, 2.20 +/- 1.67 on pre-return retention, 2.46 +/- 1.19 on return
online accuracy, and 1.56 +/- 0.50 on final mean accuracy. These are paired
means with nominal 95% Student-t half-widths; they have not been corrected for
multiple comparisons.

CPAM-64 nearly matches the first-shift adaptability of RLS factor 0.9999
(70.22% versus 70.21%) while improving pre-return retention by 5.31 points,
return-online accuracy by 2.97 points, and final mean accuracy by 3.10 points.
This is the clearest evidence that CPAM may move above, rather than merely move
along, the observed RLS stability-plasticity frontier.

The current defensible conclusion is narrow:

> In this small, no-replay, recurring-domain experiment, bounded probationary
> associative features improve the adaptation-retention balance of cumulative
> RLS without discounting observations, and retain substantially more prior
> performance than ordinary matched online Adam.

Do not currently claim that CPAM is state of the art, that it never forgets,
that it is universally faster than backpropagation, that it is more energy
efficient, or that it scales to lifelong learning.

## 2. Submission standard

The target is a venue-neutral research package that can later be adapted to a
specific NeurIPS-style conference or a named Elsevier journal. Elsevier is a
publisher, not a single venue, so final length, template, and reference style
must be chosen only after selecting a journal.

Before submission, the study should satisfy all of these gates:

1. **Novelty:** a completed literature audit distinguishes CPAM from analytic
   continual learning, online sequential extreme learning machines, kernel
   RLS, resource-allocating networks, constructive networks, and established
   stability-plasticity methods.
2. **Generality:** actual accuracy experiments cover at least three datasets,
   including one natural-image dataset, and both domain- and class-incremental
   streams.
3. **Comparison quality:** CPAM is compared with close analytic methods,
   ordinary online backpropagation, specialized continual-backprop methods,
   and memory-matched replay.
4. **Statistical quality:** primary hypotheses and metrics are fixed before
   final test runs; tuning data and test data are disjoint; uncertainty,
   effect sizes, and multiplicity are handled explicitly.
5. **Systems honesty:** state, peak memory, update latency, throughput, and
   scaling are measured on CPU and GPU where applicable. Energy is omitted
   unless it is measured.
6. **Reproducibility:** one command regenerates each main table and figure from
   machine-readable per-seed artifacts; environments, hardware, seeds, and
   configurations are recorded.

## 3. Work packages

### WP0: Freeze and audit the existing evidence

- Preserve `results/memory-capstone.json` and
  `results/memory-backprop-comparison.json` as immutable source artifacts.
- Add a result manifest containing the command, configuration hash, git commit,
  runtime environment, hardware, and SHA-256 digest for every paper artifact.
- Create a paper-analysis script that reads JSON and produces tables, paired
  differences, confidence intervals, and plots. Do not manually copy computed
  numbers into figures.
- Re-run the current capstone once from a clean environment and confirm that all
  ten per-seed results reproduce within documented numerical tolerances.
- Record why the backprop throughput table differs from the capstone-only table:
  use the measurements from the matched comparison when models appear together.
- Preserve raw per-seed outcomes, not only aggregates.

**Acceptance:** the current headline table, all reported paired differences,
state sizes, and diagnostic counts can be regenerated from raw artifacts with
one documented command.

### WP1: Complete the literature and novelty audit

**Status:** Initial audit completed 2026-08-21 in
[literature_review.md](literature_review.md). The audit narrows the claim to
CPAM's proposal-confirmation-maturity lifecycle and adds budgeted KRLS,
RAN/MRAN, OS-ELM, and ARTMAP as priority baselines. Refresh the search and
perform equation-level checks of the closest papers before submission.

Create a standalone annotated literature report. For every close method, record
its learning setting, whether it stores samples, whether it discounts history,
how features are created, whether mature features move, update complexity,
state complexity, and the precise difference from CPAM.

At minimum, cover:

- catastrophic forgetting and the stability-plasticity dilemma;
- complementary learning systems and synaptic consolidation;
- LMS, RLS, recursive ridge regression, and kernel adaptive filtering;
- OS-ELM and dynamically growing extreme-learning machines;
- ACIL, GKEAL, RanPAC, and later analytic continual-learning methods;
- resource-allocating networks, growing RBF networks, adaptive resonance
  theory, growing neural gas, and Cascade-Correlation;
- replay, generative replay, EWC, Synaptic Intelligence, Learning without
  Forgetting, GEM/A-GEM, DER/DER++, and online continual-learning protocols;
- backpropagation alternatives such as feedback alignment, local learning, and
  Forward-Forward, while making clear that CPAM is an analytic output-learning
  method rather than a general replacement for gradient training;
- fixed and pretrained representations in continual learning;
- JEPA/I-JEPA only as representation-learning context and a future direction.

The novelty audit must actively search for counterexamples to the proposed
claim. Especially inspect work that combines RLS with dynamically added or
frozen local kernels.

**Acceptance:** the related-work matrix supports a precise contribution claim,
or the claim and method name are revised before final experiments.

### WP2: Build a common benchmark protocol

Use a dataset-agnostic event interface with strict
`predict -> record -> update` ordering. Evaluation passes must not update model
weights, optimizer state, running statistics, candidate banks, or transient
state. All methods see identical event orders and evaluation points.

Core datasets:

| Dataset | Role | Size/image shape | Required streams |
|---|---|---|---|
| sklearn digits | Fast ablation and continuity with existing work | 1,797, 8x8 grayscale | Existing recurring drift and class-incremental |
| MNIST | Actual 60k grayscale scaling | 60,000 train, 28x28 | Recurring domain and 5x2 class-incremental |
| Fashion-MNIST | Harder 60k grayscale scaling | 60,000 train, 28x28 | Recurring domain and 5x2 class-incremental |
| CIFAR-10 | Natural-image generalization | 50,000 train, 32x32 RGB | Corruption/domain recurrence and 5x2 class-incremental |

Fashion-MNIST images are 28x28, not 64x64. The existing blank-image scaling
test matches its stream length and dimensions but is not an accuracy result.

Optional strengthening dataset, chosen after the core suite works:

- CIFAR-100 for longer class-incremental sequences; or
- CORe50 for a naturally sequential visual stream.

Do not download datasets automatically during installation or unit tests.
Provide an explicit download/preparation command, resumable downloads, checksums,
and a documented cache location. Downloads can be staged when bandwidth is
available.

#### Domain-incremental protocol

- Keep the label space fixed and do not reveal domain identity to the learner.
- Include abrupt and gradual transitions plus recurrence.
- For grayscale data, use original, polarity inversion, translation, occlusion,
  noise/contrast, and original returns, but predeclare one primary stream before
  final testing.
- For CIFAR-10, use documented, label-preserving corruptions at fixed severity;
  inspect transformed examples and exclude changes that alter plausible labels.
- Evaluate all encountered domains after every phase without learning.

#### Class-incremental protocol

- Use five phases of two classes for ten-class datasets.
- Keep a single ten-class output space active from the beginning; do not provide
  task identity at prediction time.
- Train each observation once unless a separately labeled repeated-exposure
  experiment is being run.
- Evaluate all classes encountered so far after every phase.
- Use several seeded class orders and publish every order.
- Define how a newly observed class is incorporated before final testing. CPAM,
  RLS, and baselines must use the same output-space policy.

#### Representation tracks

Run two explicitly separated tracks:

1. **Fully forward-only controlled track:** deterministic fixed features shared
   by every method. Signed+magnitude convolution remains the grayscale baseline;
   define a deterministic fixed spatial encoder of matched width for RGB.
2. **Frozen pretrained-feature track:** a standard frozen encoder may be used
   to test the memory mechanism on stronger representations. Report its origin,
   pretraining data, and gradient training. Claims in this track concern the
   online memory/readout, not an end-to-end backprop-free system.

**Acceptance:** all models consume byte-identical feature/event streams, locked
evaluation is tested, and the protocol can run a small local fixture without
network access.

### WP3: Implement the required baseline matrix

**Partial status (2026-08-21):** cumulative RLS, the factor curve,
immediate-maturity features, online Adam controls, budgeted ALD-KRLS, a
direct-link Resource-Allocating Network, bounded fast-learning Fuzzy ARTMAP,
and OS-ELM are implemented. The four new analytic/constructive methods use
disjoint development tuning and CPAM-32/64 state ceilings. OS-ELM outperforms
CPAM at both ceilings, so the remaining specialized continual-learning and
replay controls cannot be treated as optional strengthening checks.

#### Analytic and simple online controls

- LMS with tuned learning rate.
- Cumulative RLS with factor 1.
- Full predeclared RLS forgetting-factor curve.
- Immediate-maturity local features without probation.
- OS-ELM or the closest reproducible sequential ELM configuration.
- ACIL and RanPAC-style analytic baselines where their assumptions match the
  class-incremental protocol.
- A kernel analytic/RLS baseline if the novelty audit identifies it as the
  closest structural competitor.

#### Backpropagation controls

- Online SGD and Adam linear classifiers.
- A state-near MLP trained once per observation.
- EWC and Synaptic Intelligence as non-replay regularization controls.
- Learning without Forgetting or an equivalent distillation control where
  labels/classes evolve.
- Experience Replay and DER++ as replay controls.

Replay methods violate CPAM's no-raw-replay contract but are necessary practical
comparators. Give them two budgets: the same persistent bytes as CPAM-32 and the
same bytes as CPAM-64. Count samples, labels, logits, indices, and buffer
metadata in that budget.

For each baseline, disclose whether it uses task boundaries, replay, pretrained
features, epochs, batches, gradients, or test-time adaptation. Do not mix
different information assumptions in a single ranking without marking them.

**Acceptance:** every headline comparison has matched features, stream order,
number of updates, tuning budget, evaluation points, and either matched state or
an explicit resource axis.

### WP4: Generalize and harden CPAM

- Rename the paper-facing method CPAM while retaining compatibility with
  existing `managed_memory_*` result names.
- Separate feature width, mature capacity, candidate capacity, class count, and
  image shape in configuration and result schemas.
- Support ten-class class-incremental evaluation and future output expansion
  without modifying old predictions during locked evaluation.
- Preserve the defining invariants:
  - every labeled event updates cumulative RLS statistics with unit weight;
  - there is no forgetting factor or age window in CPAM;
  - candidates do not affect prediction or RLS statistics;
  - a second nearby, same-label observation confirms a candidate;
  - a confirmed center becomes immutable;
  - active mature centers are never replaced;
  - mature and candidate capacities are bounded;
  - no raw observation buffer is retained.
- Add numerical-stability diagnostics for covariance symmetry, conditioning,
  finite updates, and long streams.
- Benchmark exact dense RLS against a block, diagonal, sketch, or inverse-free
  alternative only as a separately named approximation. Do not silently change
  CPAM's estimator to obtain favorable scaling.
- Test capacity saturation over streams substantially longer than the point at
  which all current capacities fill.

**Acceptance:** invariant tests pass; 60,000-example real-data streams complete;
checkpoints resume exactly; no mature center changes bitwise after promotion;
state stays independent of stream length at fixed capacities.

### WP5: Preregister tuning, hypotheses, and statistics

Freeze the analysis plan before the final test run.

Primary hypotheses:

1. CPAM improves pre-return retention and final mean accuracy over cumulative
   RLS without materially reducing first-shift adaptation.
2. CPAM occupies a point above the measured RLS forgetting-factor frontier.
3. Probation improves the adaptation-retention aggregate over immediate
   maturity at matched mature and candidate capacity.
4. Against a state-near online-backprop model without replay, CPAM improves
   retention and final joint performance, while backprop may retain an immediate
   adaptation advantage.

Primary metrics:

- prequential/online accuracy;
- first-shift adaptation accuracy or adaptation area under the curve;
- pre-return retention;
- return-online accuracy;
- final mean accuracy over all encountered domains/classes;
- average forgetting and backward transfer for class-incremental tests.

Resource metrics:

- persistent state bytes;
- peak host and accelerator memory;
- prediction and update latency separately;
- events per second;
- checkpoint size;
- approximate multiply-add counts where reliable.

Tuning policy:

- Use disjoint development seeds and test seeds.
- Give method families comparable trial budgets.
- Tune CPAM capacity, RBF width, minimum center distance, and regularization only
  on development data.
- Tune optimizer, learning rate, regularization, replay ratio, and method-specific
  parameters for baselines on the same development protocol.
- Never choose a configuration from final-test performance.

Statistics:

- Use at least ten paired seeds everywhere and target twenty for headline
  comparisons if compute permits.
- Report paired mean differences, 95% confidence intervals, and standardized
  effect sizes.
- Use a hierarchical or repeated-measures analysis across datasets for the
  overall claim rather than treating every seed from every dataset as
  independent.
- Correct families of secondary comparisons, for example with Holm's method.
- Publish all tested configurations and negative runs.

**Acceptance:** the analysis file, seed lists, configuration grids, hypotheses,
and primary figure definitions are committed before final test artifacts are
generated.

### WP6: Run the decisive experiments

Run in this order so failures are found cheaply:

1. Reproduce the 8x8 capstone and complete its full ablation chain.
2. Run MNIST and Fashion-MNIST controlled-feature domain streams.
3. Run their class-incremental streams.
4. Run CIFAR-10 with controlled fixed features.
5. Run the frozen-pretrained-feature track.
6. Run resource-matched replay and specialized continual-learning baselines.
7. Repeat headline comparisons at the final seed count.
8. Run long-stream capacity and numerical-stability tests.
9. Run CPU and GPU systems measurements in isolated benchmark jobs.

Each experiment writes an append-only result artifact containing per-event or
per-phase metrics, per-seed summaries, full configuration, software versions,
hardware, elapsed time, and state diagnostics.

**Scientific success criterion:** CPAM should lie above the RLS factor frontier
on at least two of the three core datasets, with the result supported by paired
uncertainty, and should offer a reproducible retention/final-performance
advantage over at least one state- or memory-matched gradient baseline. Report
the result even if this criterion fails; failure changes the paper into a
narrow or negative study rather than justifying selective omission.

### WP7: Produce the paper package

- Maintain a venue-neutral LaTeX source as the canonical manuscript.
- Maintain a detailed Markdown literature dossier and evidence ledger.
- Generate every numerical table and plot from result artifacts.
- Put the signed+magnitude encoder selection and compact polarity ablation in
  the main paper because they justify the stable representation.
- Put the moving predictive representation, predictive-surprise experiment,
  entropy, sparse responsibility, fast values, and adaptive keys in the
  supplement as informative negative or secondary results.
- Include a reproducibility checklist, limitations, broader impacts, compute
  statement, and data/license statement.
- Only after selecting a target venue, wrap the neutral content in the current
  official venue template and enforce its length/anonymity rules.

## 4. Main figures and tables to produce

1. **Method diagram:** fixed encoder -> base features -> mature local keys ->
   cumulative RLS prediction, with a separate dormant candidate path.
2. **Algorithm box:** predict, leverage calculation, candidate confirmation or
   proposal, cumulative unit-weight RLS update.
3. **RLS frontier plot:** adaptation versus retention/final mean for every
   forgetting factor, with CPAM capacities overlaid.
4. **Recurring-stream timeline:** domain order and locked evaluation points.
5. **Learning curves:** prequential accuracy through shifts and returns.
6. **Capacity frontier:** accuracy versus persistent state and update latency.
7. **Cross-dataset summary:** paired CPAM differences with confidence intervals.
8. **Baseline table:** quality and resource metrics with information assumptions.
9. **Ablation table:** cumulative RLS, immediate maturity, leverage, probation,
   candidate management, and capacities.
10. **Supplementary negative-results table:** entropy, responsibility, fast
    values, adaptive keys, moving predictive features, and predictive surprise.

## 5. Risks and decision rules

- **Closest prior art subsumes the mechanism:** revise the novelty claim toward
  the empirical analysis or extend the method before spending on final runs.
- **CPAM only helps inversion-like drift:** do not generalize; present it as a
  local-invariance result or stop the major-venue submission.
- **Natural-image fixed features are too weak:** use a shared frozen pretrained
  encoder, but narrow the claim to backprop-free online adaptation.
- **Exact RLS becomes infeasible:** report the quadratic boundary and evaluate a
  named approximation; do not conceal the algorithmic change.
- **Replay wins at matched bytes:** report it. The remaining value may be privacy,
  no raw storage, constant per-event operation, or analytic simplicity, but only
  if those advantages are measured and relevant.
- **Specialized backprop retains as well as CPAM:** remove any broad anti-backprop
  framing and focus on the conditions where CPAM remains useful.
- **Capacity saturates early:** characterize the saturation and either develop a
  principled consolidation/compression extension or state the lifetime bound.
- **No consistent multi-dataset gain:** publish as a scoped proof of concept or
  negative study rather than averaging away heterogeneous outcomes.

## 6. Deferred work

Recommendation systems, reinforcement learning, robotics, learned world models,
unsupervised representation learning, end-to-end JEPA-like learning, language,
and production self-modification are not required for this first paper. They
are strong follow-up directions only after the supervised CPAM memory claim is
established on standard continual-learning benchmarks.
