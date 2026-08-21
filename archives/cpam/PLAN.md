# No-Backprop Continual Learning: Project Plan

## 1. Objective

Build and evaluate a recurrent learning system that remains operational while it
learns. The system processes one event at a time, predicts before learning from
that event, updates from local state and broadcast feedback, and never performs
backpropagation through a stored history.

The research question is:

> Can local online plasticity trade some offline optimization performance for
> immediate adaptation, bounded temporal memory, and continuous operation?

The project is successful only if those claims are measured against conventional
and simpler baselines. Producing code that merely runs is not sufficient.

## 2. Definitions and Constraints

### Core learner

The core learner must satisfy all of the following:

- It follows a `predict -> score -> learn` order for every event.
- It does not use automatic differentiation, `.backward()`, or a computation
  graph retained across timesteps.
- It does not use backpropagation through time (BPTT).
- It consumes a stream without epochs or required task boundaries.
- Its working memory is independent of the total stream length.
- Its weight updates use current local activity, persistent eligibility state,
  and an optional broadcast signal such as error, reward, novelty, or surprise.
- It supports batch size one. Parallel independent streams may be added later
  for throughput without changing the learning rule.

### Comparison code

A conventional BPTT baseline may use PyTorch and automatic differentiation, but
it must live in a clearly separated `baselines` package. The core package must
not import or depend on that implementation. Automated checks will enforce this
boundary.

### Non-goals for the MVP

- Matching the accuracy of large backpropagation-trained models.
- Training a language model or transformer from scratch.
- Claiming lower energy use without hardware measurements.
- Claiming biological equivalence to a human brain.
- Allowing an unbounded replay buffer.
- Deploying a self-modifying model to an uncontrolled production environment.

## 3. Hypotheses

The experiments will test these hypotheses independently:

1. **Bounded temporal memory:** Core learner memory remains approximately flat
   as stream length increases, while full BPTT activation memory grows with its
   unroll window.
2. **Immediate adaptation:** The core learner improves after each observation
   and adapts to a distribution change without an offline retraining phase.
3. **No global backward pass:** Useful temporal behavior can emerge from local
   eligibility traces plus a broadcast learning signal.
4. **Operational continuity:** Inference, recurrent-state evolution, and learning
   can be interleaved without resetting the model or pausing for epochs.
5. **Stability is achievable:** Gating, normalization, and slow consolidation
   can reduce drift and forgetting enough for extended streaming experiments.

The project does not assume that the local learner will be faster, more accurate,
or more sample-efficient than BPTT. Those are empirical questions.

## 4. MVP Architecture

### 4.1 Streaming interface

Each environment emits one transition at a time:

```text
observation_t -> model prediction/action -> outcome_t and feedback_t
```

The model API should make ordering explicit:

```python
prediction = learner.predict(observation)
metrics.record(prediction, target)
learner.learn(target=target, reward=reward)
```

Calling `learn` before `predict`, or learning twice from the same event, should
raise an error in development mode.

### 4.2 Recurrent state

The initial model is a small tanh recurrent reservoir:

```text
h_t = tanh(W_in x_t + W_rec h_(t-1) + b)
y_t = W_out h_t + c
```

Start with a fixed, spectrally controlled recurrent matrix. This isolates online
readout learning before recurrent plasticity is introduced.

### 4.3 Online readout

Implement two readout update rules:

1. LMS / delta rule for a minimal constant-memory learner.
2. Recursive least squares (RLS) as a faster-adapting but higher-memory baseline.

Both update after every scored prediction and require no backpropagation.

### 4.4 Eligibility-based recurrent plasticity

After the fixed-reservoir baseline is stable, add an approximate local trace:

```text
eligibility_ij(t) = decay * eligibility_ij(t-1)
                    + post_sensitivity_i(t) * pre_activity_j(t-1)

hidden_signal_i(t) = fixed_feedback_i @ output_error(t)

delta_W_ij(t) = learning_rate
                * hidden_signal_i(t)
                * eligibility_ij(t)
```

The feedback projection is fixed and is not the transpose of the forward
weights. Updates may be accumulated for a short fixed interval, but no historical
activations or computation graphs may be retained.

### 4.5 Stability mechanisms

Add mechanisms individually so their effects can be measured:

- spectral-radius control at initialization
- update clipping
- weight decay or norm constraints
- activity homeostasis
- novelty/surprise-gated plasticity
- fast plastic weights with decay
- slow consolidation weights
- optional replacement of persistently inactive units

Replay is deliberately excluded from the first experiments. A bounded or
generative replay mechanism can be evaluated later as a separate intervention.

## 5. Benchmarks

All MVP benchmarks are generated locally, deterministic by seed, and require no
downloaded datasets.

### Benchmark A: Nonstationary signal prediction

A continuous signal changes frequency, phase, amplitude, or noise regime at
unannounced change points. The model predicts the next value.

Tests:

- immediate online improvement
- adaptation latency after a regime change
- memory and compute over a long stream
- recovery when a previously seen regime returns

### Benchmark B: Delayed association

A cue is presented, followed by distractor timesteps, and a later query requires
the earlier cue. Feedback arrives only at the query.

Tests:

- whether eligibility traces bridge delayed feedback
- performance as the delay increases
- comparison with fixed-reservoir and truncated-BPTT baselines

### Benchmark C: Continual classification

Synthetic classes arrive in changing or recurring contexts without explicit task
boundaries.

Tests:

- catastrophic forgetting
- sustained plasticity
- forward and backward transfer
- effects of gating and slow consolidation

Real datasets such as MNIST, audio, or control environments are deferred until
the mechanisms pass these synthetic tests.

## 6. Baselines

Every material claim needs an appropriate comparison:

1. Frozen random reservoir and frozen readout.
2. Fixed reservoir with online LMS readout.
3. Fixed reservoir with online RLS readout.
4. Plastic recurrent model with local eligibility traces.
5. Online/truncated BPTT model with a matched recurrent width.
6. Ablations with plasticity gating, consolidation, or homeostasis removed.

Model sizes, seeds, data order, and evaluation points must be matched wherever
possible. Parameter count and extra optimizer/trace state must be reported rather
than comparing model weights alone.

## 7. Measurements

### Learning quality

- prequential loss or accuracy: score before learning from each event
- rolling performance over a fixed window
- final performance and area under the online-learning curve
- performance across at least five random seeds

### Adaptation and retention

- steps required to recover after a distribution change
- error immediately before and after each change point
- performance when a previous regime returns
- forgetting and backward-transfer measures
- learning ability late in a long run versus early in the run

### Systems behavior

- peak resident memory and, if applicable, accelerator memory
- memory versus stream length and BPTT window length
- events per second
- prediction latency and update latency separately
- estimated multiply-add counts where practical
- checkpoint size and auxiliary state size

### Stability and diagnostics

- recurrent activity mean, variance, and saturation rate
- weight, update, and eligibility norms
- inactive-unit fraction
- effective representation rank where practical
- frequency of clipped or rejected updates

Energy efficiency is not claimed unless measured on suitable hardware.

## 8. Acceptance Gates

### Gate 1: Mechanical correctness

- The core package contains no autograd or backward calls.
- Unit tests verify trace decay and hand-computed updates on tiny networks.
- Predict-before-learn ordering is enforced.
- Runs are reproducible from a seed and configuration.

### Gate 2: Online learning

- The online readout improves over the frozen baseline on Benchmark A across a
  majority of seeds.
- Metrics are genuinely prequential and do not leak the current target.

### Gate 3: Temporal credit

- Eligibility-based recurrent plasticity improves on at least one delayed task
  over an otherwise matched learner without eligibility traces.
- The result is repeated across multiple seeds and delay lengths.

### Gate 4: Bounded memory

- Core learner memory reaches a steady range and shows no systematic growth when
  stream length is increased by at least an order of magnitude.
- BPTT memory is measured across several unroll lengths for comparison.

### Gate 5: Continual stability

- Adaptation and retention are reported separately.
- At least one stability intervention improves the tradeoff over the ungated
  plastic model without hiding failure cases.

Failure to pass a gate produces a documented result and a focused next
experiment; it does not justify silently changing the project definition.

## 9. Repository Shape

The intended layout is:

```text
no_backprop/
├── PLAN.md
├── README.md
├── pyproject.toml
├── src/
│   ├── no_backprop/
│   │   ├── streams.py
│   │   ├── reservoir.py
│   │   ├── readouts.py
│   │   ├── eligibility.py
│   │   ├── plasticity.py
│   │   ├── metrics.py
│   │   └── experiment.py
│   └── baselines/
│       └── bptt.py
├── tests/
│   ├── test_online_protocol.py
│   ├── test_readout_updates.py
│   ├── test_eligibility.py
│   ├── test_memory_bound.py
│   └── test_no_autodiff.py
├── configs/
└── results/
```

Generated checkpoints and large experiment outputs must not be committed.
Compact summaries and plots may be committed when they document a reproducible
result.

## 10. Implementation Sequence

### Milestone 0: Project contract and scaffold

- Create package metadata, CLI skeleton, configuration format, and test setup.
- Document the strict core/baseline boundary.
- Add deterministic random-number handling and structured result output.

Deliverable: tests run and an empty experiment can be reproduced.

### Milestone 1: Fixed reservoir and online readouts

- Implement streaming protocol and Benchmark A.
- Implement recurrent state, LMS, RLS, and frozen controls.
- Add prequential metrics, checkpointing, and long-stream memory measurement.

Deliverable: first complete no-backprop learner and comparison report.

### Milestone 2: Eligibility traces

- Implement local recurrent eligibility state and fixed feedback signals.
- Add hand-calculated unit tests and delayed-association benchmark.
- Compare immediate versus accumulated updates.

Deliverable: evidence for or against local temporal credit assignment.

### Milestone 3: Stability and continual learning

- Add Benchmark C.
- Evaluate gating, normalization, homeostasis, and fast/slow weights separately.
- Measure retention, late-run plasticity, and recurring-context recovery.

Deliverable: quantified stability/plasticity tradeoff and ablations.

### Milestone 4: Conventional baseline and systems comparison

- Add isolated matched BPTT baseline.
- Profile memory, throughput, update latency, and accuracy.
- Ensure reports distinguish inference memory, training memory, and auxiliary
  plasticity state.

Deliverable: defensible comparison with traditional training.

### Milestone 5: Domain expansion

Choose one domain based on the MVP results:

- audio or sensor prediction for streaming adaptation
- small robotics/control environment for delayed reward
- image stream for continual classification
- on-device or neuromorphic prototype for systems benefits

This milestone requires a separate scope decision and is not part of the MVP.

### Milestone 6: Scalable continual memory

- Compare exact, diagonal, and block-diagonal RLS with LMS and bounded class
  prototypes.
- Add a genuinely protected fast/slow memory and preserve negative ablations.
- Sweep forgetting factors on shuffled, class-ordered, and recurring-drift
  streams.
- Separate repeated exposure from augmentation effects.
- Measure stream-length scaling through 60,000 lazy 8x8 and 28x28 images.
- Measure and project feature-width time and memory scaling.

Deliverable: a measured stability/plasticity/complexity frontier and a selected
memory mechanism for the next representation-learning stage.

### Milestone 7: Forward-only predictive representations

- Build a JEPA-inspired context/target representation objective without
  reproducing I-JEPA or importing backpropagation into the core.
- Train the predictor with the selected Milestone 6 memory rule.
- Compare fixed, locally plastic, and slowly updated target encoders.
- Measure representation collapse, effective rank, drift, downstream accuracy,
  retention, and bounded learner state.

Deliverable: evidence for or against learning useful predictive representations
with continuous forward-only updates.

## 11. Experiment Discipline

- Every run writes its full configuration, seed, code revision, and summary.
- Hyperparameters are tuned on designated development streams, not test streams.
- Failed runs and negative results are retained in compact metadata.
- Each claimed improvement needs an ablation and multiple seeds.
- Mechanism changes and hyperparameter changes are not mixed in one comparison.
- The simplest baseline that explains a result is preferred.

## 12. Git Strategy

The repository currently has no commits. For the MVP, establish one verified
baseline before creating experimental branches. If Git management is authorized,
use milestone commits such as:

```text
scaffold streaming experiment framework
implement fixed reservoir and online readouts
add eligibility-based recurrent plasticity
measure continual stability and bounded memory
compare against matched BPTT baseline
```

After the baseline is established, use branches for isolated research variants,
for example `experiment/replay` or `experiment/predictive-coding`.

## 13. First Execution Target

The first implementation pass should complete Milestones 0 and 1 only. It should
end with a reproducible command that:

1. Generates a nonstationary stream.
2. Runs frozen, LMS, and RLS reservoir models.
3. Scores each prediction before updating.
4. Produces rolling error, adaptation, memory, and throughput summaries.
5. Passes unit and long-stream memory tests.

Only after this vertical slice is trustworthy should recurrent eligibility
plasticity be introduced.
