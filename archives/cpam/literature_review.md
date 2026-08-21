# Literature and Novelty Audit for CPAM

Last updated: 2026-08-21  
Literature cutoff: 2026-08-21  
Working method: Cumulative Probationary Associative Memory (CPAM)

## 1. Purpose and method

This report supports WP1 of [future_work.md](future_work.md). It asks a narrower
question than a general continual-learning survey:

> Which prior methods already contain CPAM's individual mechanisms or their
> combination, and what claim remains defensible?

The search prioritized original papers and official proceedings or publisher
pages. It covered six overlapping literatures:

1. catastrophic interference and stability-plasticity;
2. adaptive filters, kernel RLS, and online sequential analytic learning;
3. resource-allocating, constructive, and adaptive-resonance networks;
4. modern continual-learning regularization, replay, and expansion;
5. analytic class-incremental learning with frozen representations;
6. backpropagation alternatives and predictive representation learning.

This is a reproducible starting audit, not a claim that every paper using a
different name has been found. The highest-risk remaining search is older and
less-indexed work combining budgeted kernel RLS with supervised dictionary
growth. The audit should be refreshed immediately before submission.

## 2. CPAM mechanism being audited

The implemented method must be compared as a conjunction of properties, not by
the generic labels “continual,” “RBF,” “RLS,” or “backpropagation-free.”

1. A stable fixed encoder emits base vector `x_t`.
2. Prediction uses one feature vector containing base coordinates and every
   mature RBF activity; there is no expert router.
3. The output readout is exact recursive ridge regression with forgetting factor
   exactly one. Every labeled event updates its cumulative statistics once.
4. A prediction error becomes structurally eligible only when its normalized
   RLS leverage is below the cumulative pre-update mean and it is sufficiently
   distant from mature centers.
5. An eligible error creates a dormant candidate. The candidate has no effect on
   prediction or the RLS feature vector.
6. A later nearby event with the same label confirms the candidate. Their online
   centroid becomes a mature RBF center.
7. A mature center is immutable and is never replaced.
8. A bounded candidate bank may reclaim a now-correctly-classified proposal or
   replace a more novel unresolved proposal with a less-novel one. This policy
   reuses provisional structural summaries only.
9. No raw-example replay buffer, gradient, optimizer, task identity, age window,
   or temporal forgetting factor is used by CPAM during the stream.
10. Mature and candidate capacity are fixed, so persistent state is independent
    of stream length but grows quadratically with expanded feature width under
    exact dense RLS.

The unusual design choice is that **low**, rather than high, leverage admits a
proposal after startup: high leverage means the shared representation has not
yet accumulated enough evidence to justify permanent local structure.

## 3. Executive conclusion

### 3.1 Claims ruled out by prior art

The paper must not claim any of the following:

- the first backpropagation-free continual learner;
- the first replay-free analytic continual learner;
- the first use of RLS for continual or sequential learning;
- the first online learner with dynamically added local/RBF units;
- the first learner to combine prediction error and novelty for recruitment;
- the first network that freezes recruited features;
- the first bounded or sparsified kernel/RBF online model;
- the first architecture motivated by stability-plasticity;
- the first method to retain cumulative statistics instead of raw samples.

Each of these ideas has clear prior art.

### 3.2 Defensible provisional novelty

No paper found in this audit contains the exact CPAM combination:

> unit-weight cumulative RLS over a single prediction path, with below-mean RLS
> leverage used as familiarity rather than novelty, a supervised two-observation
> dormant probation stage, immutable confirmed local centers, and bounded reuse
> restricted to provisional candidates rather than mature representations.

The safest contribution claim is therefore:

> CPAM is a bounded online associative feature-management rule for cumulative
> RLS. It delays permanent local feature creation until an error recurs in a
> familiar, label-consistent region, preserving a stable mature basis while
> avoiding explicit temporal discounting and raw replay. Empirically, it is
> evaluated by whether it moves above the measured RLS forgetting-factor
> frontier under recurring nonstationarity.

This is a **novel integration and evaluation claim**, not a claim that RLS,
leverage, RBF recruitment, confirmation, freezing, or bounded dictionaries are
individually new.

### 3.3 Novelty risk

Novelty risk is **medium**. CPAM is closest to the intersection of:

- Platt's resource-allocating network and its minimal/growing successors;
- sparse kernel RLS dictionary construction;
- ARTMAP's stable supervised category recruitment;
- Cascade-Correlation's frozen recruited features; and
- ACIL/RanPAC-style cumulative analytic continual learning.

The empirical contribution must therefore be substantial. A small accuracy gain
on one dataset will not carry the method solely on architectural novelty.

## 4. Closest-prior-art matrix

Legend: “No replay” means no stored raw-example rehearsal. “Fixed mature” means
an accepted feature/category is protected from later center movement; output
weights may still change.

| Method | Online/sequential | Analytic/RLS output | Adds local features/categories | Error or label controls growth | Confirmation before use | Fixed mature feature | Bounded without replacing mature features | No raw replay | Closest difference from CPAM |
|---|---|---|---|---|---|---|---|---|---|
| Factor-1 RLS | Yes | Yes | No | No | No | Fixed basis | Yes | Yes | CPAM adds supervised probationary local coordinates |
| Sparse KRLS | Yes | Yes | Kernel dictionary | Usually input-space novelty/linear dependence | No | Dictionary commonly retained | Variant-dependent | Yes | Adds a kernel atom immediately; generally unsupervised dictionary test |
| RAN | Yes | LMS originally | RBF units | Novel input plus large error | No | Centers retained; parameters may adapt | No in original | Yes | Immediate recruitment, hand-set thresholds, no dormant confirmation |
| MRAN/GGAP-RBF | Yes | Second-order or sequential variants | RBF units | Error, novelty, significance | Usually windowed evidence, not dormant prediction-free confirmation | Can adapt/prune | Prunes active units | Yes | Explicit growth/pruning and sliding criteria rather than immutable mature memory |
| ARTMAP | Yes | No | Recognition categories | Label mismatch and vigilance | Resonance/search within a trial | Stable category learning | Capacity/vigilance dependent | Yes | Match tracking and winner categories, not cumulative RLS or two-event probation |
| Cascade-Correlation | Stage-wise | Output rule plus candidate optimization | Hidden units | Residual correlation | Candidate-training phase | Incoming weights frozen | Generally grows | Reuses training set | Not strict single-pass online; candidates are trained before installation |
| OS-ELM | Yes/chunked | RLS output | Fixed random hidden layer | No | No | Fixed random basis | Fixed width | Yes | No data-dependent structural memory |
| ACIL | Phase/chunk incremental | Recursive ridge | Fixed random expansion | New classes/phases | No | Frozen backbone/expansion | Fixed width | Yes | Analytic class incremental learning without local online recruitment |
| GKEAL | Phase incremental | Recursive analytic classifier | Fixed Gaussian kernel embedding | Few-shot class phases | No | Frozen backbone | Fixed configured map | Yes | Kernel embedding is designed before continual phases rather than probationary growth |
| RanPAC | Phase incremental | Gram/covariance analytic classifier | Fixed random projection | Class statistics | No | Frozen pretrained features | Fixed width | Yes | Strong fixed representation and analytic head, no event-triggered local features |
| Any-SSR | Task incremental | RLS router | Task-specific LoRA experts | Explicit task progression | No | Frozen base plus separate experts | Grows per task | Yes | RLS routes among task modules; CPAM has one path and no task identity |
| Forward Projection | One pass, layer-wise | Closed-form ridge | Fixed-width learned layers | Random input/label target projection | No | Layer fixed after fit | Fixed width | No replay | General feedback-free representation fitting, not continual drift/memory |
| CPAM | Event-wise | Unit-weight RLS | Mature local RBF features | Error, label, distinctness, low leverage | **Yes: later nearby same-label event** | **Yes, bitwise immutable centers** | **Yes: only dormant slots reused** | Yes | Audited method |

## 5. Foundations: why stability and plasticity conflict

### 5.1 Catastrophic interference

[McCloskey and Cohen (1989)](https://doi.org/10.1016/S0079-7421(08)60536-8)
demonstrated that sequential learning can catastrophically alter distributed
representations when new updates reuse weights supporting earlier mappings.
This is the foundational problem CPAM addresses experimentally. CPAM does not
guarantee zero interference: its cumulative output weights remain shared, and
new features lack retrospective activations for old samples.

[Kirkpatrick et al. (2017)](https://doi.org/10.1073/pnas.1611835114) introduced
Elastic Weight Consolidation (EWC), using a Fisher-weighted quadratic penalty to
slow changes to parameters important for prior tasks. EWC is relevant as a
gradient-based no-replay comparator, not close architectural prior art. It
typically assumes identifiable task transitions for importance consolidation,
whereas current CPAM operates without task labels.

[Zenke, Poole, and Ganguli (2017)](https://proceedings.mlr.press/v70/zenke17a.html)
introduced Synaptic Intelligence (SI), accumulating an online path-integral
measure of parameter contribution and later protecting important parameters.
Its ongoing importance accounting makes it a stronger comparison than ordinary
Adam, although it remains gradient-based and commonly uses task boundaries for
consolidation.

### 5.2 Complementary learning systems

[McClelland, McNaughton, and O'Reilly
(1995)](https://doi.org/10.1037/0033-295X.102.3.419) argued for complementary
fast episodic and slow cortical learning systems, with gradual interleaved
consolidation. This is conceptual motivation for the project's early fast/slow
experiments, not evidence that CPAM implements a biological complementary
learning system. CPAM's successful form has one cumulative prediction path;
dormant candidates are structural proposals, not an episodic replay system.

### 5.3 Adaptive resonance and stable category learning

[ART 2](https://pubmed.ncbi.nlm.nih.gov/20523470/) learns stable recognition
categories online using top-down expectations, match/reset dynamics, and a
vigilance parameter. [ARTMAP](https://doi.org/10.1016/0893-6080(91)90012-T)
extends ART to supervised nonstationary classification: prediction mismatch can
raise vigilance and drive category search or creation. ARTMAP is serious prior
art for stable supervised recruitment.

Differences from CPAM:

- ART uses competitive category selection, resonance, and a manually selected
  vigilance baseline; CPAM uses dense local activities plus one RLS readout.
- ART can create a category during the current learning trial; CPAM keeps a
  proposal prediction-free until a later same-label local confirmation.
- CPAM's gate is derived from current RLS covariance and its cumulative mean,
  although RBF width and distance remain configured hyperparameters.
- CPAM's mature centers are immutable and candidate reuse is explicitly
  separated from mature memory.

ARTMAP should appear in the main related-work section, not only the supplement.

## 6. Adaptive filters and analytic online learning

### 6.1 LMS and RLS

The Widrow-Hoff/LMS family supplies the minimal per-event delta update. Its
constant learning rate gives plasticity but repeatedly changes the same shared
weights. Recursive least squares instead maintains an inverse correlation
matrix and recursively computes the regularized least-squares solution for a
fixed basis. A forgetting factor below one exponentially downweights earlier
evidence; factor one retains unit-weight cumulative sufficient statistics.

These algorithms are foundations, not contributions. The paper must state that
CPAM's structural rules sit on top of ordinary factor-1 RLS.

### 6.2 Kernel recursive least squares: the most important collision

[Engel, Mannor, and Meir
(2004)](https://doi.org/10.1109/TSP.2004.830985) introduced kernel recursive
least squares (KRLS) with online sparsification. Their approximate linear
dependency criterion decides whether a new input contributes an independent
kernel dictionary element, allowing nonlinear RLS without retaining every
sample as a center.

This rules out “first dynamic nonlinear RLS memory” and “first RLS novelty-based
feature allocation.” The paper must directly compare CPAM with a budgeted KRLS
or explain why exact matching is infeasible.

The distinction is still meaningful:

- KRLS dictionary admission is normally based on input-space/kernel linear
  dependence and occurs when the sample arrives. CPAM's proposal is supervised,
  error-driven, and admitted when RLS leverage is *low enough to indicate
  familiarity*.
- A CPAM proposal is dormant and cannot affect prediction until a later nearby
  same-label event confirms it.
- CPAM protects mature centers and recycles only provisional proposals; budgeted
  kernel methods often delete, merge, project, or forget active dictionary
  elements.
- CPAM preallocates new coordinates in one finite RLS matrix. It is not an exact
  kernel expansion over all past samples.

The novelty section should use “probationary feature management for cumulative
RLS,” not “a novel kernel RLS algorithm,” unless future mathematical work
supports that stronger description.

### 6.3 Extreme learning machines and OS-ELM

[Huang, Zhu, and Siew
(2006)](https://doi.org/10.1016/j.neucom.2005.12.126) formalized Extreme
Learning Machines (ELM): random fixed hidden features with analytically fitted
output weights. [Liang et al.
(2006)](https://doi.org/10.1109/TNN.2006.880583) proposed Online Sequential ELM
(OS-ELM), using RLS-style updates for samples arriving one-by-one or in chunks.

OS-ELM is a mandatory simple analytic baseline because it combines fixed
backpropagation-free features and sequential least squares. CPAM differs by
allocating supervised local coordinates online rather than fixing random hidden
features before the stream.

ELM terminology should be handled carefully. Random-feature models and analytic
single-layer learning predate the ELM name; the paper need not enter that naming
debate. It only needs to represent the OS-ELM mechanism accurately.

## 7. Resource-allocating and constructive networks

### 7.1 Resource-Allocating Network

[Platt (1991)](https://doi.org/10.1162/neco.1991.3.2.213) proposed a
Resource-Allocating Network (RAN) that adds a local computational unit when an
input is unusual and the prediction error is large. Otherwise it updates the
existing network with LMS. RAN learns online without repeated presentation of
training patterns.

RAN is the closest conceptual ancestor of CPAM and must be prominent in the
paper. It already establishes:

- online local-unit recruitment;
- joint novelty and error conditions;
- compact representations without repeating samples; and
- RBF-like locality for rapid adaptation.

CPAM's differentiators are not “neuron recruitment” or “local memory.” They are
the covariance-derived familiarity interpretation, the dormant two-event
supervised probation stage, immutable maturity, unit-weight cumulative RLS, and
bounded provisional-only management.

### 7.2 Minimal and growing/pruning RBF networks

[Yingwei, Sundararajan, and Saratchandran
(1997)](https://doi.org/10.1162/neco.1997.9.2.461) combined RAN growth with
pruning based on a hidden unit's output contribution, producing the Minimal RAN
(MRAN). [Huang, Saratchandran, and Sundararajan
(2005)](https://doi.org/10.1109/TNN.2004.836241) proposed GGAP-RBF, growing and
pruning units using significance estimates.

Some RAN successors require error to remain large across a window before
growth. That superficially resembles CPAM confirmation, but it is not the same
mechanism:

- a residual window is an age-limited sequence statistic;
- CPAM stores a dormant labeled structural proposal that is matched by local
  geometry on a later event;
- windowed RAN criteria generally install a unit after threshold satisfaction,
  while CPAM ensures the provisional center never enters historical RLS
  statistics;
- MRAN/GGAP actively prune installed units, whereas CPAM never replaces mature
  centers.

The final paper should avoid the phrase “the first method requiring persistent
error before recruitment.” Use the precise phrase “prediction-free,
label-consistent probation before immutable activation.”

### 7.3 Cascade-Correlation

[Fahlman and Lebiere
(1989)](https://papers.nips.cc/paper_files/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html)
introduced Cascade-Correlation. It starts with a minimal network, trains
candidate hidden units against residual error, installs a selected unit, and
freezes its incoming weights. It demonstrates that constructed, frozen features
and no backpropagation through already-installed hidden units are old ideas.

Unlike CPAM, Cascade-Correlation alternates optimization phases over a reusable
training set, explicitly trains candidate input weights, and normally grows the
network until a stopping condition. CPAM is event-wise, uses local centers
formed from two observations, and bounds both mature and candidate banks.

### 7.4 Growing Neural Gas and topology learners

Growing Neural Gas and related competitive-learning models incrementally add
nodes to represent input topology. They matter as broad structural-learning
context but are less direct than RAN and ARTMAP: they are generally unsupervised,
do not use cumulative supervised RLS, and learn topology rather than a bounded
error-conditioned classifier. Cite the original Fritzke work in the final
bibliography after verifying the archival proceedings record.

## 8. Modern analytic continual learning

### 8.1 ACIL

[Zhuang et al. (2022)](https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html)
introduced Analytic Class-Incremental Learning (ACIL). With a fixed feature
mapping and recursive analytic update, ACIL reproduces the joint regularized
least-squares solution without retaining historical examples. It explicitly
claims “absolute memorization” in this weight-equivalence sense.

ACIL invalidates any broad claim that replay-free RLS is new. It also supplies a
standard for mathematical precision: CPAM should **not** inherit ACIL's batch
equivalence claim after feature recruitment, because past events were evaluated
before later RBF coordinates existed.

ACIL is class-incremental and phase/chunk oriented, whereas current CPAM is
event-wise and domain-recurring. A class-incremental CPAM comparison with ACIL
is required for a journal paper.

### 8.2 GKEAL

[Zhuang et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.html)
introduced Gaussian Kernel Embedded Analytic Learning for few-shot
class-incremental learning. It freezes a backbone and combines a Gaussian kernel
embedding with a recursively updated analytic classifier.

GKEAL is especially close in vocabulary—Gaussian/local representation plus
analytic learning—but its kernel representation is constructed as part of a
phase-wise architecture, not proposed, confirmed, and frozen event-by-event in
response to recurring errors. It is nevertheless a mandatory related-work
comparison and a desirable baseline if its protocol can be matched.

### 8.3 RanPAC

[McDonnell et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html)
showed that pretrained representations, random projections, and analytic class
statistics can be highly effective for domain- and class-incremental learning.
RanPAC strengthens two lessons for this project:

1. representation quality can dominate the continual-learning head; and
2. analytic heads over frozen features are competitive modern baselines, not
   merely historical adaptive filters.

CPAM's controlled fixed-feature experiments isolate its memory mechanism. A
second shared-pretrained-feature track is required to show whether probationary
recruitment adds value once the representation is strong.

### 8.4 Recent analytic extensions

[AnaCP (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/699b19e638a086ed3a6d1710c5aea504-Paper-Conference.pdf)
learns an analytic contrastive projection to adapt features while retaining a
closed-form continual-learning structure. It is relevant to CPAM's negative
moving-representation result and shows that recent analytic CL is moving beyond
fixed random expansions.

[Any-SSR (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html)
uses RLS to train a router over task-specific LoRA subspaces for continual LLM
learning. It differs sharply from CPAM—explicit expert isolation and routing
versus one shared prediction path—but shows that analytic routing and
replay-free task assignment are already active research areas.

[Recursive Metadata Normalization (ICML
2025)](https://proceedings.mlr.press/v267/shah25a.html) uses RLS to continually
remove confounder effects in intermediate representations. It is not a direct
classifier or memory competitor, but it is relevant to the claim that recursive
statistics can stabilize features under distribution change.

F-OAL and other 2024–2026 analytic-online submissions should remain in a
watchlist until their final archival status and exact protocols are verified.
Do not cite a withdrawn or anonymous review version as established prior art.

## 9. Replay and gradient-based continual-learning comparators

These methods are not close architectural ancestors, but they define the
empirical bar. A journal paper cannot justify CPAM from ordinary Adam alone.

### 9.1 Regularization and distillation

- [EWC](https://doi.org/10.1073/pnas.1611835114) penalizes movement of
  parameters important to earlier tasks.
- [Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html)
  accumulates online parameter-importance information.
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282) uses
  distillation targets on new-task inputs to preserve previous outputs without
  retaining old raw data.

All use gradients, but EWC and SI store summaries rather than old examples,
making them important no-raw-replay controls. Their task-boundary assumptions
must be disclosed. LwF retains old behavior through soft targets, which is a
different form of historical information from CPAM's sufficient statistics.

### 9.2 Episodic replay and gradient constraints

[GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
stores episodic samples and constrains new gradients to avoid increasing their
loss. It also introduced forward/backward-transfer evaluation measures.

[A-GEM](https://openreview.net/forum?id=Hkf2_sC5FX) reduces GEM's cost using an
average episodic-memory gradient and promotes single-pass evaluation with
disjoint development tasks. It is methodologically important to CPAM's final
protocol even if a simpler replay baseline is a more direct resource match.

[Chaudhry et al. (2019)](https://arxiv.org/abs/1902.10486) showed that simple
experience replay with very small episodic memories can be a strong single-pass
baseline. This is why CPAM must face byte-matched replay rather than arguing
from “no replay” alone.

[Maximally Interfered Retrieval](https://papers.nips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
selects replay items predicted to be most harmed by the incoming update. It is a
stronger but more expensive retrieval baseline and useful context for CPAM's
error-triggered structural allocation.

[Dark Experience Replay and
DER++](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
combine rehearsal with stored logits and, for DER++, ground-truth replay. The
paper targets general continual learning without clean task boundaries, making
DER++ a required practical comparator for CPAM's recurring streams.

### 9.3 Expandable gradient-trained architectures

[Progressive Neural Networks](https://arxiv.org/abs/1606.04671) freeze prior
task columns and add a new column per task, avoiding forgetting at state that
grows with tasks. [Dynamically Expandable Networks
(DEN)](https://openreview.net/forum?id=Sk7KsfW0-) selectively retrain and expand
network capacity, with splitting/duplication to limit semantic drift.

These methods show that protected expansion is established continual-learning
strategy. They typically use explicit task transitions, gradient optimization,
and task-specific structure. CPAM instead uses bounded event-triggered local
features without a task descriptor. The final paper should compare concepts and
resource scaling, even if implementing both is outside the first baseline tier.

## 10. Backpropagation-free learning context

### 10.1 Feedback and target alignment

[Lillicrap et al. (2016)](https://www.nature.com/articles/ncomms13276) showed
that fixed random feedback weights can replace exact transposed forward weights
for error transport under feedback alignment. [Nokland
(2016)](https://papers.nips.cc/paper_files/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html)
sent output error directly to hidden layers with Direct Feedback Alignment.
[Difference Target Propagation](https://arxiv.org/abs/1412.7525) uses learned
approximate inverses to provide layer targets.

These methods address deep credit assignment. CPAM does not: its encoder and
mature centers are fixed, while its output layer is solved analytically. Cite
them to delimit scope, not to imply direct competition.

### 10.2 Forward-only and closed-form alternatives

[Hinton's Forward-Forward algorithm](https://arxiv.org/abs/2212.13345) replaces
the usual forward/backward pair with positive and negative forward passes and
layer-local objectives. It still performs iterative optimization and requires a
negative-data construction; it is not a cumulative analytic memory.

[O'Shea and Rajendran's Forward Projection
(2026)](https://www.nature.com/articles/s41467-026-69161-1) is especially
important current context. It generates local hidden targets from random
projections of inputs and labels, then fits layers with closed-form ridge
regression in a single pass without retrograde communication. It demonstrates
that closed-form feedback-free representation learning is now a peer-reviewed
research direction.

Forward Projection is not a continual-learning or recurring-drift method: it
collects sufficient statistics for a layer, completes its fit, and moves to the
next layer. CPAM updates one output/memory system after every event and manages
recurring structural error. A future combination—Forward Projection for a
stable or append-only encoder plus CPAM for online memory—is plausible, but not
part of the current contribution.

### 10.3 Appropriate backpropagation-free claim

Use:

> CPAM requires no backward pass or gradient optimizer during the continual
> stream because its shared output mapping is updated analytically and its local
> centers are created by a non-gradient structural rule.

Do not use:

> CPAM solves deep credit assignment or replaces end-to-end backpropagation.

## 11. Predictive and stable representations

[I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)
predicts representations of masked image regions from visible context using
learned context and target encoders. The repository's “JEPA-inspired” probe is
not I-JEPA: it uses a small fixed spatial basis and cumulative RLS predictor.

That probe learned its masked target but moved the coordinate system consumed
by downstream cumulative statistics, worsening final retention. The result is a
useful negative observation:

> cumulative sufficient statistics are only directly meaningful while their
> feature coordinates remain stable.

This does not prove predictive representation learning is incompatible with
CPAM. It motivates append-only features, frozen targets, covariance transport,
or explicit consolidation. Keep this work in the supplement/future work rather
than presenting it as a co-equal paper contribution.

## 12. Detailed positioning by CPAM component

### 12.1 Unit-weight cumulative RLS

**Prior art:** classical RLS, OS-ELM, ACIL, GKEAL, RanPAC, Any-SSR.  
**What may remain:** using cumulative RLS as the stable shared substrate under
probationary feature recruitment.  
**Required comparison:** factor sweep, OS-ELM, ACIL/RanPAC, KRLS.

### 12.2 RLS leverage

**Prior art:** leverage and covariance novelty are standard in recursive
regression and sparse dictionary methods.  
**What may remain:** using below-cumulative-mean normalized leverage as a
threshold-free familiarity gate for *error-conditioned structural proposals*.
  
**Risk:** an equivalent “coherence after familiarity” rule may exist under
kernel sparsification terminology. Search this before submission.

### 12.3 Two-observation probation

**Prior art:** error persistence windows in RAN successors; resonance/match
testing in ART; candidate-training phases in Cascade-Correlation.  
**What may remain:** a dormant, prediction-free, labeled proposal confirmed by
a later local match before exposing a fixed coordinate to cumulative RLS.  
**Required ablation:** immediate maturity versus probation at matched capacity.

### 12.4 Immutable mature centers

**Prior art:** frozen feature detectors in Cascade-Correlation; protected
categories in ART; fixed dictionaries/features in many kernel and analytic
methods.  
**What may remain:** combining immutability with provisional-only bounded reuse
and cumulative output learning.  
**Required evidence:** bitwise-center invariants and return reactivation counts.

### 12.5 Bounded candidate management

**Prior art:** budget maintenance, pruning, merging, and dictionary deletion in
online kernel/RBF methods.  
**What may remain:** recycling only prediction-inactive candidates while never
deleting mature features or removing their originating events from cumulative
statistics.  
**Required comparison:** unmanaged versus managed banks, candidate capacities,
and a budgeted KRLS/RBF baseline with the same bytes.

### 12.6 No raw replay

**Prior art:** EWC, SI, LwF, OS-ELM, ACIL, GKEAL, RanPAC, and many parameter
isolation methods.  
**What may remain:** no-replay behavior paired with the CPAM structural rule.
  
**Required comparison:** byte-matched replay anyway; privacy/storage constraints
do not imply superior predictive performance.

## 13. Baseline priority implied by the audit

### Tier A: mandatory closest-mechanism baselines

1. Cumulative RLS and the complete forgetting-factor frontier.
2. Immediate-maturity CPAM ablation.
3. OS-ELM with matched fixed representation and state.
4. Budgeted sparse KRLS with disclosed dictionary admission/deletion policy.
5. A RAN/MRAN-style online RBF learner with tuned novelty/error criteria.
6. ACIL and RanPAC-style analytic class-incremental baselines.
7. ARTMAP or Fuzzy ARTMAP on compatible feature streams.

If implementation cost forces prioritization, KRLS and RAN rank above adding
another ordinary optimizer because they challenge method novelty directly.

### Tier B: mandatory empirical continual-learning baselines

1. Online SGD and Adam, linear and state-near MLP.
2. EWC and Synaptic Intelligence.
3. Experience Replay at CPAM-32 and CPAM-64 byte budgets.
4. DER++ at those same byte budgets.
5. A distillation method for class-incremental streams.

### Tier C: strengthening baselines

- GKEAL where its few-shot protocol can be reproduced fairly;
- A-GEM or MIR;
- a dynamically expandable gradient architecture;
- AnaCP or another learned analytic projection on pretrained features;
- Forward Projection as a separate representation-learning comparison, not as a
  direct continual-memory baseline.

## 14. Required paper wording

### Related-work paragraph skeleton

> CPAM lies at the intersection of analytic continual learning and constructive
> local networks. Recursive analytic learners such as OS-ELM and ACIL preserve
> cumulative information without exemplar replay, while kernel RLS methods add
> sparse nonlinear dictionary elements online. Resource-allocating networks and
> ARTMAP independently established error/novelty-driven or mismatch-driven
> category growth, and Cascade-Correlation froze installed feature detectors.
> CPAM does not originate these components. It combines them differently: an
> error in an RLS-familiar region creates a prediction-inactive labeled
> candidate; only a later local same-label match exposes an immutable RBF
> coordinate; and bounded management reuses provisional candidates without
> replacing mature features or discounting the shared unit-weight RLS
> statistics.

### Contribution sentence

> We propose and evaluate a probationary feature-management rule for cumulative
> RLS that separates structural proposal, confirmation, and immutable maturity,
> and test whether this separation improves the recurring-domain
> adaptation-retention frontier under fixed memory capacity.

### Backpropagation sentence

> The online CPAM update is backpropagation-free, but the method is not presented
> as a general solution to deep credit assignment; its stable encoder is fixed,
> and its exact RLS state scales quadratically with expanded feature width.

## 15. Open literature questions before submission

1. Does a budgeted KRLS variant already use delayed dictionary admission after
   repeated supervised confirmation?
2. Does a RAN/MRAN descendant explicitly maintain dormant candidates that do not
   affect output before promotion?
3. Has ARTMAP been combined with exact RLS output learning and immutable
   prototype centers?
4. Is there prior work using low leverage as familiarity to trigger expansion,
   rather than high leverage as novelty to expand the dictionary?
5. Do recent analytic CL methods dynamically expand their feature basis while
   preserving/re-aligning old sufficient statistics?
6. Which 2025–2026 analytic-online papers are archival and which remain
   preprints, withdrawn submissions, or workshop work?
7. What is the strongest public implementation for byte-matched KRLS, RAN,
   ARTMAP, ACIL, and RanPAC under a one-pass event protocol?

Record answers with exact equations and page references in the next audit, not
only abstracts.

## 16. Annotated primary-source bibliography

### Stability, consolidation, and continual learning

- **McCloskey & Cohen (1989), Catastrophic Interference in Connectionist
  Networks.** Foundational sequential-interference demonstration.
  [DOI](https://doi.org/10.1016/S0079-7421(08)60536-8)
- **McClelland, McNaughton & O'Reilly (1995), Why There Are Complementary
  Learning Systems.** Fast hippocampal storage and slow interleaved cortical
  learning; conceptual motivation, not a CPAM implementation.
  [DOI](https://doi.org/10.1037/0033-295X.102.3.419)
- **Kirkpatrick et al. (2017), EWC.** Gradient regularization using estimated
  parameter importance. [PNAS](https://doi.org/10.1073/pnas.1611835114)
- **Zenke, Poole & Ganguli (2017), Synaptic Intelligence.** Online accumulation
  of synaptic contribution followed by protection.
  [PMLR](https://proceedings.mlr.press/v70/zenke17a.html)
- **Li & Hoiem (2016), Learning without Forgetting.** No-old-data distillation
  baseline. [arXiv record](https://arxiv.org/abs/1606.09282)

### Analytic, kernel, and sequential learning

- **Engel, Mannor & Meir (2004), Kernel Recursive Least Squares.** Online kernel
  regression with sparse dictionary admission; closest analytic structural
  prior. [DOI](https://doi.org/10.1109/TSP.2004.830985)
- **Huang, Zhu & Siew (2006), Extreme Learning Machine.** Fixed random hidden
  features with analytic output fitting.
  [DOI](https://doi.org/10.1016/j.neucom.2005.12.126)
- **Liang et al. (2006), Online Sequential ELM.** Sequential RLS output updates
  over fixed random features. [DOI](https://doi.org/10.1109/TNN.2006.880583)
- **Zhuang et al. (2022), ACIL.** Recursive ridge class-incremental learning with
  fixed features and joint-solution equivalence.
  [NeurIPS](https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html)
- **Zhuang et al. (2023), GKEAL.** Gaussian kernel embedding plus recursive
  analytic few-shot class-incremental learning.
  [CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.html)
- **McDonnell et al. (2023), RanPAC.** Pretrained features, random projections,
  and analytic class statistics for continual learning.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html)
- **Shah et al. (2025), Recursive Metadata Normalization.** RLS inside changing
  representations to remove confounding effects.
  [PMLR](https://proceedings.mlr.press/v267/shah25a.html)
- **Tong et al. (2025), Any-SSR.** RLS task routing over isolated LoRA experts.
  [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html)
- **AnaCP (2025), Analytic Contrastive Projection.** Analytic adaptation of
  feature representation for continual learning.
  [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/699b19e638a086ed3a6d1710c5aea504-Paper-Conference.pdf)

### Constructive and category-growing networks

- **Fahlman & Lebiere (1989), Cascade-Correlation.** Trains and installs hidden
  units, then freezes incoming weights.
  [NIPS](https://papers.nips.cc/paper_files/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html)
- **Platt (1991), Resource-Allocating Network.** Adds local units under novelty
  and error criteria in an online stream.
  [DOI](https://doi.org/10.1162/neco.1991.3.2.213)
- **Carpenter, Grossberg & Reynolds (1991), ARTMAP.** Supervised real-time stable
  category learning using prediction mismatch and vigilance.
  [DOI](https://doi.org/10.1016/0893-6080(91)90012-T)
- **Yingwei, Sundararajan & Saratchandran (1997), MRAN.** RAN growth plus
  contribution-based pruning.
  [DOI](https://doi.org/10.1162/neco.1997.9.2.461)
- **Huang, Saratchandran & Sundararajan (2005), GGAP-RBF.** Sequential
  significance-based growth and pruning.
  [DOI](https://doi.org/10.1109/TNN.2004.836241)
- **Rusu et al. (2016), Progressive Neural Networks.** Frozen task columns with
  lateral transfer and state growth.
  [arXiv record](https://arxiv.org/abs/1606.04671)
- **Yoon et al. (2018), Dynamically Expandable Networks.** Gradient-trained
  selective retraining, expansion, and unit splitting.
  [OpenReview](https://openreview.net/forum?id=Sk7KsfW0-)

### Replay and online continual-learning baselines

- **Lopez-Paz & Ranzato (2017), GEM.** Episodic-memory gradient constraints and
  continual-transfer metrics.
  [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
- **Chaudhry et al. (2019), A-GEM.** Efficient average-gradient constraint and
  single-pass/disjoint-development protocol.
  [OpenReview](https://openreview.net/forum?id=Hkf2_sC5FX)
- **Chaudhry et al. (2019), Tiny Episodic Memories.** Demonstrates the strength
  of simple small-buffer replay in one-pass learning.
  [arXiv record](https://arxiv.org/abs/1902.10486)
- **Aljundi et al. (2019), Maximally Interfered Retrieval.** Selects replay
  examples expected to suffer most from the current update.
  [NeurIPS](https://papers.nips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
- **Buzzega et al. (2020), Dark Experience Replay.** General continual-learning
  replay with stored logits; includes DER++.
  [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

### Backpropagation alternatives and predictive representations

- **Lillicrap et al. (2016), Feedback Alignment.** Fixed random backward weights
  can transmit useful error signals.
  [Nature Communications](https://www.nature.com/articles/ncomms13276)
- **Nokland (2016), Direct Feedback Alignment.** Direct fixed random projection
  of output error to hidden layers.
  [NIPS](https://papers.nips.cc/paper_files/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html)
- **Lee et al. (2014/2015), Difference Target Propagation.** Learned inverse
  mappings provide hidden targets.
  [arXiv record](https://arxiv.org/abs/1412.7525)
- **Hinton (2022), Forward-Forward.** Positive and negative forward passes with
  local goodness objectives.
  [arXiv record](https://arxiv.org/abs/2212.13345)
- **Assran et al. (2023), I-JEPA.** Masked prediction in latent representation
  space using learned context and target encoders.
  [CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)
- **O'Shea & Rajendran (2026), Forward Projection.** Single-pass closed-form
  layer fitting with random input/label target projections and no retrograde
  communication.
  [Nature Communications](https://www.nature.com/articles/s41467-026-69161-1)

## 17. Audit outcome and next action

WP1 supports continuing the project, but it narrows the novelty claim. The
current method is not merely “RLS plus neurons”: that territory is occupied by
KRLS, RAN, RBF growth/pruning, and analytic continual learning. The paper must
name CPAM's **proposal-confirmation-maturity lifecycle** as the method and use
the RLS frontier as its central empirical test.

Before the final paper run, complete equation-level reading of KRLS, RAN, MRAN,
GGAP-RBF, ARTMAP, ACIL, GKEAL, RanPAC, and AnaCP. The immediate experimental
consequence is clear: add budgeted KRLS, RAN/MRAN, OS-ELM, and ARTMAP to the
analytic baseline plan. Without them, reviewers could reasonably attribute
CPAM's result to known sparse-kernel or resource-allocation behavior.

