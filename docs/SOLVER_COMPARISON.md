# OS-ELM output-solver comparison

## Research question

The experiment freezes one deterministic OS-ELM random feature map and changes
only the supervised output update. Except for an explicitly labelled
memory-matched smaller-feature control, all solvers receive the same feature
vectors, targets, event order, split, and update count. None uses
backpropagation, replay, task identity, or a forgetting factor.

For feature vector $h_t$ and one-hot target $y_t$, define cumulative sufficient
statistics

$$S_t = \lambda I + \sum_{i=1}^{t} h_i h_i^T, \qquad
  C_t = \sum_{i=1}^{t} h_i y_i^T.$$

Exact ridge weights are $W_t = C_t^T S_t^{-1}$.

## Solvers

- **LMS/NLMS:** $W_t = W_{t-1} + \eta (y_t-W_{t-1}h_t)h_t^T /
  (\epsilon+\lVert h_t\rVert^2)$. This is linear-state but does not retain
  cumulative sufficient statistics.
- **Exact cumulative RLS:** applies Sherman--Morrison to $S_t^{-1}$ with
  forgetting factor fixed to exactly one. It is algebraically equivalent to
  fitting ridge regression on every observation seen so far, up to floating
  point error.
- **Diagonal RLS:** retains only the diagonal of the inverse-covariance
  approximation after each rank-one update.
- **Block RLS:** retains within-block inverse-covariance terms and discards
  cross-block terms. Block size is declared in configuration.
- **Cumulative RPLS:** "RPLS" has multiple published meanings, including
  moving-window and exponentially weighted variants. Here it means a
  factor-free recursive sufficient-statistic implementation. It updates $S_t$
  and $C_t$ without raw data and solves each output in the supervised Krylov
  subspace $span(c,Sc,\ldots,S^{a-1}c)$. The component count $a$ is declared.
  This operationalization tests latent-rank regularization, not reduced state:
  its stored covariance remains quadratic.
- **Frequent-Directions ridge:** retains exact $C_t$ and a bounded
  deterministic sketch $B_t$ such that $B_t^T B_t \approx \sum h_i h_i^T$.
  It computes $(B_t^T B_t+\lambda I)^{-1}C_t$ with the Woodbury identity.
  The implementation stores a `2 * sketch_rank` workspace and compresses it
  to `sketch_rank` rows using singular-value shrinkage.
- **Memory-matched smaller exact RLS:** uses a declared prefix of the same
  frozen random coordinates plus the same bias coordinate. Because it changes
  representation width, it is a resource control and is never presented as a
  solver-only comparison.

## Evaluation contract

Hyperparameters are selected using development seeds only. Held-out seeds are
run once after selection. Every result stores online pre-update accuracy,
locked final accuracy, segment checkpoint accuracy, adaptation and retention,
backward transfer, sample-efficiency checkpoints, non-finite-state counts,
prediction/update latency, throughput, exact persistent array bytes, and
traced peak allocations. The latter is an implementation-level Python/NumPy
measurement rather than a claim about GPU memory.

Primary conclusions are separated into predictive quality, quality under a
persistent-memory budget, and quality under a per-event compute budget. A
single method need not win all three.

## References defining the chosen families

- Qin, S. J. (1998), *Recursive PLS algorithms for adaptive data modeling*,
  Computers & Chemical Engineering 22(4--5), 503--514.
- Ghashami, Liberty, Phillips, and Woodruff (2016), *Frequent Directions:
  Simple and Deterministic Matrix Sketching*, SIAM Journal on Computing 45(5).
- Shi and Phillips (2021), *A Deterministic Streaming Sketch for Ridge
  Regression*, AISTATS/PMLR 130.

The implementation does not claim to reproduce Qin's moving-window or
forgetting-factor algorithms. That distinction is intentional because this
study asks whether cumulative, factor-free state can replace exact RLS.
