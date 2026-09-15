# Reproducing WarpGrad, and porting it to language models

Reproduction of **Meta-Learning with Warped Gradient Descent**, Flennerhag,
Rusu, Pascanu, Visin, Yin and Hadsell, ICLR 2020 ([arXiv:1909.00025v2](https://arxiv.org/abs/1909.00025)).

Hardware: 2x Tesla T4 (compute capability 7.5, so fp16 and never bf16) plus a
laptop CPU for the two-dimensional work. Code written from the paper text only.
The authors' reference implementation was deliberately not read.

> **Status.** This document is filled in as runs finish. Sections marked
> RUNNING are not yet complete and contain no numbers. No verdict is recorded
> before its measurement has passed its gate.

---

## 1. The claims, stated so they can be falsified

| # | Claim | Where | What would confirm it | Kind |
|---|-------|-------|----------------------|------|
| **C1** | Warp-layers improve adaptation to held-out tasks, against the same task-learner with no warp | Tables 1, 2 | Warp-Leap 83.8 vs Leap 74.8 vs Reptile 70.8 on Omniglot | headline |
| **C2** | The method works at 100+ adaptation steps, where backpropagating through training cannot | Section 4.2 | MAML stalls at 56.7 while Warp-Leap reaches 83.8 | headline |
| **C3** | Warping turns a loss surface on which gradient descent struggles into one on which it does not | Figures 3, 7, Appendix D | trajectories reach minima from starting points where plain descent fails | mechanism |
| **C4** | More warp capacity is better, and going beyond block-diagonal preconditioning helps | Tables 3, 4 | 88.0 (2-layer) > 84.4 (linear) > 74.8 (none); and 81.3 > 68.0 > 40.1 | mechanism |
| **C5** | The learned geometry is not the Fisher information matrix | Figure 9, Appendix G | post-warp activations centre near zero but their covariance stays far from the identity | mechanism |
| **C6** | Offline meta-training beats online meta-training | Appendix F | 84.4 vs 76.3 | mechanism |
| **C7** | The first-order objective (Eq. 12) costs little against the exact one (Eq. 11) | Appendix F | 83.1 vs 84.4, inside one standard deviation | mechanism |

C1 and C2 are the abstract's claims. C3 is the picture. C4 is the part the
authors argue is genuinely new, because prior work (T-Nets, Meta-Curvature) was
already doing block-diagonal preconditioning.

---

## 2. How the reproduction was resized

The paper's benchmarks are miniImageNet, tieredImageNet, Omniglot, a maze
navigation reinforcement learning task, and continual sine regression. Running
those on two T4s would be a slow re-run of somebody else's experiment.

The question that set the scope was: **what is the cheapest setting in which each
claim is still falsifiable?** That gave three tiers.

### Tier 0. The mountain (Appendix D), reproduced exactly

Appendix D is already tiny. It is run verbatim, with no resizing at all: the same
task family, the same 100 adaptation steps at learning rate 0.1, the same 100
meta-steps with 10 initialisations each, the same two-layer warp with 30 hidden
units and tanh. This tier tests C3 and it costs about 30 seconds.

We then *added* a quantitative evaluation that the paper does not perform, because
Appendix D is presented as intuition-building rather than as evidence. See
section 6.1 for why that distinction matters to the verdict.

### Tier 1 and 2. Language modelling

The paper's multi-shot benchmark is Omniglot under the protocol of Flennerhag
et al. (2019): 50 alphabets, each alphabet a separate 20-way classification task,
whole alphabets held out at meta-test time, 100 adaptation steps per task.

The structure that carries the claim is: **many tasks, each the same kind of
problem over a different symbol system, with whole symbol systems held out.**

The language-modelling analogue is exact. Each **language** is a task, each task
is next-byte prediction, and whole languages, including whole **scripts**, are
held out. A meta-learner that has never seen one byte of Georgian must adapt to
Georgian in 100 gradient steps using only the geometry it learned elsewhere.

| | Paper (Omniglot) | This reproduction |
|---|---|---|
| task | one alphabet, 20-way classification | one language, next-byte prediction |
| task-learner | 4 conv blocks, 64 filters | 6-layer byte-level GPT, d_model 256 |
| warp placement | after each conv block | after each transformer block |
| adaptation | 100 SGD steps | 100 SGD steps |
| meta-train tasks | 25 alphabets | 25 languages |
| held-out tasks | 10 alphabets | 7 languages across 7 scripts |
| what is held out | whole alphabets | whole languages, and 3 whole scripts |

The held-out split is deliberately mixed so the report can separate two different
questions that would otherwise be confounded:

- **uk, ca, fa, mr** are new languages in a script that *was* present during
  meta-training (Cyrillic via ru and bg, Latin, Arabic via ar, Devanagari via hi).
- **ka (Georgian), ta (Tamil), ko (Hangul)** are new languages in a writing
  system that appears **nowhere** in meta-training.

If warping transfers only within a script, those two groups will separate. That
is a question the Omniglot setup cannot ask, because all its alphabets are
rendered the same way.

### What was legitimate to swap, and what was not

The rule followed: you may swap what the paper freezes or holds fixed across
arms. You may not swap the thing under test.

**Swapped.** The task-learner architecture (convolutional to transformer), the
data (images to bytes), the normaliser inside warp-layers (BatchNorm to
LayerNorm), the number of meta-training steps.

**Not swapped.** The warp-layer idea, the meta-objective (Eq. 11 and Eq. 12), the
training algorithms (Algorithms 1 and 2), the inner loop (plain SGD, 100 steps),
the rule that warp parameters are frozen during adaptation, and the rule that the
task loss and the meta loss are evaluated on different batches.

### Deviations table

| Item | Paper | Here | Why it is still a test of the claim |
|---|---|---|---|
| Warp normaliser | BatchNorm | LayerNorm | Batch statistics over a language-modelling batch are a different and worse object than over an image batch, and every transformer in practice uses LayerNorm. Applied identically to every arm. |
| Meta-objective default | Eq. 11 (exact) | Eq. 12 (first-order) | Double backward in fp16 on a T4 is slow and fragile. The paper measures the cost at 1.3 points, inside one standard deviation (Table 3). C7 checks this rather than assuming it. |
| Replay buffer | all 2000 iterates | every 5th iterate | Eq. 11 is an expectation over p(theta given tau) and the iterates are Monte-Carlo samples of it, so subsampling costs variance and not bias. At 5M parameters, storing all iterates would need about 40 GB. |
| Precision | not stated | fp16 with GradScaler | Compute capability 7.5 has fast fp16 and no bf16. |
| Tokenisation | n/a | raw UTF-8 bytes, vocab 256 | A BPE tokenizer trained on the meta-training languages would leak information about them into the held-out languages. Omniglot shares no such preprocessing across alphabets, so bytes are the faithful choice. |
| Meta learning rate, Tier 0 | **not stated** | swept, see 6.1 | Appendix D omits it. It turns out to change the sign of the result, which is why it is swept rather than picked. |
| Warp init | **not stated** | identity, or near it | See section 5.2. This is the single most consequential unstated choice in the paper. |

---

## 3. What the paper leaves ambiguous, and what we did

### 3.1 The Appendix D formula has a typo

Appendix D prints the third term of the task family as

```
- b3 exp( -(x1 + a3)^2 - x1^2 )
```

with `x1` in both slots. The family is otherwise transparently a randomised
MATLAB `peaks` function, and `peaks` has

```
- b3 exp( -(x1 + a3)^2 - x2^2 )
```

This is not cosmetic. With `x1` in both slots the third term does not depend on
`x2` at all, so instead of a localised bump it becomes an infinite ridge running
the length of the `x2` axis, which changes the shape of every surface in the
family.

**We implement both** behind `--third-term {peaks,paper}` and default to `peaks`,
because the surfaces drawn in the paper's own Figures 3 and 7 show localised
bumps and not ridges. `tests/test_mountain.py` asserts that the family reduces to
MATLAB `peaks` exactly at `s=5, a=(1,1,1), b=(3,10,1/3)`, which pins the
transcription of every other term, and separately asserts that the two readings
really do differ so the choice cannot be quietly ignored.

### 3.2 Warp initialisation is unstated and it decides the experiment

Appendix D says the warp is "a 2-layer feed-forward network with a hidden-state
size of 30 and tanh non-linearities". It does not say how it is initialised.

A freshly initialised two-layer tanh network is an arbitrary squashing map. Its
hidden activations live in a bounded box, so its **image is a bounded, curved
blob** in the plane. There is no reason that blob should contain the square
`[-3,3]^2` that Appendix D draws starting points from. When it does not, a
starting point is simply unreachable, and WarpGrad cannot be started there at all.

We measured it. With the literal reading, **the warp can reach only 25.4 percent
of the initialisation domain**. Section 5.1 explains why that turned a broken
measurement into a confident wrong answer.

We therefore also implement a residual parameterisation, `Omega(theta) = theta +
g(theta)` with `g`'s output layer zeroed, so the warp begins as exactly the
identity and WarpGrad begins as exactly gradient descent. Coverage is then 100
percent and, as a free correctness check, the measured advantage at meta-step 0
must be exactly 0.0000. It is.

The same reasoning is applied to every warp-layer in the language tier: each kind
is initialised to the identity map so that the no-warp control arm is exact
rather than approximate.

---

## 4. Verifying the paper's mathematics directly

Before measuring anything, two of the paper's claims are checkable numerically.

### Eq. 9, the first-order equivalence

Section 2.3 argues that a plain gradient step taken in the warped space P is
equivalent, to first order in the step size, to the ideal preconditioned step in
the native space W:

```
(L . Omega)(theta - alpha * Dtheta) = L(gamma - alpha * Dgamma) + O(alpha^2)
```

This is load-bearing. It is the reason you are allowed to descend in P at all. If
the error were only first order, the whole construction would be unjustified.

`tests/test_mountain.py::test_first_order_equivalence_error_scales_as_alpha_squared`
halves alpha six times and checks that the measured gap between the two sides
falls by a factor of about four each time. Quadratic scaling means O(alpha^2).
Linear scaling would have meant the paper's justification does not hold.
**It scales quadratically.** Eq. 9 holds numerically.

### G inverse is a valid Riemann metric

Section 2.3 defines `G^-1 = [Dx Omega][Dx Omega]^T` and argues it is
positive-definite whenever `Dx Omega` has full rank, which is what gives WarpGrad
gradient descent's convergence guarantees. Constructed as a matrix times its own
transpose it is symmetric and positive-semi-definite by algebra, and the test
confirms this holds numerically at 64 random points.

### The linear-versus-non-linear distinction, tested on the Jacobian itself

The paper's stated contribution over T-Nets and Meta-Curvature is that a
non-linear warp gives preconditioning that depends on the data, and so escapes
the block-diagonal structure of those methods. That is a statement about the
Jacobian, so it can be tested directly rather than inferred from accuracy.

`tests/test_warp.py::test_linear_warp_is_block_diagonal_and_mlp_warp_is_not`
computes `Dx omega` at two different inputs and asserts the linear warp gives the
same matrix both times while the two-layer warp does not. **It does.** The
mechanism is present as described.

---

## 5. What broke

This is the section that cost the most time and is worth the most.

### 5.1 The synthetic experiment confidently reported the wrong sign

The first complete run of Tier 0 produced this, over 200 randomly sampled tasks:

```
win_rate            0.17
mean_delta         -0.678
median_delta       -0.308
```

Read at face value, that says WarpGrad loses badly to plain gradient descent. It
is wrong, and nothing in the number itself says so.

The tell was in a field printed alongside it:

```
median_inversion_residual   0.620
max_inversion_residual      2.074
```

Here is the problem. Gradient descent starts at a point `x0` on the loss surface.
WarpGrad starts at a point `theta0` in its own coordinate system, which puts it at
`Omega(theta0)` on the surface. For the comparison to mean anything these must be
the same point, so we have to solve `Omega(theta0) = x0`. The residual is how
badly that solve failed. On a domain of `[-3,3]`, a typical miss of 0.62 means
**the two optimisers were usually starting from different places on the
mountain.** Whichever one happened to get dropped nearer a valley won, and the
result measured the warp's range rather than its quality.

The cause is section 3.2: the literal warp reaches only a quarter of the domain.

**The figure showed it too, which is why figures are worth drawing.** In the
first version of Figure 7, the warped surface in the top row occupied a small
patch of the plane and the trajectory floated visibly off it. In the one panel
where the inversion had succeeded, with residual 9.5e-07, WarpGrad genuinely won
(final loss -2.00 against gradient descent's +0.00). The mechanism was working
wherever it was measurable, and unmeasurable everywhere else.

**Fix.** `synthetic/check_warp.py` now measures coverage before anything else and
refuses the sweep below 80 percent:

```
plain     coverage  25.4%   VERDICT: REFUSE
residual  coverage 100.0%   VERDICT: PROCEED
```

Every comparison is additionally gated per pair, and excluded pairs are counted
in the output rather than dropped silently.

**The lesson.** The reported quantity was a ratio whose denominator was quietly
undefined for most of the sample. Reporting the read-out's own residual next to
the result is what made it visible. A metric that decodes something back into
another space should always report its own decoding error.

### 5.2 The degenerate shortcut: running away wins

While building the gate we computed the score of a policy that does no
optimisation at all, and simply walks away from the origin until the loss decays
to zero. Every term in Appendix D's family is a polynomial multiplied by a
Gaussian, so every surface flattens to zero far from the origin.

```
flee-to-infinity   mean +0.090   median +0.032   beats random guessing 59.4%
```

Because `b_i ~ Cat(-5,...,5)` can be negative, which flips pits into bumps, a
majority of the surfaces in this family are ones where **fleeing beats guessing**.
Any optimiser that simply diverges banks a positive score. This is reported next
to every result so nobody mistakes it for learning, and it is why the normalised
scale is anchored to a real reference rather than to raw loss.

### 5.3 A one-line initialisation bug that would have invalidated the whole ladder

`WarpedGPT.__init__` called `self.apply(self._init)` after building its modules.
That helper rewrites every `nn.Linear` to `normal(0, 0.02)`. Warp-layers are
built from `nn.Linear`, so the call was **silently overwriting every warp-layer's
identity initialisation**: the `proj.weight = I` of the linear warp, and the
zeroed residual branches of the `residual` and `mlp` warps.

The consequence would have been that no warp arm started as the baseline, so the
entire capacity ladder for C4 would have been comparing randomly initialised
extra layers of differing sizes, not geometries. Accuracies would still have come
out, ordered by something, and the ordering would have looked like a result.

Caught by `test_identity_warp_model_is_bit_identical_to_a_plain_gpt`, which is in
the repository precisely because "the control arm is exact" is an assumption worth
making executable. Fixed by constructing warp-layers after the general
initialiser has run.

### 5.4 Bits per byte is not comparable across writing systems

UTF-8 spends one byte per character on Latin, two on Cyrillic, Greek, Hebrew and
Arabic, and three on Devanagari, Georgian, Tamil and CJK. In a three-byte script
two of every three bytes are continuation bytes whose value is nearly determined
by the byte before them.

Measured on the actual corpus, the smoothed-bigram floor per script:

| script | bytes/char | bigram floor (bits/byte) |
|---|---|---|
| Georgian | 3 | **1.670** |
| Tamil | 3 | 1.842 |
| Devanagari | 3 | 2.018 |
| Hebrew | 2 | 2.380 |
| Arabic | 2 | 2.386 |
| Greek | 2 | 2.502 |
| Cyrillic | 2 | 2.553 |
| Hangul | 3 | 3.433 |
| Japanese | 3 | 3.519 |
| Latin | 1 | 3.566 |

A model can post 2.0 bits/byte on Georgian while understanding no Georgian, purely
by learning UTF-8. Averaging raw bits per byte over the held-out languages would
have ranked methods mostly by **which scripts happened to land in the split**,
which is a property of our own split and not of any method.

So every language is bracketed against its own floor and its own ceiling, both
measured on that language's own data, and the reported score is

```
score = (bigram_floor - achieved) / (bigram_floor - oracle_ceiling)
```

The zero point is the bigram floor rather than the weaker unigram floor, because
the bigram table is achievable without learning anything transferable, and
transfer is the claim under test.

### 5.5 A key collision that overwrote a statistic with a file path

`summarise()` returned a dictionary whose `"warp"` key held the WarpGrad
statistics, and the caller then wrote `s["warp"] = args.warp` to record which
checkpoint produced them. The statistics were replaced by a string and the
printer crashed on the next line.

Trivial, and worth recording for one reason: it crashed. The neighbouring bugs in
5.1 and 5.3 did not crash, they returned plausible numbers. The cheap bugs
announce themselves and the expensive ones do not.

---

## 6. Results

### 6.1 Tier 0, the mountain

RUNNING. Figures are generated and the gate has been passed with the residual
parameterisation. The meta learning rate sweep, which Appendix D does not
specify, is in progress and is required before any verdict is recorded, because
early runs show the sign of the effect depends on it.

### 6.2 Tier 1 and 2, language modelling

RUNNING. The ceiling measurement (`llm/oracle.py`) must pass before the sweep is
launched.

---

## 7. What was not tested

Listed exhaustively, because a reproduction that does not say what it skipped is
not reporting, it is advertising.

- **miniImageNet and tieredImageNet few-shot (Table 1).** Not attempted. The
  Warp-MAML arm, which needs backpropagation through the adaptation process, is
  not implemented at all, so nothing here speaks to the few-shot claims.
- **The reinforcement learning maze (Section 4.3, Figure 4 right).** Not
  attempted. Warp-RNN, the HyperNetwork-style recurrent warp of Eq. 20, and
  Algorithm 3 (continual meta-training) are not implemented.
- **Continual learning sine regression (Section 4.3, Figures 5, 12, 13).** Not
  attempted. Eq. 19 and Eq. 23 are not implemented.
- **C5, the Natural Gradient comparison (Appendix G, Figure 9).** Planned but not
  yet run. Nothing is claimed about whether the learned geometry is Fisher-like.
- **KFAC as a baseline.** Not implemented, so the Table 4 comparison against a
  second-order method has no counterpart here.
- **Learned inner learning rate** (the "learned alpha" row of Table 3). Not run.
- **FiLM task embeddings** (the "TA" row of Table 3). Not run.
- **Absolute numbers.** Nothing in this report is comparable to the paper's
  accuracies. The task, the data, the architecture and the scale are all
  different by construction. Only the *direction and relative size* of effects
  transfer, and only those are claimed.
