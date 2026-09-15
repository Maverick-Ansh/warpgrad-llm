"""
The mountain: Appendix D's synthetic family of 2-D loss surfaces.

This is the substrate for Figure 3 and Figure 7 of Flennerhag et al. (ICLR 2020),
the figures that show what "warping a loss surface" actually looks like.

THE PAPER'S DEFINITION, VERBATIM
--------------------------------
    "A learner is faced with the task of minimising an objective function of the
     form f_tau(x1,x2) = g1(x1) exp(g2(x2)) - g3(x1) exp(g4(x1,x2))
     - g5 exp(g6(x1)), where each task f_tau is defined by scale and rotation
     functions g that are randomly sampled from a predefined distribution.
     Specifically, each task is defined by the objective function

        f_tau(x1,x2) =  b1 (a1 - x1)^2 exp(-x1^2 - (x2 + a2)^2)
                      - b2 (x1/s - x1^3 - x2^5) exp(-x1^2 - x2^2)
                      - b3 exp(-(x1 + a3)^2 - x1^2)

     where each a, b and s are randomly sampled parameters from

        s    ~ Cat(1, 2, ..., 9, 10)
        a_i  ~ Cat(-1, 0, 1)
        b_i  ~ Cat(-5, -4, ..., 4, 5).

     The task is to minimise the given objective from a randomly sampled
     initialisation, x_{i=1,2} ~ U(-3, 3).  During meta-training, we train on a
     task for 100 steps using a learning rate of 0.1."
                                                        -- Appendix D, page 18

WHAT THIS FAMILY *IS*
---------------------
It is a randomised MATLAB `peaks` function.  `peaks` is the standard three-bump
test surface

    z = 3(1-x)^2 exp(-x^2 - (y+1)^2)
        - 10(x/5 - x^3 - y^5) exp(-x^2 - y^2)
        - (1/3) exp(-(x+1)^2 - y^2)

and the paper's family is exactly that with (s, a1, a2, a3) and (b1, b2, b3)
resampled per task.  Setting s=5, a=(1,1,1), b=(3,10,1/3) must reproduce `peaks`
bit for bit.  `tests/test_mountain.py` asserts precisely this, which is a much
stronger check than "the tensor has the right shape".

THE DISCREPANCY WE HAD TO RESOLVE
---------------------------------
The third term as printed in the paper is

        - b3 exp(-(x1 + a3)^2 - x1^2)          <- note: x1 twice

but `peaks` has

        - b3 exp(-(x1 + a3)^2 - x2^2)          <- note: x2 in the second slot

These are not cosmetically different.  With x1 in both slots the third term does
not depend on x2 at all, so instead of a localised bump it becomes an infinite
ridge running the whole length of the x2 axis.  That changes the shape of every
surface in the family.

We implement both and expose `third_term`:

    third_term="peaks"  -> exp(-(x1+a3)^2 - x2^2)   the `peaks`-faithful reading
    third_term="paper"  -> exp(-(x1+a3)^2 - x1^2)   the literal printed reading

We default to "peaks" and justify it in REPORT.md: the surfaces drawn in the
paper's own Figure 3 and Figure 7 show localised bumps, not ridges, so the
printed x1 is a typo.  Both are reproduced side by side so a reader can check
that judgement instead of taking our word for it.

Every task is BOUNDED.  Each of the three terms is a polynomial multiplied by a
Gaussian, so every term decays to 0 as |x| grows.  A minimum therefore always
exists and no run can diverge to -infinity.  This is an invariant of the
environment, and `tests/test_mountain.py` asserts it, because a physics bug
quietly becomes "a hard dataset" and would confound every downstream number.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The three categorical supports of Appendix D.
S_SUPPORT = torch.arange(1, 11, dtype=torch.float32)       # 1 .. 10
A_SUPPORT = torch.tensor([-1.0, 0.0, 1.0])                 # -1, 0, 1
B_SUPPORT = torch.arange(-5, 6, dtype=torch.float32)       # -5 .. 5
INIT_RANGE = (-3.0, 3.0)                                   # x ~ U(-3, 3)

TASK_STEPS = 100        # "we train on a task for 100 steps"
TASK_LR = 0.1           # "using a learning rate of 0.1"


@dataclass(frozen=True)
class MountainTask:
    """One sampled loss surface f_tau.  Frozen so a task cannot mutate mid-run."""

    s: float
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    third_term: str = "peaks"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate f_tau.  `x` is (..., 2), returns (...)."""
        x1, x2 = x[..., 0], x[..., 1]
        a1, a2, a3 = self.a
        b1, b2, b3 = self.b

        t1 = b1 * (a1 - x1) ** 2 * torch.exp(-x1**2 - (x2 + a2) ** 2)
        t2 = b2 * (x1 / self.s - x1**3 - x2**5) * torch.exp(-(x1**2) - x2**2)
        # the one ambiguous character in the whole appendix
        second = x1**2 if self.third_term == "paper" else x2**2
        t3 = b3 * torch.exp(-((x1 + a3) ** 2) - second)

        return t1 - t2 - t3

    def grid(self, n: int = 200, lim: float = 3.0):
        """Dense evaluation for surface plots.  Returns (X, Y, Z) as (n, n)."""
        g = torch.linspace(-lim, lim, n)
        Y, X = torch.meshgrid(g, g, indexing="ij")
        return X, Y, self(torch.stack([X, Y], dim=-1))


def sample_task(gen: torch.Generator, third_term: str = "peaks") -> MountainTask:
    """Draw f_tau ~ p(f) exactly as Appendix D specifies."""
    def cat(support: torch.Tensor, k: int = 1):
        idx = torch.randint(len(support), (k,), generator=gen)
        return support[idx]

    s = cat(S_SUPPORT).item()
    a = tuple(cat(A_SUPPORT, 3).tolist())
    b = tuple(cat(B_SUPPORT, 3).tolist())
    return MountainTask(s=s, a=a, b=b, third_term=third_term)


def sample_init(gen: torch.Generator, n: int = 1) -> torch.Tensor:
    """x_{i=1,2} ~ U(-3, 3), shape (n, 2)."""
    lo, hi = INIT_RANGE
    return torch.rand(n, 2, generator=gen) * (hi - lo) + lo


def matlab_peaks(x: torch.Tensor) -> torch.Tensor:
    """Reference `peaks`, used only by the test that pins the family down."""
    x1, x2 = x[..., 0], x[..., 1]
    return (
        3 * (1 - x1) ** 2 * torch.exp(-x1**2 - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * torch.exp(-(x1**2) - x2**2)
        - (1 / 3) * torch.exp(-((x1 + 1) ** 2) - x2**2)
    )
