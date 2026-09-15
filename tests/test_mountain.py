"""Smoke tests that pin down the paper's rules, not just tensor shapes.

Run: python -m pytest tests/ -q      (or: python tests/test_mountain.py)

The point of each test is written out, because a test you cannot explain is a
test you cannot trust.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from synthetic.mountain import (
    MountainTask, matlab_peaks, sample_task, sample_init,
    S_SUPPORT, A_SUPPORT, B_SUPPORT, TASK_STEPS, TASK_LR,
)
from warpgrad.geometry import ExplicitWarp


def test_family_is_randomised_peaks():
    """Appendix D's family must collapse to MATLAB `peaks` at the right params.

    This is the strongest available check that we transcribed the formula
    correctly.  If a sign, an exponent or a term ordering were wrong, this
    would fail even though every shape stayed right.
    """
    task = MountainTask(s=5.0, a=(1.0, 1.0, 1.0), b=(3.0, 10.0, 1 / 3),
                        third_term="peaks")
    x = torch.rand(500, 2) * 6 - 3
    assert torch.allclose(task(x), matlab_peaks(x), atol=1e-5), \
        "family does not reduce to MATLAB peaks"


def test_paper_reading_differs_from_peaks_reading():
    """The Appendix D typo is material, so we must not silently pick one.

    If these two agreed, the ambiguity would not matter and we could stop
    talking about it.  They do not agree, so REPORT.md owes the reader a
    justification for the choice.
    """
    kw = dict(s=5.0, a=(1.0, 1.0, 1.0), b=(3.0, 10.0, 1 / 3))
    x = torch.rand(500, 2) * 6 - 3
    peaks = MountainTask(**kw, third_term="peaks")(x)
    paper = MountainTask(**kw, third_term="paper")(x)
    assert not torch.allclose(peaks, paper, atol=1e-3), \
        "the two readings agree, which contradicts the analysis in the docstring"


def test_every_surface_is_bounded():
    """Environment invariant: no task can run away to -infinity.

    Every term is polynomial times Gaussian, so f -> 0 far from the origin.
    A surface that was unbounded below would let an optimiser "win" by
    diverging, which would quietly turn into a fake result downstream.
    """
    gen = torch.Generator().manual_seed(0)
    far = torch.tensor([[8.0, 8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, -8.0]])
    for _ in range(200):
        task = sample_task(gen)
        assert task(far).abs().max() < 1e-6, f"{task} does not decay at infinity"
        # and it is finite everywhere on the region of interest
        _, _, Z = task.grid(n=64)
        assert torch.isfinite(Z).all(), f"{task} has non-finite values on the grid"


def test_sampling_respects_appendix_d_supports():
    """The task distribution p(f) must be the one the paper specifies."""
    gen = torch.Generator().manual_seed(1)
    for _ in range(500):
        t = sample_task(gen)
        assert t.s in S_SUPPORT.tolist(), f"s={t.s} off support"
        assert all(v in A_SUPPORT.tolist() for v in t.a), f"a={t.a} off support"
        assert all(v in B_SUPPORT.tolist() for v in t.b), f"b={t.b} off support"
    x = sample_init(gen, 4096)
    assert x.min() >= -3.0 and x.max() <= 3.0, "init outside U(-3,3)"


def test_metric_is_a_valid_riemann_metric():
    """Section 2.3: G^-1 = [DxOmega][DxOmega]^T must be symmetric and PSD.

    This is the paper's entire argument for why WarpGrad keeps gradient
    descent's convergence guarantees.  A preconditioner that was not PSD could
    point the update uphill.
    """
    warp = ExplicitWarp(dim=2, hidden=30, init="plain")
    theta = torch.rand(64, 2) * 6 - 3
    G_inv = warp.metric_inverse(theta)
    assert torch.allclose(G_inv, G_inv.transpose(-1, -2), atol=1e-5), "not symmetric"
    ev = torch.linalg.eigvalsh(G_inv)
    assert (ev >= -1e-6).all(), f"not PSD, min eigenvalue {ev.min().item()}"


def test_residual_init_starts_as_exact_identity():
    """With init='residual', WarpGrad must start as plain gradient descent.

    Omega = identity means DxOmega = I means G^-1 = I means the update is
    exactly the unpreconditioned one.  This is what makes any later improvement
    attributable to meta-learning rather than to a lucky initial warp.
    """
    warp = ExplicitWarp(dim=2, hidden=30, init="residual")
    theta = torch.rand(32, 2) * 6 - 3
    assert torch.allclose(warp(theta), theta, atol=1e-6), "Omega != identity at init"
    I = torch.eye(2).expand(32, 2, 2)
    assert torch.allclose(warp.jacobian(theta), I, atol=1e-5), "DxOmega != I at init"
    assert torch.allclose(warp.metric_inverse(theta), I, atol=1e-5), "G^-1 != I at init"


def test_first_order_equivalence_error_scales_as_alpha_squared():
    """Eq. 9 is the load-bearing claim of Section 2.3.  Test it numerically.

    Eq. 9 says
        (L . Omega)(theta - alpha*Dtheta) = L(gamma - alpha*Dgamma) + O(alpha^2)

    so the gap between a step taken in P-space and the ideal step taken in
    W-space must shrink quadratically in the step size.  We halve alpha five
    times and check the measured gap drops by roughly 4x each time.  If the
    gap shrank only linearly, the paper's justification for descending in
    P-space would not hold and every result built on it would be suspect.
    """
    torch.manual_seed(0)
    warp = ExplicitWarp(dim=2, hidden=30, init="plain")
    task = MountainTask(s=5.0, a=(1.0, 1.0, 1.0), b=(3.0, 10.0, 1 / 3))

    theta = (torch.rand(256, 2) * 4 - 2).requires_grad_(True)
    gamma = warp(theta)
    L = task(gamma).sum()
    # Eq. 7: Dtheta = [DxOmega]^T grad L(gamma)
    d_theta, = torch.autograd.grad(L, theta)
    # Eq. 8: Dgamma = DxOmega Dtheta
    J = warp.jacobian(theta.detach())
    d_gamma = torch.einsum("bij,bj->bi", J, d_theta)

    ratios = []
    prev = None
    for k in range(6):
        alpha = 0.1 * (0.5 ** k)
        lhs = task(warp(theta.detach() - alpha * d_theta))     # (L . Omega) step in P
        rhs = task(gamma.detach() - alpha * d_gamma)           # ideal step in W
        gap = (lhs - rhs).abs().mean().item()
        if prev is not None:
            ratios.append(prev / max(gap, 1e-30))
        prev = gap

    # O(alpha^2) => halving alpha divides the gap by ~4.  Allow a wide band,
    # we only need to distinguish quadratic from linear (which would give ~2).
    for r in ratios[1:]:
        assert 2.8 < r < 5.5, f"gap ratio {r:.2f} is not consistent with O(alpha^2)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} smoke tests passed")
