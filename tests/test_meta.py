"""Smoke tests for the meta-learners.  CPU only, seconds to run.

The headline one is `test_leap_gradient_is_not_the_reptile_direction`.  An
earlier version of llm/meta.py computed the Reptile direction and called it
Leap.  Nothing crashed, both arms trained, and the results table would have
contained two baselines that were secretly the same algorithm.  This file exists
so that cannot happen again.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from llm.data import ByteSampler
from llm.meta import MetaLearner
from llm.model import GPTConfig, WarpedGPT

CFG = dict(n_layer=2, d_model=64, n_head=4, block_size=32, dropout=0.0)


def _sampler(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    # structured bytes, not uniform noise: a model must be able to LEARN
    # something, or every gradient is noise and the tests measure nothing
    txt = ("the quick brown fox jumps over the lazy dog. " * 500).encode()
    data = np.frombuffer(txt, dtype=np.uint8)[:n].copy()
    s = ByteSampler(data, CFG["block_size"], device="cpu", seed=seed)
    s.lang = "test"
    return s


def _learner(arm="warp_leap", kind="linear", **kw):
    torch.manual_seed(0)
    m = WarpedGPT(GPTConfig(warp_kind=kind, **CFG))
    return MetaLearner(m, arm=arm, inner_lr=0.05, inner_steps=8,
                       device="cpu", amp_dtype=None, **kw)


def test_leap_gradient_is_not_the_reptile_direction():
    """Eq. 17 must differ from theta_K - theta_0, or one baseline is a duplicate.

    Reptile moves theta0 toward wherever training ended up.  Leap shortens the
    trajectory's LENGTH in the joint space of parameters and loss, so a step that
    bought a big loss drop for a small parameter move is weighted differently
    from one that did not.  They point in genuinely different directions.

    We compare cosine similarity.  Identical directions would give 1.0.
    """
    L = _learner("leap", "identity")
    s = _sampler()
    traj, leap_g = L.adapt(s, batch_size=4, steps=8, collect=False, track_leap=True)

    leap_vec = torch.cat([leap_g[n].reshape(-1) for n in L.theta0])
    reptile_vec = torch.cat([
        (traj.final[n] - L.theta0[n].detach()).reshape(-1) for n in L.theta0])

    cos = torch.nn.functional.cosine_similarity(
        leap_vec.unsqueeze(0), reptile_vec.unsqueeze(0)).item()
    assert leap_vec.norm() > 0, "Leap gradient is all zeros"
    assert abs(cos) < 0.999, (
        f"Leap gradient is collinear with the Reptile direction (cos={cos:.6f}). "
        "The two baselines would be the same algorithm."
    )
    assert traj.leap_length is not None and traj.leap_length > 0, \
        "trajectory length is not positive"


def test_warp_parameters_do_not_move_during_adaptation():
    """Appendix E: phi is FIXED during task adaptation.

    This is not an optimisation detail, it is the definition of the method.  phi
    is the geometry.  If it moved during adaptation, the geometry would be
    shifting underneath the optimiser and 'warped gradient descent' would mean
    nothing.
    """
    L = _learner("warp_leap", "mlp")
    before = {n: p.detach().clone() for n, p in L.phi.items()}
    theta_before = {n: p.detach().clone() for n, p in L.theta0.items()}
    L.adapt(_sampler(), batch_size=4, steps=6, collect=False)

    for n, p in L.phi.items():
        assert torch.equal(p.detach(), before[n]), f"warp param {n} moved during adaptation"
    # and theta0 itself must not have been mutated either: adaptation works on a copy
    for n, p in L.theta0.items():
        assert torch.equal(p.detach(), theta_before[n]), \
            f"theta0 {n} was mutated by adaptation instead of being copied"


def test_exact_and_approx_objectives_give_different_phi_gradients():
    """Eq. 11 keeps the second-order term, Eq. 12 drops it.  They must differ.

    If they produced the same gradient, `--exact` would be a no-op and claim C7
    ('the approximation costs little') would be untestable rather than confirmed.
    """
    grads = {}
    for exact in (False, True):
        L = _learner("warp_leap", "linear", exact=exact)
        s = _sampler()
        traj, _ = L.adapt(s, batch_size=4, steps=4, stride=1, collect=True)
        for p in L.phi.values():
            p.grad = None
        L.warp_meta_grad(traj.points[0], _sampler(seed=1), batch_size=4)
        grads[exact] = torch.cat([L.phi[n].grad.reshape(-1) for n in L.phi])

    assert torch.isfinite(grads[True]).all(), "exact objective produced non-finite grads"
    assert torch.isfinite(grads[False]).all(), "approx objective produced non-finite grads"
    rel = (grads[True] - grads[False]).norm() / grads[False].norm().clamp_min(1e-12)
    assert rel > 1e-4, (
        f"Eq. 11 and Eq. 12 gave effectively identical phi gradients (rel diff "
        f"{rel:.2e}).  The second-order term is not being taken."
    )


def test_no_warp_arms_refuse_to_run_with_a_real_warp():
    """A control arm that secretly has a geometry is not a control."""
    torch.manual_seed(0)
    m = WarpedGPT(GPTConfig(warp_kind="mlp", **CFG))
    for arm in ("leap", "reptile", "joint"):
        try:
            MetaLearner(m, arm=arm, device="cpu", amp_dtype=None)
        except AssertionError:
            continue
        raise AssertionError(f"arm={arm!r} accepted a non-identity warp")


def test_adaptation_actually_reduces_the_task_loss():
    """Sanity: the inner loop must learn something on learnable data.

    If adaptation did nothing, every arm would score the same and the whole
    comparison would be measuring noise. This is the cheapest possible guard
    against that.
    """
    L = _learner("warp_leap", "linear")
    traj, _ = L.adapt(_sampler(), batch_size=8, steps=30, collect=False)
    first, last = np.mean(traj.losses[:5]), np.mean(traj.losses[-5:])
    assert last < first, f"adaptation did not reduce loss: {first:.4f} -> {last:.4f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} meta tests passed")
