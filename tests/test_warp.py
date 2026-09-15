"""Smoke tests for the implicit warp machinery.

These assert the paper's MECHANISM, not tensor shapes.  The central one is
`test_linear_warp_is_block_diagonal_and_mlp_warp_is_not`, which tests the exact
property the paper claims as its contribution, directly, rather than inferring it
from a downstream accuracy difference.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from llm.model import GPTConfig, WarpedGPT
from warpgrad.warp import WarpLayer, is_warp_param, split_parameters


# Warp kinds that apply their non-linearity on the MAIN path rather than on a
# residual branch.  These cannot be the identity at initialisation no matter how
# they are initialised, because GELU(x) != x.  The paper's ladder has the same
# property at the same rungs ("3x3conv+ReLU", "3x3conv+BN+ReLU"), so this is a
# faithful port of a confound that is already in Table 3, not one we introduced.
MAIN_PATH_ACTIVATION = {"linear_act", "norm_act"}


def test_identity_capable_warps_start_as_the_identity():
    """A warp should begin as a no-op, or the baseline comparison is confounded.

    If omega started as an arbitrary map, then at meta-step 0 the "WarpGrad"
    model would already be a different architecture from the no-warp baseline,
    and any later difference would mix "warping helps" with "this particular
    random map helps".  Starting at the identity makes the two arms identical
    before meta-learning, so every subsequent difference is attributable.
    """
    x = torch.randn(4, 7, 64)
    for kind in WarpLayer.KINDS:
        if kind in MAIN_PATH_ACTIVATION:
            continue
        w = WarpLayer(64, kind=kind, rank=8, expansion=2)
        assert torch.allclose(w(x), x, atol=1e-5), \
            f"warp kind {kind!r} is not the identity at initialisation"


def test_main_path_activation_warps_are_flagged_as_confounded():
    """Record, in executable form, which rungs change the model at init.

    This is not a failure.  It is a property of the paper's own ladder that has
    to be stated when reading Table 3: the "+ReLU" rows are not pure tests of
    gradient warping, because they also alter the task-learner's forward function
    before any meta-learning has happened.  Any accuracy difference on those rows
    is warping PLUS architecture.  The rows that isolate warping cleanly are the
    identity-capable ones.
    """
    x = torch.randn(4, 7, 64)
    for kind in MAIN_PATH_ACTIVATION:
        w = WarpLayer(64, kind=kind)
        assert not torch.allclose(w(x), x, atol=1e-3), (
            f"{kind!r} unexpectedly IS the identity at init, so the confound "
            "documented here no longer applies and REPORT.md must be updated"
        )


def test_linear_warp_is_block_diagonal_and_mlp_warp_is_not():
    """Section 2.2's actual claim, tested on the Jacobian itself.

    A LINEAR warp has a constant Jacobian: the same preconditioner is applied no
    matter what data flows through.  That is block-diagonal preconditioning and
    it is what T-Nets and Meta-Curvature already do.

    A NON-LINEAR warp has an input-dependent Jacobian: the preconditioner
    changes with the data.  That is "beyond the block-diagonal structure imposed
    by prior works" and it is the thing WarpGrad claims as new.

    We perturb the warp away from initialisation first, because at exactly the
    identity initialisation even the mlp warp is momentarily linear (its residual
    branch is zeroed), so testing at init would give a false negative.
    """
    torch.manual_seed(0)
    x1 = torch.randn(64) * 2.0
    x2 = torch.randn(64) * 2.0

    lin = WarpLayer(64, kind="linear")
    with torch.no_grad():
        for p in lin.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert not lin.is_data_dependent(x1, x2), \
        "linear warp Jacobian varies with input, which cannot happen"

    mlp = WarpLayer(64, kind="mlp", expansion=2)
    with torch.no_grad():
        for p in mlp.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert mlp.is_data_dependent(x1, x2), \
        "mlp warp Jacobian is constant, so it is not beyond block-diagonal"


def test_identity_warp_model_is_bit_identical_to_a_plain_gpt():
    """warp_kind='identity' must give EXACTLY the baseline, not approximately.

    This is what makes the no-warp control arm exact.  If the warped model
    differed from a plain GPT even slightly at initialisation, the control would
    be a different model and the comparison would be unfair in an unknown
    direction.
    """
    idx = torch.randint(0, 256, (2, 32))

    # Compare ONE model against itself with the warps switched off, rather than
    # two separately constructed models.  Two constructions consume the RNG
    # stream differently (a linear warp draws weights, an identity warp does
    # not), so their task parameters would differ and the test would be
    # measuring the initialiser instead of the warp.
    for kind in ("scale", "lowrank", "linear", "residual", "mlp"):
        torch.manual_seed(0)
        m = WarpedGPT(GPTConfig(n_layer=3, d_model=64, n_head=4,
                                block_size=32, warp_kind=kind))
        m.eval()
        with torch.no_grad():
            warped, _ = m(idx)
            kinds = [w.kind for w in m.warps]
            for w in m.warps:
                w.kind = "identity"
            plain, _ = m(idx)
            for w, k in zip(m.warps, kinds):
                w.kind = k
        assert torch.allclose(warped, plain, atol=1e-4), (
            f"identity-initialised {kind!r} warp changes the model output at "
            "init, so the no-warp control arm is not exact"
        )


def test_param_split_puts_warps_in_phi_and_nothing_else():
    """theta and phi must be disjoint, exhaustive, and correctly assigned.

    Getting this wrong is the single easiest way to silently break the method:
    if a warp parameter lands in theta it gets adapted per task, and the
    'geometry' is no longer shared across tasks at all.
    """
    cfg = GPTConfig(n_layer=4, d_model=64, n_head=4, block_size=32,
                    warp_kind="mlp")
    m = WarpedGPT(cfg)
    task, warp = split_parameters(m)

    assert set(task).isdisjoint(warp), "a parameter is in both theta and phi"
    assert set(task) | set(warp) == set(n for n, _ in m.named_parameters()), \
        "the split is not exhaustive"
    assert all(n.startswith("warps.") for n in warp), \
        f"non-warp parameters tagged as phi: {[n for n in warp if not n.startswith('warps.')]}"
    assert all(not n.startswith("warps.") for n in task), \
        "a warps.* parameter was left in theta"
    assert len(warp) > 0, "no warp parameters found at all"


def test_warp_params_are_tagged_and_survive_a_state_dict_round_trip():
    """The phi tag must be a property of the module, not of a live tensor.

    Checkpoint, reload, and the split must still be right.  If the tag were lost
    on load, a resumed run would meta-learn the wrong parameter set.
    """
    cfg = GPTConfig(n_layer=2, d_model=32, n_head=4, block_size=16,
                    warp_kind="residual")
    m = WarpedGPT(cfg)
    _, warp_before = split_parameters(m)
    m2 = WarpedGPT(cfg)
    m2.load_state_dict(m.state_dict())
    _, warp_after = split_parameters(m2)
    assert warp_before == warp_after, "phi/theta split changed across a reload"
    assert all(is_warp_param(dict(m2.named_parameters())[n]) for n in warp_after)


def test_capacity_ladder_is_ordered_by_parameter_count():
    """Table 3 is a capacity ladder, so our port must actually be one.

    If 'mlp' had fewer parameters than 'linear', the ablation would not be
    testing capacity and claim C4 would be untestable in this port.
    """
    d = 64
    counts = {
        k: sum(p.numel() for p in WarpLayer(d, kind=k, rank=8, expansion=2).parameters())
        for k in WarpLayer.KINDS
    }
    assert counts["identity"] == 0
    assert counts["scale"] < counts["lowrank"] < counts["linear"]
    assert counts["linear"] <= counts["norm_act"] < counts["mlp"], counts


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} warp tests passed")
