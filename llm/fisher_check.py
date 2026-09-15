"""C5: is the learned geometry the Fisher information matrix, or something else?

Reproduces the analysis of Appendix G and Figure 9 on a transformer.

THE QUESTION
============
Natural Gradient Descent preconditions by the inverse Fisher information matrix.
It is intractable to compute, so methods like KFAC and Natural Neural Nets
approximate it block-diagonally.  WarpGrad also preconditions.  So: is WarpGrad
just rediscovering the Fisher matrix by a different route?

The paper says no, and gives a sharp test.

THE TEST, FROM THE PAPER
========================
    "Because warp-layers are linear in this configuration, if the learned
     geometry is approximately Fisher, post-warp activations should be
     zero-centred and the layer-wise covariance matrix should satisfy
     Cov(omega(i)(h(i)(x)), omega(i)(h(i)(x))) = I, where I is the identity
     matrix (Desjardins et al., 2015).  If true, Warp-Leap would learn a
     block-diagonal approximation to the Inverse Fisher Matrix, as Natural
     Neural Nets."
                                                        -- Appendix G, page 22

So there are two separate things to check, and the paper finds they come apart:

    "we find that, in general, WarpGrad-Leap has zero-centered post-warp
     activations. ... However, we find that the correlation structure is
     significantly different from what we would expect if Warp-Leap were to
     represent the Fisher matrix; post-warp covariances are significantly
     dissimilar from the identity matrix and varies across layers."
                                                        -- Appendix G, page 23

That is the finding under test: **centring yes, whitening no.**  Half of the
Natural-Neural-Nets signature appears and half does not, which is evidence that
the learned geometry is not the Fisher matrix.

WHAT WE MEASURE
===============
At every warp-layer, on held-out data, during adaptation:

  MEAN        E[h] before the warp and E[omega(h)] after it.  "Zero-centred"
              means the post-warp value sits near 0.

  SHATTEN-1   ||Cov(h,h) - I||_S1, the sum of the singular values of the
              difference between the activation covariance and the identity.
              Zero means perfectly white (decorrelated and unit variance),
              which is what a Fisher-like preconditioner would produce.

WHY THE TRANSFORMER PORT NEEDS ONE EXTRA CONTROL
================================================
A transformer block ends in LayerNorm-based residual arithmetic, so activations
entering a warp-layer are ALREADY roughly centred, for reasons that have nothing
to do with meta-learning.  On the paper's convolutional net the pre-warp
activations were positive because of ReLU, which made "post-warp is centred" an
informative observation.  Here it would be nearly free.

So we add a control the paper did not need: the SAME measurement on an
**untrained warp**, initialised to the identity.  Any Fisher-like signature must
show up as a difference from that control, not merely as a small absolute
number.  Without this control the test would look passed no matter what.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from llm.data import META_TEST, ByteSampler, build_corpus
from llm.meta import functional_loss
from llm.model import GPTConfig, WarpedGPT


def shatten1_minus_identity(X: torch.Tensor) -> float:
    """||Cov(X) - I||_S1 where X is (n_samples, d).

    Shatten-1 is the nuclear norm: the sum of singular values.  It is zero only
    when the covariance IS the identity, so it is a single number for "how far
    from white is this".
    """
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    n = max(X.shape[0] - 1, 1)
    C = (X.T @ X) / n
    D = C - torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return float(torch.linalg.svdvals(D).sum())


@torch.no_grad()
def collect(model, theta, phi, buffers, sampler, n_batches=8, batch_size=8,
            max_vectors=4096):
    """Run forward, tapping the residual stream before and after every warp.

    We re-implement the forward pass rather than using hooks, because the whole
    point is to read the value on both sides of each warp-layer, and doing that
    explicitly is clearer than registering eight hooks and hoping the ordering
    is what we think it is.
    """
    merged = {**theta, **phi, **buffers}
    L = model.cfg.n_layer
    pre = [[] for _ in range(L)]
    post = [[] for _ in range(L)]

    for bi, (x, _y) in enumerate(sampler.sequential_batches(batch_size, n_batches)):
        if bi >= n_batches:
            break
        T = x.shape[1]
        posn = torch.arange(T, device=x.device)
        h = (torch.nn.functional.embedding(x, merged["wte.weight"])
             + torch.nn.functional.embedding(posn, merged["wpe.weight"]))
        for i, (blk, wrp) in enumerate(zip(model.blocks, model.warps)):
            h = blk(h)
            pre[i].append(h.reshape(-1, h.shape[-1]).float().cpu())
            h = wrp(h)
            post[i].append(h.reshape(-1, h.shape[-1]).float().cpu())

    out = []
    for i in range(L):
        a = torch.cat(pre[i])[:max_vectors]
        b = torch.cat(post[i])[:max_vectors]
        out.append(dict(
            layer=i + 1,
            mean_pre=float(a.mean()), mean_post=float(b.mean()),
            absmean_pre=float(a.mean(0).abs().mean()),
            absmean_post=float(b.mean(0).abs().mean()),
            s1_pre=shatten1_minus_identity(a) / a.shape[1],
            s1_post=shatten1_minus_identity(b) / b.shape[1],
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a .pt written by llm/train.py")
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/llm/fisher_check.json")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = WarpedGPT(cfg).to(a.device).eval()
    buffers = {k: v.to(a.device) for k, v in model.named_buffers()}
    theta = {k: v.to(a.device) for k, v in ck["theta0"].items()}
    phi_trained = {k: v.to(a.device) for k, v in ck["phi"].items()}
    # CONTROL: the untrained warp, which is the identity by construction
    phi_init = {k: v.detach().clone().to(a.device)
                for k, v in WarpedGPT(cfg).named_parameters()
                if k in phi_trained}

    corpus, _ = build_corpus(a.data_dir, verbose=False)
    langs = [l for l in (a.langs or META_TEST) if l in corpus]

    rows = {"trained": [], "control_identity_warp": []}
    for lang in langs:
        va = ByteSampler(corpus[lang].val, cfg.block_size, a.device, seed=21)
        rows["trained"].append(collect(model, theta, phi_trained, buffers, va,
                                       a.n_batches, a.batch_size))
        rows["control_identity_warp"].append(
            collect(model, theta, phi_init, buffers, va, a.n_batches, a.batch_size))

    def agg(key, field):
        M = np.array([[l[field] for l in run] for run in rows[key]])
        return M.mean(axis=0)

    print("=" * 78)
    print("C5  Is the learned geometry Fisher-like?")
    print("    Fisher-like would mean: post-warp activations centred at 0 AND")
    print("    post-warp covariance equal to the identity (Shatten-1 near 0).")
    print("=" * 78)
    print(f"{'layer':<7}{'|mean| pre':>12}{'|mean| post':>13}"
          f"{'S1 pre':>10}{'S1 post':>10}{'S1 post (ctrl)':>16}")
    print("-" * 78)
    ap_, aP = agg("trained", "absmean_pre"), agg("trained", "absmean_post")
    sp, sP = agg("trained", "s1_pre"), agg("trained", "s1_post")
    cP = agg("control_identity_warp", "s1_post")
    for i in range(len(ap_)):
        print(f"{i + 1:<7}{ap_[i]:>12.4f}{aP[i]:>13.4f}"
              f"{sp[i]:>10.4f}{sP[i]:>10.4f}{cP[i]:>16.4f}")
    print("-" * 78)
    print(f"{'mean':<7}{ap_.mean():>12.4f}{aP.mean():>13.4f}"
          f"{sp.mean():>10.4f}{sP.mean():>10.4f}{cP.mean():>16.4f}")
    print("=" * 78)

    centred = bool(aP.mean() < ap_.mean())
    whitened = bool(sP.mean() < 0.25)
    moved = bool(abs(sP.mean() - cP.mean()) > 0.02)
    print(f"  post-warp more centred than pre-warp : {centred}")
    print(f"  post-warp covariance close to I      : {whitened}"
          f"   (Shatten-1 per dim = {sP.mean():.3f})")
    print(f"  differs from the identity-warp control: {moved}")
    print()
    if centred and not whitened:
        print("  MATCHES the paper: centring yes, whitening no.  The learned")
        print("  geometry is NOT the Fisher information matrix.")
    elif centred and whitened:
        print("  DOES NOT match the paper: both signatures present, which would")
        print("  be consistent with a Fisher-like geometry.")
    else:
        print("  Neither signature is clean.  Reported as inconclusive, not as")
        print("  a refutation: absence of a signature in this port is weaker")
        print("  evidence than the paper's, because a transformer residual")
        print("  stream is already roughly centred before any warp is applied.")
        print("  That is what the control column is for.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(langs=langs, rows=rows,
                   verdict=dict(centred=centred, whitened=whitened, moved=moved)),
              open(a.out, "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(1, len(ap_) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(x, ap_, "o-", label="pre-warp")
        axes[0].plot(x, aP, "s-", label="post-warp")
        axes[0].axhline(0, color="#333", lw=0.8)
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("mean |E[h]| per unit")
        axes[0].set_title("Is it centred?"); axes[0].legend(fontsize=8)
        axes[1].plot(x, sp, "o-", label="pre-warp")
        axes[1].plot(x, sP, "s-", label="post-warp")
        axes[1].plot(x, cP, "^--", color="#999", label="post-warp, untrained warp")
        axes[1].axhline(0, color="#333", lw=0.8)
        axes[1].set_xlabel("layer")
        axes[1].set_ylabel("Shatten-1 of Cov - I, per dim")
        axes[1].set_title("Is it whitened?  (0 would mean Fisher-like)")
        axes[1].legend(fontsize=8)
        fig.suptitle("C5: Figure 9 reproduction on a transformer", fontsize=11)
        fig.tight_layout()
        os.makedirs(a.figdir, exist_ok=True)
        fig.savefig(f"{a.figdir}/llm_fisher_check.png", dpi=150)
        plt.close(fig)
        print(f"wrote {a.figdir}/llm_fisher_check.png")
    except Exception as e:
        print(f"figure skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
