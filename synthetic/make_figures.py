"""Draw Figure 3 and Figure 7: the mountain, before and after warping.

    "Left: synthetic experiment illustrating how WarpGrad warps gradients (see
     Appendix D for full details).  Each task f ~ p(f) defines a distinct loss
     surface (W, bottom row).  Gradient descent (black) on these surfaces
     struggles to find a minimum.  WarpGrad meta-learns a warp omega to produce
     better update directions (magenta; Section 2.4).  In doing so, WarpGrad
     learns a meta-geometry P where standard gradient descent is well behaved
     (top row)."
                                                  -- Figure 3 caption, page 4

    "Example trajectories on three task loss surfaces.  We start Gradient
     Descent (black) and WarpGrad (magenta) from the same initialisation; while
     SGD struggles with the curvature, the WarpGrad optimiser has learned a warp
     such that gradient descent in the representation space (top) leads to rapid
     convergence in model parameter space (bottom)."
                                           -- Figure 7 caption, pages 18-19


TWO THINGS THE PAPER LEAVES UNSAID, AND WHAT WE DID ABOUT THEM
==============================================================

(1) "FROM THE SAME INITIALISATION" IS NOT FREE.
Gradient descent starts at a point x0 in W-space, the space where the loss
actually lives.  WarpGrad starts at a point theta0 in P-space, and its position
in W-space is Omega(theta0).  These coincide only if Omega happens to be the
identity, which after meta-training it is not.  So "the same initialisation"
cannot mean "the same numbers in both coordinate systems", because then the two
optimisers would be starting from different points on the surface and the
comparison would be rigged in whichever direction Omega happened to shift things.

We therefore INVERT the warp: given x0, we solve

        theta0 = argmin_theta || Omega(theta) - x0 ||^2

numerically, so that Omega(theta0) = x0 and both optimisers genuinely begin at
the same point on the same surface.  Omega is not guaranteed invertible, so we
report the residual || Omega(theta0) - x0 || for every panel we draw.  A figure
whose inversion residual is large is a figure that is comparing two different
starting points, and the reader deserves to know which one they are looking at.
`--no-invert` reproduces the naive reading (theta0 = x0) for comparison.

(2) THE PAPER SELECTS ITS PANELS, AND SO DO WE, OUT LOUD.
Appendix D says the initialisation is "chosen such that standard gradient
descent struggles".  That is a legitimate thing to do when the figure's job is
to illustrate a mechanism, but it means the panels are not a random sample and
must not be read as an average-case result.  We select the same way, we print
the selection rule, and we additionally produce `fig_unselected.png` from
uniformly random (task, init) pairs so the typical case is on the record too.
The aggregate win rate over unselected pairs is the number that belongs in a
results table.  The mountain is an illustration, not evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # never render inline: plot output floods the MCP result JSON
import matplotlib.pyplot as plt
import numpy as np
import torch

from synthetic.mountain import TASK_LR, TASK_STEPS, sample_init, sample_task
from synthetic.train_mountain import plain_gd_trajectory, warp_trajectory
from warpgrad.geometry import ExplicitWarp

BLACK = "#111111"   # gradient descent, as in the paper
MAGENTA = "#E0218A"  # WarpGrad, as in the paper


def invert_warp(warp, x0, steps=600, lr=0.08):
    """Find theta0 with Omega(theta0) ~= x0, so both optimisers start together.

    Returns (theta0, residual).  The residual is reported, never hidden: it is
    the read-out error of this instrument.
    """
    theta = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((warp(theta) - x0) ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual = (warp(theta) - x0).norm(dim=-1)
    return theta.detach(), residual


def _surface(fn, lim, n=140):
    g = torch.linspace(-lim, lim, n)
    Y, X = torch.meshgrid(g, g, indexing="ij")
    with torch.no_grad():
        Z = fn(torch.stack([X, Y], dim=-1))
    return X.numpy(), Y.numpy(), Z.numpy()


def _surf3d(ax, X, Y, Z, title):
    ax.plot_surface(
        X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
        alpha=0.88, rstride=2, cstride=2,
    )
    ax.contour(X, Y, Z, zdir="z", offset=float(Z.min()), levels=14,
               cmap="viridis", alpha=0.45, linewidths=0.6)
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=46, azim=-62)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)


def _traj3d(ax, traj, fn, color, label, lift=0.06):
    """Overlay a trajectory on a 3-D surface, lifted slightly so it stays visible."""
    with torch.no_grad():
        z = fn(traj).numpy()
    t = traj.numpy()
    span = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
    ax.plot(t[:, 0], t[:, 1], z + lift * span, color=color, lw=2.0,
            label=label, zorder=10)
    ax.scatter(t[:1, 0], t[:1, 1], z[:1] + lift * span, color=color, s=34,
               marker="o", edgecolor="white", linewidth=0.8, zorder=11)
    ax.scatter(t[-1:, 0], t[-1:, 1], z[-1:] + lift * span, color=color, s=60,
               marker="*", edgecolor="white", linewidth=0.8, zorder=11)


def run_pair(task, warp, x0, invert=True, steps=TASK_STEPS, lr=TASK_LR):
    """Run GD and WarpGrad from the same W-space point.  Returns everything."""
    x0 = x0.reshape(1, 2)
    gd = plain_gd_trajectory(task, x0, steps, lr)[:, 0]              # (T+1, 2) in W

    if invert:
        theta0, residual = invert_warp(warp, x0)
    else:
        theta0, residual = x0.clone(), (warp(x0) - x0).norm(dim=-1).detach()
    tp, tw = warp_trajectory(task, warp, theta0, steps, lr)
    with torch.no_grad():
        out = dict(
            gd_w=gd, wg_p=tp[:, 0], wg_w=tw[:, 0],
            residual=float(residual.item()),
            gd_final=float(task(gd[-1:]).item()),
            wg_final=float(task(tw[-1:, 0]).item()),
            gd_best=float(task(gd).min().item()),
            wg_best=float(task(tw[:, 0]).min().item()),
        )
    return out


def figure(tasks, warp, x0s, path, invert=True, plim=3.0, wlim=3.0, title=""):
    """Two rows: P-space (top, what WarpGrad descends) and W-space (bottom)."""
    n = len(tasks)
    fig = plt.figure(figsize=(4.1 * n, 8.0))
    recs = []

    for i, (task, x0) in enumerate(zip(tasks, x0s)):
        r = run_pair(task, warp, x0, invert=invert)
        recs.append(r)

        # ---- top row: the meta-geometry P, where plain GD is well behaved ----
        axP = fig.add_subplot(2, n, i + 1, projection="3d")
        Xp, Yp, Zp = _surface(lambda t: task(warp(t)), plim)
        _surf3d(axP, Xp, Yp, Zp, f"P-space (warped)   task {i + 1}")
        _traj3d(axP, r["wg_p"], lambda t: task(warp(t)), MAGENTA, "WarpGrad")

        # ---- bottom row: the native mountain W, where the loss really lives --
        axW = fig.add_subplot(2, n, n + i + 1, projection="3d")
        Xw, Yw, Zw = _surface(task, wlim)
        _surf3d(axW, Xw, Yw, Zw, f"W-space (native)   task {i + 1}")
        _traj3d(axW, r["gd_w"], task, BLACK, "gradient descent")
        _traj3d(axW, r["wg_w"], task, MAGENTA, "WarpGrad")
        axW.text2D(
            0.02, -0.04,
            f"final loss   GD {r['gd_final']:+.2f}   Warp {r['wg_final']:+.2f}"
            f"\ninversion residual {r['residual']:.1e}",
            transform=axW.transAxes, fontsize=7.5, color="#444444",
        )

    h = [
        plt.Line2D([], [], color=BLACK, lw=2, label="Gradient descent"),
        plt.Line2D([], [], color=MAGENTA, lw=2, label="WarpGrad"),
        plt.Line2D([], [], color="#888", marker="o", ls="", label="start"),
        plt.Line2D([], [], color="#888", marker="*", ls="", ms=10, label="end"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(path, dpi=155)
    plt.close(fig)
    print(f"  wrote {path}")
    return recs


def pick_hard(warp, gen, n_panels, third_term, pool=48, invert=True):
    """Select panels the way Appendix D says it does, and say so.

    Rule, stated so it can be criticised: sample `pool` random (task, init)
    pairs, score each by (final loss of GD) - (final loss of WarpGrad), keep the
    top `n_panels`.  These are cases where gradient descent struggles most.
    """
    cands = []
    for _ in range(pool):
        task = sample_task(gen, third_term=third_term)
        x0 = sample_init(gen, 1)[0]
        r = run_pair(task, warp, x0, invert=invert)
        cands.append((r["gd_final"] - r["wg_final"], task, x0, r))
    cands.sort(key=lambda c: -c[0])
    return [(c[1], c[2]) for c in cands[:n_panels]], cands


def aggregate(warp, gen, n, third_term, invert=True):
    """Unselected, uniformly random pairs: the number that belongs in a table."""
    wins, deltas, residuals = 0, [], []
    for _ in range(n):
        task = sample_task(gen, third_term=third_term)
        x0 = sample_init(gen, 1)[0]
        r = run_pair(task, warp, x0, invert=invert)
        deltas.append(r["gd_final"] - r["wg_final"])
        residuals.append(r["residual"])
        wins += int(r["wg_final"] < r["gd_final"])
    d = np.array(deltas)
    return dict(
        n=n, win_rate=wins / n, mean_delta=float(d.mean()),
        median_delta=float(np.median(d)), std_delta=float(d.std()),
        sem_delta=float(d.std() / np.sqrt(n)),
        median_inversion_residual=float(np.median(residuals)),
        max_inversion_residual=float(np.max(residuals)),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True, help="path to meta-trained warp .pt")
    ap.add_argument("--warp-init", default="plain", choices=["plain", "residual"])
    ap.add_argument("--third-term", default="peaks", choices=["peaks", "paper"])
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-agg", type=int, default=300)
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    invert = not a.no_invert
    tag = os.path.basename(a.warp).replace("warp_", "").replace(".pt", "")

    warp = ExplicitWarp(dim=2, hidden=30, init=a.warp_init)
    warp.load_state_dict(torch.load(a.warp, map_location="cpu"))
    warp.eval()
    for p in warp.parameters():
        p.requires_grad_(False)

    gen = torch.Generator().manual_seed(a.seed)

    print("selecting hard panels (Appendix D: 'chosen such that SGD struggles')")
    hard, _ = pick_hard(warp, gen, 3, a.third_term, invert=invert)

    print("drawing Figure 3 (2 tasks, the paper's layout)")
    figure([t for t, _ in hard[:2]], warp, [x for _, x in hard[:2]],
           f"{a.outdir}/fig3_mountain_{tag}.png", invert=invert,
           title="Figure 3 reproduction: WarpGrad warps the loss surface")

    print("drawing Figure 7 (3 tasks)")
    figure([t for t, _ in hard], warp, [x for _, x in hard],
           f"{a.outdir}/fig7_trajectories_{tag}.png", invert=invert,
           title="Figure 7 reproduction: trajectories on three task loss surfaces")

    print("drawing unselected panels (uniformly random, the typical case)")
    g2 = torch.Generator().manual_seed(a.seed + 1000)
    rnd = [(sample_task(g2, third_term=a.third_term), sample_init(g2, 1)[0])
           for _ in range(3)]
    figure([t for t, _ in rnd], warp, [x for _, x in rnd],
           f"{a.outdir}/fig_unselected_{tag}.png", invert=invert,
           title="Unselected random tasks (not cherry-picked)")

    print(f"aggregating over {a.n_agg} random (task, init) pairs")
    agg = aggregate(warp, torch.Generator().manual_seed(a.seed + 99),
                    a.n_agg, a.third_term, invert=invert)
    agg["warp"] = a.warp
    agg["inverted_init"] = invert
    json.dump(agg, open(f"results/mountain/agg_{tag}.json", "w"), indent=2)
    print(json.dumps(agg, indent=2))
