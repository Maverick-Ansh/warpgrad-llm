"""Evaluate WarpGrad against gradient descent on the mountains, bracketed.

Every number this file prints is a NORMALISED score, defined in check_warp.py:

    score = (floor - achieved) / (floor - ceiling)

    1.0  found the global minimum of that surface
    0.0  no better than picking a random point in the domain
   <0.0  actively worse than picking a random point

Raw losses are not comparable across this task family, because each surface has
its own depth (b_i ~ Cat(-5,...,5)).  Averaging raw losses would let a handful of
deep surfaces dominate the mean and would tell you about the sampler rather than
about the optimiser.

THREE ARMS ARE ALWAYS REPORTED
------------------------------
  GD        plain gradient descent on the native surface, the paper's black line
  WarpGrad  plain gradient descent in P-space, the paper's magenta line
  FLEE      the degenerate shortcut: walk away from the origin until the loss
            decays to zero.  It optimises nothing.  If an arm cannot beat FLEE,
            it has not demonstrated anything, and because b_i can be negative
            FLEE is a real competitor on this family rather than a straw man.

A LEARNING-RATE CONTROL IS ALSO REPORTED
----------------------------------------
A warp whose metric G^-1 is close to a multiple of the identity is doing nothing
that a different learning rate could not do.  So we also run gradient descent at
the effective step size that the warp implies, GD-tuned, chosen per run from a
small sweep.  If WarpGrad only matches GD-tuned, then the claim "WarpGrad learns
a better geometry" is not supported by this experiment, only the weaker claim
"WarpGrad learns a better step size" is.  This control is ours, not the paper's,
and it is the sharpest test available on a 2-D problem.

Every pair is GATED on the inversion residual.  Pairs where Omega cannot reach
x0 are excluded and counted, never silently averaged in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from synthetic.check_warp import TOL, bracket, invert_warp, normalised
from synthetic.mountain import TASK_LR, TASK_STEPS, sample_init, sample_task
from synthetic.train_mountain import plain_gd_trajectory, warp_trajectory
from warpgrad.geometry import ExplicitWarp


def gd_tuned(task, x0, lrs=(0.01, 0.03, 0.1, 0.3, 1.0), steps=TASK_STEPS):
    """Best plain GD over a small learning-rate sweep: the control arm.

    This is deliberately generous to the baseline.  It gets to pick its best
    learning rate per task with full knowledge of the outcome, which no honest
    optimiser could do.  If WarpGrad still wins against that, the win is real.
    """
    best = float("inf")
    for lr in lrs:
        traj = plain_gd_trajectory(task, x0.reshape(1, 2), steps, lr)
        with torch.no_grad():
            best = min(best, float(task(traj[-1, 0].reshape(1, 2))))
    return best


def evaluate(warp, n_pairs, third_term, seed, tol=TOL, steps=TASK_STEPS, lr=TASK_LR):
    gen = torch.Generator().manual_seed(seed)
    rows, skipped = [], 0

    for _ in range(n_pairs):
        task = sample_task(gen, third_term=third_term)
        x0 = sample_init(gen, 1)

        theta0, res = invert_warp(warp, x0, steps=1500, restarts=3)
        if float(res.max()) > tol:
            skipped += 1
            continue

        br = bracket(task)
        if abs(br["floor"] - br["ceiling"]) < 1e-6:
            skipped += 1
            continue

        gd = plain_gd_trajectory(task, x0, steps, lr)
        _, tw = warp_trajectory(task, warp, theta0, steps, lr)
        with torch.no_grad():
            gd_f = float(task(gd[-1, 0].reshape(1, 2)))
            wg_f = float(task(tw[-1, 0].reshape(1, 2)))
        tuned_f = gd_tuned(task, x0, steps=steps)

        rows.append(dict(
            gd=normalised(gd_f, br),
            warp=normalised(wg_f, br),
            gd_tuned=normalised(tuned_f, br),
            flee=normalised(br["flee_to_infinity"], br),
            residual=float(res.max()),
        ))

    return rows, skipped


def summarise(rows, skipped, n_pairs):
    def stat(key):
        v = np.array([r[key] for r in rows])
        return dict(mean=float(v.mean()), sem=float(v.std() / np.sqrt(len(v))),
                    median=float(np.median(v)))

    warp = np.array([r["warp"] for r in rows])
    gd = np.array([r["gd"] for r in rows])
    tuned = np.array([r["gd_tuned"] for r in rows])
    flee = np.array([r["flee"] for r in rows])
    d = warp - gd
    d_tuned = warp - tuned

    return dict(
        n_requested=n_pairs, n_used=len(rows), n_skipped_unreachable=skipped,
        gd=stat("gd"), warp=stat("warp"), gd_tuned=stat("gd_tuned"),
        flee=stat("flee"),
        warp_beats_gd=float((warp > gd).mean()),
        warp_beats_gd_tuned=float((warp > tuned).mean()),
        warp_beats_flee=float((warp > flee).mean()),
        gd_beats_flee=float((gd > flee).mean()),
        paired_delta_vs_gd=dict(mean=float(d.mean()),
                                sem=float(d.std() / np.sqrt(len(d)))),
        paired_delta_vs_gd_tuned=dict(mean=float(d_tuned.mean()),
                                      sem=float(d_tuned.std() / np.sqrt(len(d_tuned)))),
        median_residual=float(np.median([r["residual"] for r in rows])),
    )


def stratify(rows, n_bins=4):
    """Split results by how badly PLAIN GRADIENT DESCENT did, then compare.

    This is the analysis that reconciles the figure with the average.

    Figure 3 and Figure 7 are not a random sample.  Appendix D says the
    initialisation is "chosen such that standard gradient descent struggles",
    and the figure's whole job is to show what warping does in exactly that
    situation.  So the right question is not "does WarpGrad beat gradient
    descent on average", it is:

        does WarpGrad beat gradient descent WHERE GRADIENT DESCENT STRUGGLES?

    Those are different questions and they can have different answers.  If
    WarpGrad wins on the cases where gradient descent does badly and loses
    slightly everywhere else, then both the figure and a small average effect
    are true at the same time, and neither is misleading.

    We bin by the GD score itself (quartiles, lowest first) and report the
    paired difference within each bin.  Binning on the baseline's own score and
    not on the difference is what keeps this from being a way to manufacture a
    result: the bin assignment does not look at WarpGrad at all.
    """
    gd = np.array([r["gd"] for r in rows])
    wg = np.array([r["warp"] for r in rows])
    tuned = np.array([r["gd_tuned"] for r in rows])
    order = np.argsort(gd)
    bins = np.array_split(order, n_bins)

    out = []
    for i, idx in enumerate(bins):
        d = wg[idx] - gd[idx]
        out.append(dict(
            bin=i, n=len(idx),
            gd_range=[float(gd[idx].min()), float(gd[idx].max())],
            gd_mean=float(gd[idx].mean()), warp_mean=float(wg[idx].mean()),
            tuned_mean=float(tuned[idx].mean()),
            delta_mean=float(d.mean()),
            delta_sem=float(d.std() / np.sqrt(max(len(d), 1))),
            win_rate=float((wg[idx] > gd[idx]).mean()),
        ))
    return out


def report_strata(strata):
    print("-" * 70)
    print("  STRATIFIED BY HOW BADLY PLAIN GRADIENT DESCENT DID")
    print("  (quartiles of the GD score itself, lowest first.  The figure in the")
    print("   paper is drawn from the leftmost bin by construction.)")
    print(f"  {'bin':<5}{'n':>4}{'GD score':>20}{'WarpGrad':>10}"
          f"{'delta':>10}{'sem':>8}{'win':>7}")
    for s in strata:
        rng = f"[{s['gd_range'][0]:+.2f},{s['gd_range'][1]:+.2f}]"
        label = "worst" if s["bin"] == 0 else ("best" if s["bin"] == len(strata) - 1
                                               else f"q{s['bin'] + 1}")
        print(f"  {label:<5}{s['n']:>4}{rng:>20}{s['warp_mean']:>10.4f}"
              f"{s['delta_mean']:>+10.4f}{s['delta_sem']:>8.4f}"
              f"{s['win_rate']:>7.0%}")


def report(s):
    print("=" * 70)
    print(f"  pairs used {s['n_used']}/{s['n_requested']}"
          f"   skipped as unreachable: {s['n_skipped_unreachable']}")
    print(f"  median inversion residual {s['median_residual']:.2e}")
    print("-" * 70)
    print(f"  {'arm':<12}{'mean score':>13}{'sem':>9}{'median':>10}")
    for k, label in [("warp", "WarpGrad"), ("gd", "GD"),
                     ("gd_tuned", "GD-tuned"), ("flee", "FLEE")]:
        st = s[k]
        print(f"  {label:<12}{st['mean']:>13.4f}{st['sem']:>9.4f}{st['median']:>10.4f}")
    print("-" * 70)
    d, dt = s["paired_delta_vs_gd"], s["paired_delta_vs_gd_tuned"]
    print(f"  WarpGrad - GD        {d['mean']:+.4f} +/- {d['sem']:.4f}"
          f"   ({d['mean'] / max(d['sem'], 1e-12):+.1f} sem)"
          f"   win rate {s['warp_beats_gd']:.1%}")
    print(f"  WarpGrad - GD-tuned  {dt['mean']:+.4f} +/- {dt['sem']:.4f}"
          f"   ({dt['mean'] / max(dt['sem'], 1e-12):+.1f} sem)"
          f"   win rate {s['warp_beats_gd_tuned']:.1%}")
    print(f"  beats FLEE:  WarpGrad {s['warp_beats_flee']:.1%}"
          f"   GD {s['gd_beats_flee']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True)
    ap.add_argument("--warp-init", default="residual", choices=["plain", "residual"])
    ap.add_argument("--third-term", default="peaks", choices=["peaks", "paper"])
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    warp = ExplicitWarp(dim=2, hidden=30, init=a.warp_init)
    warp.load_state_dict(torch.load(a.warp, map_location="cpu"))
    warp.eval()
    for p in warp.parameters():
        p.requires_grad_(False)

    rows, skipped = evaluate(warp, a.n_pairs, a.third_term, a.seed)
    if not rows:
        print("NO USABLE PAIRS.  Every initialisation was unreachable.")
        sys.exit(1)
    s = summarise(rows, skipped, a.n_pairs)
    s["warp_path"] = a.warp
    s["strata"] = stratify(rows)
    report(s)
    report_strata(s["strata"])

    out = a.out or f"results/mountain/eval_{os.path.basename(a.warp)[5:-3]}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(s, open(out, "w"), indent=2)
    print(f"wrote {out}")
