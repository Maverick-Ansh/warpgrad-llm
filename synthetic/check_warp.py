"""GATE: measure the instrument before trusting any comparison it produces.

This file exists because the first run of this reproduction produced a headline
number (WarpGrad win rate 0.17, i.e. losing badly to plain gradient descent)
that turned out to be an artefact of the measurement, not a property of the
method.  Phase 4 of the reproduction protocol says: build the measurement, then
gate on it, before spending GPU hours.  This is that gate.

WHAT CAN GO WRONG HERE
======================

(A) COVERAGE.  Gradient descent starts at a point x0 in W-space.  WarpGrad
    starts at theta0 in P-space and is therefore standing at Omega(theta0).
    For the comparison to mean anything we need Omega(theta0) = x0, i.e. we need
    x0 to be IN THE IMAGE of Omega.

    Omega is a 2-layer tanh network.  Its hidden activation is bounded in
    (-1,1)^30, so its image is the image of a bounded set under one linear map:
    a bounded, generally small, curved blob in R^2.  There is no reason that blob
    should contain [-3,3]^2, which is where Appendix D says initialisations come
    from.  When it does not, x0 is unreachable, WarpGrad silently starts
    somewhere else, and every loss comparison at that x0 is meaningless.

    We measure coverage directly: the fraction of x0 ~ U(-3,3)^2 for which the
    inversion residual ||Omega(theta0) - x0|| falls under a tolerance.  If
    coverage is low, the sweep is refused.

(B) THE BRACKET.  A loss number is uninterpretable on its own, because every
    task in this family has a different scale (b_i ~ Cat(-5,...,5), so one
    surface can be 10x deeper than another).  Before reading "WarpGrad got
    -2.0", you need to know:

        CEILING (best possible): min of f over the domain, found by dense grid
                                 search plus local polish.  No optimiser can beat
                                 this, so it is the top of the scale.
        FLOOR   (no information): the expected loss of a random point in the
                                  domain.  An optimiser that achieved this
                                  learned nothing.

    We then report every optimiser as a NORMALISED SCORE in [0,1]:

        score = (floor - achieved) / (floor - ceiling)

    1.0 means "found the global minimum", 0.0 means "no better than a random
    guess", and negative means "actively worse than guessing".  This is
    comparable across tasks, which raw loss is not.

(C) THE DEGENERATE SHORTCUT.  Is there a policy that scores well without
    optimising anything?  Here it is "stand still": f is bounded and decays to 0
    far from the origin, so an optimiser that runs away to infinity scores
    exactly 0 loss.  On surfaces whose minimum is positive (b can be negative,
    which flips bumps into pits and pits into bumps), fleeing to infinity BEATS
    honest descent.  We compute the flee-to-infinity score explicitly and report
    it, so nobody mistakes it for learning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from synthetic.mountain import TASK_LR, TASK_STEPS, sample_init, sample_task
from warpgrad.geometry import ExplicitWarp

TOL = 1e-2  # a starting point counts as reachable if residual < 1% of a unit


def invert_warp(warp, x0, steps=3000, lr=0.05, restarts=4):
    """Solve theta0 = argmin ||Omega(theta) - x0||^2 with multiple restarts.

    Multiple restarts matter: Omega is non-convex, so a single descent can land
    in a bad local basin and report a large residual for a point that is in fact
    reachable.  We must not blame the warp for our own solver being weak, so we
    give the solver a fair chance before declaring a point unreachable.
    """
    x0 = x0.reshape(-1, 2)
    best_theta = None
    best_res = torch.full((x0.shape[0],), float("inf"))
    g = torch.Generator().manual_seed(0)
    for r in range(restarts):
        init = x0.clone() if r == 0 else (torch.rand(x0.shape, generator=g) * 6 - 3)
        theta = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            ((warp(theta) - x0) ** 2).sum().backward()
            opt.step()
        with torch.no_grad():
            res = (warp(theta) - x0).norm(dim=-1)
            if best_theta is None:
                best_theta = theta.detach().clone()
            better = res < best_res
            best_theta[better] = theta.detach()[better]
            best_res[better] = res[better]
    return best_theta, best_res


def coverage(warp, n=512, tol=TOL, seed=0):
    """Fraction of the Appendix D initialisation domain that Omega can reach."""
    gen = torch.Generator().manual_seed(seed)
    x0 = sample_init(gen, n)
    _, res = invert_warp(warp, x0, steps=1500, restarts=3)
    res = res.numpy()
    return dict(
        n=n,
        coverage=float((res < tol).mean()),
        median_residual=float(np.median(res)),
        p90_residual=float(np.percentile(res, 90)),
        max_residual=float(res.max()),
        tol=tol,
    )


def bracket(task, lim=3.0, n=400):
    """Ceiling (global min) and floor (random guess) for one surface."""
    g = torch.linspace(-lim, lim, n)
    Y, X = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        Z = task(pts)
    ceiling = float(Z.min())

    # polish the grid argmin with a few hundred plain GD steps
    x = pts[int(Z.argmin())].clone().reshape(1, 2).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.01)
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        task(x).sum().backward()
        opt.step()
    with torch.no_grad():
        ceiling = min(ceiling, float(task(x)))

    floor = float(Z.mean())                 # expected loss of a random point
    flee = float(task(torch.tensor([[50.0, 50.0]])))  # the shortcut: run away
    return dict(ceiling=ceiling, floor=floor, flee_to_infinity=flee)


def normalised(achieved, br):
    """1.0 = global minimum, 0.0 = random guess, <0 = worse than guessing."""
    span = br["floor"] - br["ceiling"]
    if abs(span) < 1e-9:
        return float("nan")                 # degenerate flat surface
    return (br["floor"] - achieved) / span


def main(warp_path, warp_init, third_term, n_tasks, seed, out):
    warp = ExplicitWarp(dim=2, hidden=30, init=warp_init)
    warp.load_state_dict(torch.load(warp_path, map_location="cpu"))
    warp.eval()
    for p in warp.parameters():
        p.requires_grad_(False)

    print("=" * 66)
    print("(A) COVERAGE  can Omega even reach the initialisation domain?")
    cov = coverage(warp, n=256, seed=seed)
    for k, v in cov.items():
        print(f"    {k:18s} {v}")

    print()
    print("(C) SHORTCUT + (B) BRACKET over random tasks")
    gen = torch.Generator().manual_seed(seed + 1)
    flee_scores, spans, degenerate = [], [], 0
    for _ in range(n_tasks):
        t = sample_task(gen, third_term=third_term)
        br = bracket(t)
        if abs(br["floor"] - br["ceiling"]) < 1e-6:
            degenerate += 1
            continue
        spans.append(br["floor"] - br["ceiling"])
        flee_scores.append(normalised(br["flee_to_infinity"], br))
    flee_scores = np.array(flee_scores)
    print(f"    tasks sampled       {n_tasks}  (degenerate/flat: {degenerate})")
    print(f"    median bracket span {np.median(spans):.4f}")
    print(f"    flee-to-infinity    mean {flee_scores.mean():+.3f}  "
          f"median {np.median(flee_scores):+.3f}  "
          f"beats-guessing {float((flee_scores > 0).mean()):.1%}")

    verdict_ok = cov["coverage"] >= 0.80
    print()
    print("=" * 66)
    if verdict_ok:
        print(f"VERDICT: PROCEED.  coverage {cov['coverage']:.1%} >= 80%.")
    else:
        print(f"VERDICT: REFUSE.  coverage {cov['coverage']:.1%} < 80%.")
        print("  Most initialisations are OUTSIDE the image of Omega, so WarpGrad")
        print("  and gradient descent do not start from the same point and the")
        print("  comparison is undefined.  Any win rate computed here is an")
        print("  artefact of the warp's limited range, not a property of WarpGrad.")
        print("  Fix the warp parameterisation (see --warp-init residual) or")
        print("  restrict evaluation to the reachable subset and say so.")
    print("=" * 66)

    rec = dict(warp=warp_path, warp_init=warp_init, third_term=third_term,
               coverage=cov, flee_mean=float(flee_scores.mean()),
               flee_beats_guessing=float((flee_scores > 0).mean()),
               median_bracket_span=float(np.median(spans)),
               degenerate_tasks=degenerate, verdict="PROCEED" if verdict_ok else "REFUSE")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True)
    ap.add_argument("--warp-init", default="plain", choices=["plain", "residual"])
    ap.add_argument("--third-term", default="peaks", choices=["peaks", "paper"])
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mountain/check_warp.json")
    a = ap.parse_args()
    sys.exit(main(a.warp, a.warp_init, a.third_term, a.n_tasks, a.seed, a.out))
