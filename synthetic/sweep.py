"""Sweep the one hyper-parameter Appendix D does not give us, with a real eval.

Appendix D specifies the task family, the task learning rate (0.1), the number
of task steps (100), the number of meta-steps (100), the number of inits per
meta-step (10), and the warp architecture (2-layer, 30 hidden, tanh).  It does
NOT specify the meta learning rate beta, and it does not specify the meta
optimiser for this experiment.

That single missing number decides whether the experiment works.  The first run
of this reproduction used Adam at 1e-3 (the paper's stated choice elsewhere) and
found that WarpGrad did not beat gradient descent.  Before reporting that as a
failure to reproduce, we have to rule out the boring explanation: that the warp
simply never moved.  So this script

  1. builds a FIXED evaluation set of (task, init) pairs, held constant across
     every run and every meta-step, with brackets precomputed once,
  2. measures the paired score difference (WarpGrad minus GD) on that fixed set
     BEFORE meta-training and at intervals during it, and
  3. does that across a grid of meta learning rates and both warp
     parameterisations.

Measuring before any meta-update is the important part.  With init="residual"
the warp starts as exactly the identity, so the score difference at meta-step 0
must be 0.000 by construction.  If it is not, the harness is broken rather than
the method.  That is a free correctness check on the whole measurement chain,
and it costs one extra evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from synthetic.check_warp import TOL, bracket, invert_warp, normalised
from synthetic.mountain import TASK_LR, TASK_STEPS, sample_init, sample_task
from synthetic.train_mountain import meta_train, plain_gd_trajectory, warp_trajectory
from warpgrad.geometry import ExplicitWarp


class FixedEval:
    """A held-out set of (task, init) pairs that never changes.

    Brackets (ceiling and floor) depend only on the task, so they are computed
    once here and reused.  GD scores likewise do not depend on the warp, so they
    are computed once.  Only the WarpGrad arm is recomputed as the warp changes,
    which makes the in-training evaluation cheap.
    """

    def __init__(self, n=32, third_term="peaks", seed=4242,
                 steps=TASK_STEPS, lr=TASK_LR):
        gen = torch.Generator().manual_seed(seed)
        self.steps, self.lr, self.tol = steps, lr, TOL
        self.tasks, self.x0s, self.brackets, self.gd = [], [], [], []

        while len(self.tasks) < n:
            t = sample_task(gen, third_term=third_term)
            br = bracket(t)
            if abs(br["floor"] - br["ceiling"]) < 1e-6:
                continue                       # skip degenerate flat surfaces
            x0 = sample_init(gen, 1)
            traj = plain_gd_trajectory(t, x0, steps, lr)
            with torch.no_grad():
                gd_f = float(t(traj[-1, 0].reshape(1, 2)))
            self.tasks.append(t)
            self.x0s.append(x0)
            self.brackets.append(br)
            self.gd.append(normalised(gd_f, br))

        self.x0_batch = torch.cat(self.x0s, dim=0)          # (n, 2)
        self.gd = np.array(self.gd)

    def __call__(self, warp, invert_steps=800, restarts=2):
        """Return mean paired (WarpGrad - GD) normalised score on the fixed set."""
        was_training = [p.requires_grad for p in warp.parameters()]
        for p in warp.parameters():
            p.requires_grad_(False)
        try:
            theta0, res = invert_warp(warp, self.x0_batch,
                                      steps=invert_steps, restarts=restarts)
            wg, used = [], 0
            for i, (t, br) in enumerate(zip(self.tasks, self.brackets)):
                if float(res[i]) > self.tol:
                    wg.append(np.nan)          # unreachable, excluded
                    continue
                _, tw = warp_trajectory(t, warp, theta0[i:i + 1], self.steps, self.lr)
                with torch.no_grad():
                    wg.append(normalised(float(t(tw[-1, 0].reshape(1, 2))), br))
                used += 1
            wg = np.array(wg)
            mask = ~np.isnan(wg)
            if mask.sum() == 0:
                return float("nan")
            return float((wg[mask] - self.gd[mask]).mean())
        finally:
            for p, g in zip(warp.parameters(), was_training):
                p.requires_grad_(g)

    def coverage(self, warp, invert_steps=800, restarts=2):
        for p in warp.parameters():
            p.requires_grad_(False)
        _, res = invert_warp(warp, self.x0_batch, steps=invert_steps, restarts=restarts)
        for p in warp.parameters():
            p.requires_grad_(True)
        return float((res < self.tol).float().mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-lrs", type=float, nargs="+",
                    default=[1e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--warp-inits", nargs="+", default=["residual", "plain"])
    ap.add_argument("--meta-steps", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--third-term", default="peaks", choices=["peaks", "paper"])
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", default="results/mountain/sweep.json")
    a = ap.parse_args()

    print(f"building fixed eval set ({a.n_eval} pairs, seed 4242)")
    ev = FixedEval(n=a.n_eval, third_term=a.third_term)
    print(f"  GD baseline mean normalised score {ev.gd.mean():+.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []

    for init, lr, seed in itertools.product(a.warp_inits, a.meta_lrs, a.seeds):
        tag = f"{init}_lr{lr:g}_s{seed}"
        print(f"\n=== {tag} " + "=" * (50 - len(tag)))
        torch.manual_seed(seed)
        warp = ExplicitWarp(dim=2, hidden=30, init=init)

        hist = meta_train(
            warp, meta_steps=a.meta_steps, meta_lr=lr, seed=seed,
            third_term=a.third_term, log_every=0,
            eval_every=a.eval_every, eval_fn=ev,
        )
        cov = ev.coverage(warp)
        rec = dict(warp_init=init, meta_lr=lr, seed=seed,
                   eval_step=hist["eval_step"], eval_score=hist["eval_score"],
                   final_score=hist["eval_score"][-1],
                   initial_score=hist["eval_score"][0],
                   final_cond=hist["cond"][-1], coverage=cov)
        results.append(rec)
        print(f"  init {rec['initial_score']:+.4f}  ->  final {rec['final_score']:+.4f}"
              f"   cond(G^-1) {rec['final_cond']:.3f}   coverage {cov:.1%}")
        json.dump(dict(gd_baseline=float(ev.gd.mean()), runs=results),
                  open(a.out, "w"), indent=2)

    print("\n" + "=" * 74)
    print(f"{'config':<22}{'coverage':>10}{'cond':>8}{'score@0':>10}{'score@end':>12}")
    print("-" * 74)
    for init, lr in itertools.product(a.warp_inits, a.meta_lrs):
        rs = [r for r in results if r["warp_init"] == init and r["meta_lr"] == lr]
        if not rs:
            continue
        fin = np.array([r["final_score"] for r in rs])
        ini = np.array([r["initial_score"] for r in rs])
        cov = np.mean([r["coverage"] for r in rs])
        cnd = np.mean([r["final_cond"] for r in rs])
        print(f"{init + ' lr=' + format(lr, 'g'):<22}{cov:>9.1%}{cnd:>8.2f}"
              f"{ini.mean():>+10.4f}{fin.mean():>+9.4f}+/-{fin.std() / max(len(fin) ** 0.5, 1):.3f}")
    print("=" * 74)
    print("score = mean paired (WarpGrad - GD) normalised score on the fixed set.")
    print("positive means WarpGrad beats gradient descent.  0.0000 at step 0 for")
    print("init=residual is a correctness check, not a result.")
    print(f"wrote {a.out}")
