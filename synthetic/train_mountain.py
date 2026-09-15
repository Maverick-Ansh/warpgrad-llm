"""Meta-train an explicit warp on the Appendix D mountains, then draw Figure 3/7.

Implements Algorithm 1 (online meta-training) with the canonical meta-objective
Eq. 11, and its first-order variant Eq. 12.

ALGORITHM 1, VERBATIM FROM PAGE 6
---------------------------------
     1: initialise phi and p(theta0 | tau)
     2: while not done do
     3:   sample mini-batch of tasks T from p(tau)
     4:   g_phi, g_theta0 <- 0
     5:   for all tau in T do
     6:     theta^tau_0 ~ p(theta0 | tau)
     7:     for all k in 0,...,K_tau - 1 do
     8:       theta^tau_{k+1} <- theta^tau_k - alpha grad L^tau_task(theta^tau_k; phi)
     9:       g_phi    <- g_phi    + grad L(phi; theta^tau_k)
    10:       g_theta0 <- g_theta0 + grad C(theta0; theta^tau_{0:k})
    11:     end for
    12:   end for
    13:   phi    <- phi    - beta  g_phi
    14:   theta0 <- theta0 - lambda beta g_theta0
    15: end while

For the synthetic experiment there is no meta-learned initialisation, so line 10
and line 14 are inert.  Appendix D says the initialisation is drawn fresh:
"The task is to minimise the given objective from a randomly sampled
initialisation, x_{i=1,2} ~ U(-3,3)", which is p(theta0 | tau) = U(-3,3)^2 and
lambda C(theta0) = 0.

THE META-OBJECTIVE, EQ. 11
--------------------------
    "L(phi) := sum_{tau~p(tau)} sum_{theta^tau~p(theta|tau)}
               L^tau_meta( theta^tau - alpha grad L^tau_task(theta^tau; phi) ; phi )"

Read it slowly, because it is the whole idea.  You are NOT differentiating
through the training run.  You take a point theta that training happened to
visit, you take ONE step from it, and you ask "did that one step help?", then
you nudge phi to make that one step better.  The trajectory is only a device for
producing points to ask the question at.  That is what "trajectory agnostic"
means, and it is why the cost does not grow with the number of adaptation steps.

Note the two distinct ways phi enters:
  (a) inside the step, through grad L_task(theta; phi), and
  (b) outside the step, through L_meta( . ; phi).
Eq. 11 keeps both.  Eq. 12 applies a stop-gradient to the whole bracket:

    "Lhat(phi) := sum sum L^tau_meta( sg[ theta^tau
                    - alpha grad L^tau_task(theta^tau; phi) ] ; phi )"

which drops the second-order term from (a) but, unlike first-order MAML, keeps
every step in the sum rather than only the last one.

DEVIATION: Appendix D does not state the meta learning rate beta or the meta
optimiser for this experiment.  We use Adam at 1e-3, which is the choice the
paper states everywhere it states one ("We use Adam for meta-updates",
Appendix H).  Recorded in REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from synthetic.mountain import TASK_LR, TASK_STEPS, sample_init, sample_task
from warpgrad.geometry import ExplicitWarp


def plain_gd_trajectory(task, x0, steps=TASK_STEPS, lr=TASK_LR):
    """Baseline: ordinary gradient descent straight on the native surface W.

    This is the black path in Figure 3 and Figure 7.  No warp, no
    preconditioning, nothing meta-learned.
    """
    x = x0.clone().detach().requires_grad_(True)
    traj = [x.detach().clone()]
    for _ in range(steps):
        loss = task(x).sum()
        g, = torch.autograd.grad(loss, x)
        x = (x - lr * g).detach().requires_grad_(True)
        traj.append(x.detach().clone())
    return torch.stack(traj)              # (steps+1, n, 2) in W-space


def warp_trajectory(task, warp, theta0, steps=TASK_STEPS, lr=TASK_LR):
    """WarpGrad: ordinary gradient descent in P-space, viewed in W-space.

    This is the magenta path.  Note there is nothing clever in the update rule.
    It is literally `theta -= lr * grad`.  All of the intelligence sits in Omega,
    and the chain rule converts the plain step in P into a preconditioned step
    in W (Eq. 7 -> Eq. 8).
    """
    theta = theta0.clone().detach().requires_grad_(True)
    traj_p = [theta.detach().clone()]
    traj_w = [warp(theta).detach().clone()]
    for _ in range(steps):
        loss = task(warp(theta)).sum()
        g, = torch.autograd.grad(loss, theta)
        theta = (theta - lr * g).detach().requires_grad_(True)
        traj_p.append(theta.detach().clone())
        traj_w.append(warp(theta).detach().clone())
    return torch.stack(traj_p), torch.stack(traj_w)


def meta_train(
    warp,
    meta_steps=100,
    inits_per_step=10,
    task_steps=TASK_STEPS,
    task_lr=TASK_LR,
    meta_lr=1e-3,
    exact=True,
    third_term="peaks",
    seed=0,
    log_every=10,
    eval_every=0,
    eval_fn=None,
):
    """Algorithm 1.  Returns a history dict for the report.

    `eval_fn(warp) -> float` is evaluated every `eval_every` meta-steps on a
    FIXED held-out set of (task, init) pairs.  This is the only honest learning
    curve available here.  The `meta_loss` column below is NOT one: every
    meta-step samples a different surface, and surface depths vary by an order of
    magnitude because b_i ~ Cat(-5,...,5), so that column is dominated by which
    task happened to be drawn rather than by how good the warp is.

    Appendix D: "in each meta-step we sample a new task surface and a mini-batch
    of 10 random initialisations that we train separately.  We train to
    convergence and accumulate the warp meta-gradient online (Algorithm 1)."
    "We train warp parameters for 100 meta-training steps."
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(warp.parameters(), lr=meta_lr)
    hist = {"meta_loss": [], "final_task_loss": [], "cond": [], "meta_grad_norm": [],
            "eval_step": [], "eval_score": []}

    if eval_every and eval_fn is not None:
        hist["eval_step"].append(0)
        hist["eval_score"].append(eval_fn(warp))   # score BEFORE any meta-update

    for m in range(meta_steps):
        task = sample_task(gen, third_term=third_term)
        theta = sample_init(gen, inits_per_step)      # 10 inits, trained separately

        opt.zero_grad(set_to_none=True)
        running = 0.0

        for _k in range(task_steps):
            # theta_k is a SAMPLE from p(theta | tau).  It carries no history:
            # this is exactly what makes the meta-gradient trajectory agnostic.
            theta_k = theta.detach().requires_grad_(True)

            # --- inner step, line 8 --------------------------------------
            L_task = task(warp(theta_k)).sum()
            g_theta, = torch.autograd.grad(L_task, theta_k, create_graph=exact)
            theta_next = theta_k - task_lr * g_theta
            if not exact:
                theta_next = theta_next.detach()      # Eq. 12: sg[...]

            # --- meta term, line 9 ---------------------------------------
            # "did that one step help?", measured under the warp we are learning
            L_meta = task(warp(theta_next)).sum()
            L_meta.backward()                         # accumulates into .grad
            running += L_meta.item() / inits_per_step

            theta = theta_next.detach()               # advance the trajectory

        gn = torch.nn.utils.clip_grad_norm_(warp.parameters(), 1e9).item()
        opt.step()                                    # line 13

        with torch.no_grad():
            probe = sample_init(gen, 256)
        cond = warp.condition_number(probe).median().item()

        hist["meta_loss"].append(running / task_steps)
        hist["final_task_loss"].append(task(warp(theta)).mean().item())
        hist["cond"].append(cond)
        hist["meta_grad_norm"].append(gn)

        if eval_every and eval_fn is not None and (m + 1) % eval_every == 0:
            sc = eval_fn(warp)
            hist["eval_step"].append(m + 1)
            hist["eval_score"].append(sc)
            print(f"    [fixed eval] meta-step {m + 1:4d}  score {sc:+.4f}", flush=True)

        if log_every and (m % log_every == 0 or m == meta_steps - 1):
            print(
                f"  meta {m:4d}  meta_loss {hist['meta_loss'][-1]:9.4f}"
                f"  final_task {hist['final_task_loss'][-1]:9.4f}"
                f"  cond(G^-1) {cond:7.3f}  |g_phi| {gn:.3e}",
                flush=True,
            )

    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=100)
    ap.add_argument("--inits", type=int, default=10)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--warp-init", default="plain", choices=["plain", "residual"])
    ap.add_argument("--third-term", default="peaks", choices=["peaks", "paper"])
    ap.add_argument("--approx", action="store_true", help="use Eq. 12 not Eq. 11")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mountain")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    mode = "approx" if a.approx else "exact"
    tag = f"{a.warp_init}_{a.third_term}_{mode}_s{a.seed}"
    print(f"[{tag}] meta-training explicit warp, Algorithm 1")

    torch.manual_seed(a.seed)
    warp = ExplicitWarp(dim=2, hidden=30, init=a.warp_init)
    t0 = time.time()
    hist = meta_train(
        warp,
        meta_steps=a.meta_steps,
        inits_per_step=a.inits,
        meta_lr=a.meta_lr,
        exact=not a.approx,
        third_term=a.third_term,
        seed=a.seed,
    )
    print(f"[{tag}] done in {time.time() - t0:.1f}s")

    torch.save(warp.state_dict(), f"{a.out}/warp_{tag}.pt")
    json.dump(hist, open(f"{a.out}/hist_{tag}.json", "w"))
    print(f"[{tag}] saved to {a.out}/")
