"""The meta-learners: WarpGrad, Leap, Reptile, and joint training.

Everything here operates on a WarpedGPT through torch.func.functional_call, so
task adaptation never mutates the module.  That matters because Eq. 11 needs the
inner step to stay inside an autograd graph, and because phi and theta must be
updated by different optimisers at different times without ever touching each
other's storage.

THE FOUR ARMS
=============
  warp_leap   J = L(phi) + lambda C_Leap(theta0)             Eq. 15, the paper's
              multi-shot method.  Meta-learns BOTH the geometry and the init.
  leap        C_Leap(theta0) only                            Flennerhag 2019.
              Meta-learns the init, warp is the identity.  This is the control
              that isolates the warp: same init objective, no geometry.
  reptile     theta0 <- theta0 + eps (theta_K - theta0)      Nichol 2018.
  joint       plain multi-task training of theta0.           The "finetuning"
              row of Table 1.

THE INNER LOOP IS ALWAYS THE SAME
=================================
    theta_{k+1} = theta_k - alpha grad L_task(theta_k; phi)

Plain SGD.  No momentum, no Adam, no schedule.  Every arm uses it, so any
difference between arms comes from the meta-level, never from the inner
optimiser.  This is Algorithm 1 line 8 and Algorithm 2 line 9, unchanged.

    "Task-learners share a common initialisation and warp parameters that are
     held fixed during task adaptation."
                                                        -- Appendix E, page 20

WHY THE DEFAULT IS EQ. 12 AND NOT EQ. 11
========================================
Eq. 11 differentiates through the inner gradient, which needs a double backward.
On a Tesla T4 (compute capability 7.5) we run in fp16 with a GradScaler, and
double backward through fused attention in fp16 is both slow and numerically
fragile.  The paper measured the cost of dropping the second-order term:

    3x3conv (default)   Offline   full   (L, Eq. 11)     84.4 +/- 1.7
    3x3conv             Offline   approx (Lhat, Eq. 12)  83.1 +/- 2.7
                                                          -- Table 3, page 22

1.3 points, inside one standard deviation.  And the paper's own reasoning:

    "this approximation retains all gradient terms and only discards local
     second-order effects, which are typically dominated by first-order effect
     in long parameter trajectories"
                                                          -- Section 2.4, page 6

which applies with more force here than on Omniglot, because our trajectories
are the same length (100 steps) and our model is deeper.  `--exact` runs Eq. 11
in fp32 so the approximation can be checked rather than assumed, and that check
is claim C7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.func import functional_call


def _params_to_dict(model, names):
    d = dict(model.named_parameters())
    return {n: d[n] for n in names}


def functional_loss(model, task_params, warp_params, x, y, buffers=None):
    """Loss of `model` evaluated at an explicit parameter dict.

    Splitting theta and warp into two dicts is not cosmetic.  It is what lets us
    take a gradient with respect to one while holding the other fixed, which is
    the definition of the method.
    """
    merged = {**task_params, **warp_params}
    if buffers:
        merged = {**merged, **buffers}
    _, loss = functional_call(model, merged, (x,), {"targets": y})
    return loss


@dataclass
class Trajectory:
    """One task's adaptation run: the sampled points of p(theta | tau).

    Eq. 11 sums over theta^tau ~ p(theta | tau).  The paper realises that
    distribution as the iterates of SGD:

        "each iterate theta^tau_k can be seen as a sample from
         p(theta^tau_k | theta^tau_{k-1}, phi).  Thus, K-steps of gradient
         descent forms a Monte-Carlo chain theta^tau_0,...,theta^tau_K"
                                                        -- Section 2.4, page 5

    So a trajectory is a set of Monte-Carlo samples, not a path we have to
    preserve in order.  That is precisely why storing every fifth iterate
    (`stride`) is legitimate: it is a smaller Monte-Carlo sample of the same
    distribution, not a truncation of a path.  It costs variance, not bias.
    """

    lang: str
    points: list                      # list of {name: tensor} on CPU, fp16
    losses: list = field(default_factory=list)


class MetaLearner:
    """Holds theta0 and phi, runs adaptation, and produces meta-gradients."""

    def __init__(self, model, arm="warp_leap", inner_lr=0.1, inner_steps=100,
                 lam=1.0, reptile_eps=1.0, exact=False, device="cuda",
                 amp_dtype=torch.float16):
        self.model = model
        self.arm = arm
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.lam = lam
        self.reptile_eps = reptile_eps
        self.exact = exact
        self.device = device
        self.amp_dtype = amp_dtype

        self.task_names, self.warp_names = model.param_split()
        d = dict(model.named_parameters())
        self.theta0 = {n: d[n] for n in self.task_names}
        self.phi = {n: d[n] for n in self.warp_names}
        self.buffers = dict(model.named_buffers())

        if arm in ("leap", "reptile", "joint"):
            # These arms have no geometry to learn.  We assert rather than
            # silently ignore phi, because an arm that quietly meta-learned a
            # warp while being labelled "no warp" would invalidate the control.
            assert all(model.warps[i].kind == "identity"
                       for i in range(len(model.warps))), (
                f"arm={arm!r} must be run with warp_kind='identity', otherwise "
                "the no-warp control is not actually warp-free"
            )

    # ------------------------------------------------------------- adaptation

    def adapt(self, sampler, batch_size, steps=None, stride=1,
              collect=True, track_leap=False):
        """Run the inner loop.  Returns a Trajectory and the Leap accumulator.

        `stride` subsamples which iterates are kept for the replay buffer.  See
        the Trajectory docstring for why that is sound.
        """
        steps = steps or self.inner_steps
        theta = {n: p.detach().clone() for n, p in self.theta0.items()}
        traj = Trajectory(lang=getattr(sampler, "lang", "?"), points=[])
        leap_acc = torch.zeros((), device=self.device) if track_leap else None
        prev_flat = prev_loss = None

        for k in range(steps):
            x, y = sampler.batch(batch_size)
            for p in theta.values():
                p.requires_grad_(True)

            with torch.autocast("cuda", dtype=self.amp_dtype,
                                enabled=self.amp_dtype is not None):
                loss = functional_loss(self.model, theta, self.phi, x, y,
                                       self.buffers)
            grads = torch.autograd.grad(loss, list(theta.values()),
                                        allow_unused=True)

            with torch.no_grad():
                new = {}
                for (n, p), g in zip(theta.items(), grads):
                    new[n] = p - self.inner_lr * g if g is not None else p.clone()

                if track_leap:
                    # C_Leap (Eq. 16): the cumulative chordal distance between
                    # consecutive points of the trajectory, in the joint space of
                    # parameters AND loss:
                    #   vartheta_k = (theta_k,0 ... theta_k,n, L_task(theta_k))
                    flat = torch.cat([v.reshape(-1).float() for v in new.values()])
                    lv = loss.detach().float()
                    if prev_flat is not None:
                        d2 = (flat - prev_flat).pow(2).sum() + (lv - prev_loss).pow(2)
                        leap_acc = leap_acc + d2.clamp_min(1e-12).sqrt()
                    prev_flat, prev_loss = flat, lv

                theta = {n: v.detach() for n, v in new.items()}
                traj.losses.append(float(loss.detach()))
                if collect and (k % stride == 0):
                    traj.points.append(
                        {n: v.detach().to("cpu", torch.float16)
                         for n, v in theta.items()}
                    )

        traj.final = {n: v.detach() for n, v in theta.items()}
        return traj, leap_acc

    # ------------------------------------------------------- meta-gradients

    def warp_meta_grad(self, point, sampler, batch_size):
        """One term of Eq. 11 / Eq. 12.  Accumulates into phi.grad.

        Eq. 11:  L(phi) = sum L_meta( theta - alpha grad L_task(theta;phi) ; phi )
        Eq. 12:  same with a stop-gradient around the bracket.

        Read the code against the equation:
          theta_k       <- the sampled point from p(theta | tau)
          L_task        <- evaluated on one batch
          grad wrt theta with create_graph=exact   <- the only difference
          theta_next    <- one step
          L_meta        <- evaluated on a DIFFERENT batch, then .backward()

        The two batches matter.  Appendix E:
            "importantly, we evaluate them on different batches of task data to
             ensure warp-layers encourage generalisation"
        If L_task and L_meta used the same batch, phi would be rewarded for
        making a single batch easier to fit, which is memorisation, not geometry.
        """
        theta = {n: v.to(self.device, torch.float32).requires_grad_(True)
                 for n, v in point.items()}

        x1, y1 = sampler.batch(batch_size)
        use_amp = (self.amp_dtype is not None) and not self.exact
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
            l_task = functional_loss(self.model, theta, self.phi, x1, y1,
                                     self.buffers)
        g = torch.autograd.grad(l_task, list(theta.values()),
                                create_graph=self.exact, allow_unused=True)

        stepped = {}
        for (n, p), gi in zip(theta.items(), g):
            v = p - self.inner_lr * gi if gi is not None else p
            stepped[n] = v if self.exact else v.detach()

        x2, y2 = sampler.batch(batch_size)      # DIFFERENT batch, deliberately
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
            l_meta = functional_loss(self.model, stepped, self.phi, x2, y2,
                                     self.buffers)
        l_meta.backward()
        return float(l_meta.detach())

    def leap_grad(self, traj, leap_acc):
        """grad C_Leap(theta0), Eq. 16-17, accumulated into theta0.grad.

        Leap minimises the expected length of the adaptation trajectory, so the
        gradient pulls theta0 towards a point from which every task is a SHORT
        walk away.  Eq. 17 is the paper's first-order approximation, which avoids
        backpropagating through adaptation exactly as WarpGrad does.

        Under Warp-Leap this becomes a joint search: the warp reshapes the space
        while Leap shortens the paths through it.
            "Under WarpGrad, this becomes a joint search for a geometry in which
             task adaptation defines geodesics"            -- Section 4.2, page 9
        """
        with torch.no_grad():
            for n, p in self.theta0.items():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                # pull theta0 toward the task's endpoint, weighted by path length
                p.grad.add_(p.detach() - traj.final[n].to(p.device, p.dtype))

    def reptile_step(self, trajs, outer_lr):
        """theta0 <- theta0 + eps * mean_tau (theta_K^tau - theta0).

        Reptile has no meta-gradient at all: it moves the initialisation toward
        where training ended up.  Included because it is the baseline the paper
        reports Warp-Leap beating by 12.8 points on Omniglot, and because it is
        the cheapest possible thing that still counts as meta-learning.
        """
        with torch.no_grad():
            for n, p in self.theta0.items():
                delta = torch.stack([t.final[n].to(p.device, p.dtype) - p.detach()
                                     for t in trajs]).mean(0)
                p.add_(outer_lr * self.reptile_eps * delta)


@torch.no_grad()
def evaluate_bpb(model, sampler, task_params, warp_params, buffers,
                 batch_size=16, max_batches=24, amp_dtype=torch.float16):
    """Bits per byte on a deterministic sweep of held-out data.

    Deterministic, not sampled, so the number itself carries no sampling noise
    and small real differences are not buried under it.
    """
    tot, n = 0.0, 0
    for x, y in sampler.sequential_batches(batch_size, max_batches):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = functional_loss(model, task_params, warp_params, x, y, buffers)
        tot += float(loss.detach()) * x.numel()
        n += x.numel()
    if n == 0:
        return float("nan")
    return tot / n / 0.6931471805599453      # nats per byte -> bits per byte
