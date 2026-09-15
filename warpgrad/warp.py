"""Implicit warp-layers: gradient preconditioning that nobody ever computes.

This is Section 2.2 of Flennerhag et al. (ICLR 2020), the half of WarpGrad that
runs on a real network, as opposed to the explicit 2-D picture in geometry.py.

THE ONE EQUATION THAT MATTERS
=============================
    "We insert warp-layers that are universal function approximators
     parameterised by neural networks into the task-learner without restricting
     their form or how they interact with f.  In the simplest case, we interleave
     warp-layers between layers of the task-learner to obtain
     fhat = omega(L) . h(L) . ... . omega(1) . h(1) ...
     Backpropagation automatically induces gradient preconditioning, as in
     T-Nets, but in our case via the Jacobians of the warp-layers:

        dL/dtheta(i) = E[ grad l^T ( prod_{j=0}^{L-(i+1)}
                          Dx omega(L-j) Dx h(L-j) ) Dx omega(i) Dtheta h(i) ]   (6)
                                                          -- Section 2.2, page 4

Read Eq. 6 as a sentence.  The gradient that reaches task-layer i has to travel
back through every warp-layer above it.  Each warp-layer multiplies it by that
warp-layer's Jacobian.  So the gradient task-layer i finally receives is the
plain gradient premultiplied by a product of matrices that the meta-learner
controls.  That product IS the preconditioner.

The practical consequence is the nicest thing about this method:

    YOU NEVER BUILD A PRECONDITIONING MATRIX.  YOU NEVER INVERT ANYTHING.
    YOU ADD SOME LAYERS AND CALL .backward().

A d_model x d_model preconditioner for a 256-wide model would be 65k entries per
layer and would need inverting.  Eq. 6 gets the same effect from an ordinary
Linear layer and the chain rule.  That is the entire trick.

WHY NON-LINEARITY IS THE CONTRIBUTION
=====================================
    "In the special case where f is feed-forward and each omega a linear
     projection, we obtain an instance of WarpGrad that is akin to T-Nets since
     preconditioning is given by Dx omega = T.  Conversely, by making warp-layers
     non-linear, we can induce interdependence between warp-layers, allowing
     WarpGrad to model preconditioning beyond the block-diagonal structure
     imposed by prior works.  Further, this enables a form of task-conditioning
     by making Jacobians of warp-layers data dependent."
                                                          -- Section 2.2, page 4

A LINEAR warp has a constant Jacobian.  Same preconditioner for every input, so
it is a fixed block-diagonal matrix, which is what T-Nets and Meta-Curvature
already had.  A NON-LINEAR warp has an input-dependent Jacobian, so the
preconditioner changes with the data flowing through it.  That is how a warp
"conditions on the task" without ever being told which task it is on.

PORTING TABLE 3'S LADDER FROM CONVNETS TO TRANSFORMERS
=====================================================
Appendix F ablates warp capacity on Omniglot.  The ladder transfers to a
transformer residual stream almost one to one, because a 3x3 convolution over a
feature map and a d_model x d_model projection over a residual stream play the
same structural role: a learned mixing of channels.

    Omniglot warp (Table 3)                accuracy   our transformer analogue
    ------------------------------------   --------   ------------------------
    None (Leap)                            74.8       "none"
    Scaling (FiLM-like, channel-wise)      77.5       "scale"
    1x1 conv                               79.4       "lowrank"
    3x3 conv (the paper's default)         84.4       "linear"
    3x3 conv + ReLU                        83.4       "linear_act"
    3x3 conv + BN + ReLU                   85.0       "norm_act"
    3x3 conv + BN + Res + ReLU             86.3       "residual"
    2-layer 3x3 conv + BN + Res            88.0       "mlp"

The prediction under test (claim C4) is that the ORDER survives the port: more
warp capacity should mean better adaptation to held-out languages, with the
jump from block-diagonal (linear) to beyond-block-diagonal (mlp) being the one
the paper argues is its own contribution.

WARP PARAMETERS ARE FROZEN DURING TASK ADAPTATION
=================================================
    "Task-learners share a common initialisation and warp parameters that are
     held fixed during task adaptation."
                                                        -- Appendix E, page 20

This is not an optimisation detail, it is the definition of the method.  phi is
the geometry.  If phi moved during adaptation, the geometry would be changing
underneath the optimiser and "warped gradient descent" would mean nothing.
`WarpLayer` therefore tags its parameters so the training loop can split
parameters into task (theta) and warp (phi) sets and never confuse them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

WARP_TAG = "_is_warp_param"


def mark_warp(module: nn.Module) -> nn.Module:
    """Tag every parameter of `module` as a warp parameter (phi, meta-learned)."""
    for p in module.parameters(recurse=True):
        setattr(p, WARP_TAG, True)
    return module


def is_warp_param(p: torch.nn.Parameter) -> bool:
    return getattr(p, WARP_TAG, False)


def split_parameters(model: nn.Module):
    """Return (task_names, warp_names).  The split that defines the method.

    Anything inside a WarpLayer is phi and is meta-learned.  Everything else is
    theta and is adapted per task.  We split by NAME rather than by object so the
    result can be fed straight to torch.func.functional_call, which is how the
    inner loop stays differentiable without mutating the module.
    """
    task, warp = [], []
    for name, p in model.named_parameters():
        (warp if is_warp_param(p) else task).append(name)
    return task, warp


class WarpLayer(nn.Module):
    """omega: R^d -> R^d applied to a transformer residual stream.

    Shape in and out is (batch, time, d_model), so a warp-layer is a drop-in
    that can sit between any two blocks without changing anything else.

    `kind` walks the Table 3 capacity ladder:

      "identity"  no warp at all.  The control arm.  Dx omega = I, so Eq. 6
                  reduces to the ordinary gradient and WarpGrad reduces to
                  whatever the base meta-learner was.
      "scale"     per-channel gain.  Dx omega = diag(s), a diagonal
                  preconditioner.  This is Meta-SGD's expressive class (Eq. 3),
                  reached by a different route.
      "lowrank"   d -> r -> d with r << d.  Preconditioner of rank r.
      "linear"    full d x d projection.  Constant Jacobian, so a fixed
                  block-diagonal preconditioner.  THIS IS THE T-NET EQUIVALENT
                  and the paper's default.  Everything below beats it only if
                  non-linearity really buys something.
      "linear_act" linear then GELU.  First input-dependent Jacobian.
      "norm_act"  LayerNorm, linear, GELU.  LayerNorm stands in for the paper's
                  BatchNorm: batch statistics over a language-modelling batch are
                  a different and worse object than over an image batch, and
                  every transformer in practice uses LayerNorm.  Deviation, noted.
      "residual"  x + GELU(linear(LayerNorm(x))).  Jacobian is I + something, so
                  it starts near the identity and can only help.
      "mlp"       x + W2 GELU(W1 LayerNorm(x)) with an expansion.  Two layers,
                  the paper's best Omniglot warp.  Genuinely beyond
                  block-diagonal.

    INITIALISATION.  Every non-identity kind is initialised so that omega starts
    at (or extremely close to) the identity map.  The reason is the same one that
    forced the residual parameterisation in geometry.py: if the warp starts as an
    arbitrary map, the task-learner at meta-step 0 is not the baseline
    architecture, it is a different and probably worse architecture, and any
    later gap would confound "warping helps" with "this random map happened to
    help".  Starting at the identity makes WarpGrad start EXACTLY as the
    no-warp baseline, so every point of difference is attributable to
    meta-learning.  The paper does not state this and we flag it in REPORT.md.
    """

    KINDS = ("identity", "scale", "lowrank", "linear", "linear_act",
             "norm_act", "residual", "mlp")

    def __init__(self, d_model: int, kind: str = "linear",
                 rank: int = 16, expansion: int = 2):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind, self.d_model = kind, d_model

        if kind == "identity":
            pass

        elif kind == "scale":
            # Dx omega = diag(s).  Starts at s = 1, i.e. the identity.
            self.s = nn.Parameter(torch.ones(d_model))
            self.b = nn.Parameter(torch.zeros(d_model))

        elif kind == "lowrank":
            # omega(x) = x + U V x, with V zero so it starts at the identity.
            self.down = nn.Linear(d_model, rank, bias=False)
            self.up = nn.Linear(rank, d_model, bias=False)
            nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(d_model))
            nn.init.zeros_(self.up.weight)

        elif kind in ("linear", "linear_act"):
            # Full d x d.  Weight = I exactly, so it begins as the identity even
            # though it has the capacity to become any linear map.
            self.proj = nn.Linear(d_model, d_model, bias=True)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(d_model))
                self.proj.bias.zero_()

        elif kind == "norm_act":
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, d_model, bias=True)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(d_model))
                self.proj.bias.zero_()

        elif kind == "residual":
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, d_model, bias=True)
            nn.init.zeros_(self.proj.weight)      # residual branch starts at 0
            nn.init.zeros_(self.proj.bias)

        elif kind == "mlp":
            h = d_model * expansion
            self.norm = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, h, bias=True)
            self.fc2 = nn.Linear(h, d_model, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.02)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)       # residual branch starts at 0
            nn.init.zeros_(self.fc2.bias)

        mark_warp(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kind
        if k == "identity":
            return x
        if k == "scale":
            return x * self.s + self.b
        if k == "lowrank":
            return x + self.up(self.down(x))
        if k == "linear":
            return self.proj(x)
        if k == "linear_act":
            return F.gelu(self.proj(x))
        if k == "norm_act":
            return F.gelu(self.proj(self.norm(x)))
        if k == "residual":
            return x + F.gelu(self.proj(self.norm(x)))
        if k == "mlp":
            return x + self.fc2(F.gelu(self.fc1(self.norm(x))))
        raise AssertionError(f"unhandled kind {k!r}")

    # -------------------------------------------------------------- diagnostics

    def jacobian_at(self, x: torch.Tensor) -> torch.Tensor:
        """Dx omega at a single residual-stream vector, shape (d, d).

        Used only for measurement, never in training.  Eq. 6 never needs this
        matrix built, but claim C5 (is the learned geometry Fisher-like?) and the
        linear-vs-non-linear question both need to look at it directly.
        """
        x = x.detach().reshape(-1)[: self.d_model].clone().requires_grad_(True)
        return torch.func.jacrev(lambda v: self.forward(v.reshape(1, 1, -1)).reshape(-1))(x)

    def is_data_dependent(self, x1: torch.Tensor, x2: torch.Tensor,
                          atol: float = 1e-5) -> bool:
        """True when Dx omega differs between two inputs.

        This is the operational definition of "beyond block-diagonal".  A linear
        warp must return False here and an "mlp" warp must return True, and
        `tests/test_warp.py` asserts exactly that.  It is a much sharper check
        than comparing accuracies, because it tests the mechanism the paper
        claims rather than a downstream consequence of it.
        """
        j1, j2 = self.jacobian_at(x1), self.jacobian_at(x2)
        return not torch.allclose(j1, j2, atol=atol)
