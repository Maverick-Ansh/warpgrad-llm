"""
Explicit warp geometry: the P-space / W-space picture of WarpGrad.

This module implements Section 2.3, "The Geometry of Warped Gradient Descent",
of Flennerhag et al., "Meta-Learning with Warped Gradient Descent", ICLR 2020
(arXiv:1909.00025v2).

WHY AN *EXPLICIT* WARP EXISTS AT ALL
------------------------------------
There are two ways to warp gradients in the paper, and it is easy to conflate
them:

  1. IMPLICIT warp (Section 2.2, Eq. 6).  You interleave warp-layers between
     the task-learner's layers.  Nobody ever writes down a preconditioning
     matrix.  Backpropagation through the warp-layer Jacobians preconditions
     the gradient for free.  This is what you use on a real network, and it
     lives in `warpgrad/warp.py`.

  2. EXPLICIT warp (Section 2.3, Appendix D).  You write down a map
     Omega : P -> W from a "warped" parameter space P onto the manifold W
     where the task loss actually lives.  You can then *see* the geometry.
     This is only tractable in low dimension, which is exactly why the paper
     uses it for the 2-D synthetic experiment that produces Figure 3.

This file is case 2.  It is the instrument that draws the mountain.

THE PAPER'S OWN WORDS
---------------------
    "Let Omega represent the effect of warp-layers by a reparameterisation
     h(i)(x; Omega(theta;phi)(i)) = omega(i)(h(i)(x;theta(i));phi) for all x,i
     that maps from a space P onto the manifold W with gamma = Omega(theta;phi).
     We induce a metric G on W by push-forward:

        Delta theta := grad (L . Omega) (theta;phi)
                     = [Dx Omega(theta;phi)]^T grad L(gamma)      P-space   (7)

        Delta gamma := Dx Omega(theta;phi) Delta theta
                     = G(gamma;phi)^-1 grad L(gamma)              W-space   (8)

     where G^-1 := [Dx Omega][Dx Omega]^T.  Provided Omega is not degenerate
     (G is non-singular), G^-1 is positive-definite, hence a valid Riemann
     metric."
                                                     -- Section 2.3, page 5

and the first-order equivalence that licenses the whole method:

    "(L . Omega)(theta - alpha Delta theta) = L(gamma - alpha Delta gamma)
     + O(alpha^2)"                                                          (9)

WHAT THAT MEANS IN PLAIN ENGLISH
--------------------------------
You have a horrible loss surface.  You do not change it.  Instead you invent a
second, private coordinate system P, and a smooth map Omega that sends your
private coordinates onto the horrible surface.  You then run ordinary gradient
descent in your private coordinates.  Because of the chain rule, an ordinary
step in P lands as a *preconditioned* step in W, and the preconditioner is
G^-1 = [Dx Omega][Dx Omega]^T, which is automatically positive-definite.
Positive-definite means it can never point you uphill, so you keep every
convergence guarantee of plain gradient descent while moving in a much better
direction.  Eq. 9 says the two views agree to first order in the step size.

Meta-learning then means: choose Omega so that the private coordinate system is
a nice place to do gradient descent, for every task you expect to see.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExplicitWarp(nn.Module):
    """Omega : R^d -> R^d, the reparameterisation of Eq. 7-8.

    Appendix D fixes the architecture for the synthetic experiment:

        "To visualise the geometry, we use an explicit warp Omega defined by a
         2-layer feed-forward network with a hidden-state size of 30 and tanh
         non-linearities."
                                                        -- Appendix D, page 18

    So: gamma = W2 @ tanh(W1 @ theta + b1) + b2, with hidden width 30.

    DEVIATION FLAG (init).  The paper does not state how Omega is initialised.
    This matters more than it looks.  A freshly initialised 2-layer tanh net is
    a near-arbitrary squashing map, so at meta-step 0 the "warped" surface is
    not a mild deformation of the native one, it is unrelated to it.  We expose
    both readings and report both:

      init="plain"     literal reading.  Standard PyTorch init, no identity bias.
      init="residual"  Omega(theta) = theta + g(theta) with g's last layer set
                       to zero, so Omega starts exactly as the identity and
                       WarpGrad starts exactly as gradient descent.  Any
                       improvement is then unambiguously *learned*, which makes
                       the claim easier to falsify, not harder.

    We default to "plain" because it is what the paper says.  See REPORT.md for
    what each one actually does.
    """

    def __init__(self, dim: int = 2, hidden: int = 30, init: str = "plain"):
        super().__init__()
        if init not in ("plain", "residual"):
            raise ValueError(f"init must be 'plain' or 'residual', got {init!r}")
        self.dim, self.init = dim, init
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        if init == "residual":
            # zero final layer => g(theta) == 0 => Omega == identity at step 0
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """theta lives in P-space, the returned gamma lives in W-space."""
        g = self.fc2(torch.tanh(self.fc1(theta)))
        return theta + g if self.init == "residual" else g

    # ---------------------------------------------------------------- metric

    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        """Dx Omega(theta), shape (batch, dim, dim).

        Row i, column j is d gamma_i / d theta_j.  This is the object that does
        all the work in Eq. 7 and Eq. 8.
        """
        theta = theta.detach().requires_grad_(True)
        # jacrev over a single point, vmapped across the batch. Cheap at d=2.
        jac = torch.vmap(torch.func.jacrev(self.forward))(theta.unsqueeze(1))
        return jac.squeeze(1).squeeze(2)

    def metric_inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """G^-1 = [Dx Omega][Dx Omega]^T, the preconditioner of Eq. 8.

        Symmetric by construction and positive-semi-definite by construction,
        which is the paper's argument that WarpGrad inherits the convergence
        guarantees of gradient descent.  It is positive-*definite*, and hence a
        genuine Riemann metric, exactly when Dx Omega has full rank, which is
        the paper's non-degeneracy condition.
        """
        J = self.jacobian(theta)
        return J @ J.transpose(-1, -2)

    def condition_number(self, theta: torch.Tensor) -> torch.Tensor:
        """Largest over smallest eigenvalue of G^-1.

        A diagnostic the paper does not report but that we use as a probe-free
        measurement of how much warping is actually happening.  1.0 means the
        warp is locally a rotation plus uniform scale, so it is doing nothing
        that a learning rate could not do.  Large values mean genuinely
        anisotropic preconditioning.
        """
        ev = torch.linalg.eigvalsh(self.metric_inverse(theta))
        return ev[..., -1] / ev[..., 0].clamp_min(1e-12)
