"""CEILING and no-meta-learning FLOOR for the language tier.

Run this BEFORE the meta-learning sweep.  It answers a question that decides
whether the sweep is worth running at all:

    On a held-out language, with this architecture and this much data, what is
    the best bits-per-byte anyone could reach?  And what does the SAME model
    reach with no meta-learning at all?

If the gap between those two is small, then no meta-learner can demonstrate
anything on this setup, and the honest move is to widen the gap (more capacity,
more adaptation budget) rather than to run a sweep and report noise.

TWO REFERENCE ARMS
==================
  ORACLE      the same architecture trained on that ONE language from scratch
              with Adam for many times the adaptation budget.  This is the
              ceiling: a meta-learner adapting in 100 SGD steps should not beat
              a model that trained on the language properly.

  SCRATCH-K   the same architecture, random initialisation, adapted with exactly
              the same 100-step SGD inner loop every meta-learner uses.  No
              meta-learning of any kind.

SCRATCH-K IS THE PAPER'S OWN BASELINE
=====================================
    Method            10-way 640-shot    20-way 100-shot
    SGD (no meta-training)   58.1 +/- 1.5     51.0
    KFAC (no meta-training)     --            56.0
    ...
    Warp-Leap               80.4 +/- 1.6     83.6 +/- 1.9
                                                       -- Table 1, page 8

The paper's whole multi-shot claim is the distance between the "no meta-training"
rows and the Warp-Leap row.  So SCRATCH-K is not a nicety, it is the quantity
being claimed.  Note the paper gave its no-meta-training baselines a 4x larger
batch and a 10x larger learning rate to keep the comparison fair on compute:

    "To render no-pretraining a competitive option within a fair computational
     budget, we allow SGD and KFAC to use 4x larger batch sizes, enabling 10x
     larger learning rates."                             -- Appendix E, page 20

We do the same, with `--scratch-lr-mult` and `--scratch-batch-mult`, and we
report the tuned-over-that-grid best, so the baseline is as strong as we can
reasonably make it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from llm.bracket import build_floors
from llm.data import META_TEST, ByteSampler, build_corpus
from llm.meta import evaluate_bpb, functional_loss
from llm.model import GPTConfig, WarpedGPT


def train_oracle(cfg, tr, va, steps, batch_size, lr, device, amp_dtype,
                 eval_batches=24, log_every=500, seed=0):
    """Train from scratch on ONE language with Adam.  Returns the best val bpb."""
    torch.manual_seed(seed)
    model = WarpedGPT(cfg).to(device)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is not None)

    best, curve = float("inf"), []
    for k in range(steps):
        x, y = tr.batch(batch_size)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if (k + 1) % log_every == 0 or k == steps - 1:
            model.eval()
            bpb = evaluate_bpb(model, va, params, {}, buffers,
                               batch_size, eval_batches, amp_dtype)
            model.train()
            best = min(best, bpb)
            curve.append((k + 1, bpb))
            print(f"      oracle step {k + 1:5d}  val bpb {bpb:6.3f}"
                  f"  (best {best:6.3f})", flush=True)
    return best, curve


def scratch_k(cfg, tr, va, steps, batch_size, lr, device, amp_dtype,
              eval_batches=24, seed=0):
    """Random init, then exactly the meta-learners' inner loop.  No meta-learning."""
    torch.manual_seed(seed)
    model = WarpedGPT(cfg).to(device)
    theta = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    for _ in range(steps):
        x, y = tr.batch(batch_size)
        for p in theta.values():
            p.requires_grad_(True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = functional_loss(model, theta, {}, x, y, buffers)
        g = torch.autograd.grad(loss, list(theta.values()), allow_unused=True)
        with torch.no_grad():
            theta = {n: (p - lr * gi).detach() if gi is not None else p.detach()
                     for (n, p), gi in zip(theta.items(), g)}
    return evaluate_bpb(model, va, theta, {}, buffers, batch_size,
                        eval_batches, amp_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None, help="default: META_TEST")
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--oracle-steps", type=int, default=3000)
    ap.add_argument("--oracle-batch", type=int, default=32)
    ap.add_argument("--oracle-lr", type=float, default=3e-4)
    ap.add_argument("--inner-steps", type=int, default=100)
    ap.add_argument("--inner-lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scratch-lr-mult", type=float, nargs="+",
                    default=[1.0, 3.0, 10.0])
    ap.add_argument("--scratch-batch-mult", type=int, default=4)
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--out", default="results/llm/ceilings.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    amp_dtype = None if a.fp32 else torch.float16
    langs = a.langs or META_TEST
    corpus, _ = build_corpus(a.data_dir, verbose=False)
    langs = [l for l in langs if l in corpus]
    floors = build_floors(corpus, verbose=False)

    cfg = GPTConfig(n_layer=a.n_layer, d_model=a.d_model, n_head=a.n_head,
                    block_size=a.block_size, warp_kind="identity", dropout=0.0)

    out = {}
    for l in langs:
        print(f"\n=== {l} ({corpus[l].script}) " + "=" * 40, flush=True)
        tr = ByteSampler(corpus[l].train, cfg.block_size, a.device, seed=11)
        va = ByteSampler(corpus[l].val, cfg.block_size, a.device, seed=12)
        t0 = time.time()

        ceiling, curve = train_oracle(cfg, tr, va, a.oracle_steps, a.oracle_batch,
                                      a.oracle_lr, a.device, amp_dtype)

        # SCRATCH-K, tuned over the paper's own generosity grid
        best_scratch, best_cfg = float("inf"), None
        for mult in a.scratch_lr_mult:
            for bm in (1, a.scratch_batch_mult):
                s = scratch_k(cfg, tr, va, a.inner_steps, a.batch_size * bm,
                              a.inner_lr * mult, a.device, amp_dtype)
                if s < best_scratch:
                    best_scratch, best_cfg = s, dict(lr_mult=mult, batch_mult=bm)
        f = floors[l]
        span = f["bigram"] - ceiling
        out[l] = dict(script=f["script"], unigram=f["unigram"], bigram=f["bigram"],
                      oracle=ceiling, oracle_curve=curve,
                      scratch_k=best_scratch, scratch_cfg=best_cfg,
                      span=span, minutes=(time.time() - t0) / 60)
        print(f"  bigram floor {f['bigram']:6.3f} | oracle ceiling {ceiling:6.3f}"
              f" | span {span:6.3f} | scratch-{a.inner_steps} {best_scratch:6.3f}"
              f" {best_cfg}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)

    # ------------------------------------------------------------- the verdict
    print("\n" + "=" * 78)
    print(f"{'lang':<6}{'script':<12}{'bigram':>8}{'oracle':>8}{'span':>8}"
          f"{'scratch':>9}{'usable?':>10}")
    print("-" * 78)
    usable = []
    for l, r in out.items():
        ok = r["span"] >= 0.15
        usable.append(ok)
        print(f"{l:<6}{r['script']:<12}{r['bigram']:>8.3f}{r['oracle']:>8.3f}"
              f"{r['span']:>8.3f}{r['scratch_k']:>9.3f}"
              f"{'yes' if ok else 'NO':>10}")
    print("-" * 78)
    print("span = bigram floor - oracle ceiling.  It is the ENTIRE range a")
    print("meta-learner has to work in on that language.  A language with a")
    print("narrow span cannot distinguish between methods and is excluded.")
    n_ok = sum(usable)
    if n_ok >= max(3, len(out) // 2):
        print(f"\nVERDICT: PROCEED.  {n_ok}/{len(out)} held-out languages are usable.")
    else:
        print(f"\nVERDICT: REFUSE.  only {n_ok}/{len(out)} languages are usable.")
        print("  The model or the data budget is too small for any meta-learner")
        print("  to show a difference here.  Widen the bracket before sweeping:")
        print("  more capacity, more adaptation steps, or a larger batch.")
    print("=" * 78)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
