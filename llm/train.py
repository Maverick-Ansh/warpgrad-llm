"""Meta-train one arm on the language task distribution, then meta-test it.

This is Algorithm 2 (offline meta-training) by default, because the paper found
it decisively better than Algorithm 1 on the benchmark we are porting:

    "The gains of the offline variant can be dramatic: in our Omniglot
     experiment (Section 4.1), offline meta-training allows us to update warp
     parameters 2000 times with each meta-batch, improving final test accuracy
     from 76.3% to 84.3%"
                                                        -- Section 2.5, page 7

`--algorithm online` runs Algorithm 1 instead, which is claim C6.

WHAT MEASUREMENT DECIDES EVERYTHING
===================================
Meta-test is: take the meta-learned theta0 and phi, freeze phi, run exactly the
same 100-step SGD adaptation on a language the meta-learner has NEVER seen, and
measure bits per byte on that language's held-out validation stream.

That number is then bracketed per language against its own smoothed-bigram floor
and its own from-scratch oracle ceiling (llm/bracket.py), because raw bits per
byte is not comparable across writing systems.  Nothing in this file reports an
unbracketed average.

DEVIATIONS FROM THE PAPER, ALL DELIBERATE
=========================================
  buffer stride     The paper stores all 2000 iterates of a meta-batch.  At
                    ~5M parameters that is ~40 GB, so we store every `stride`-th
                    iterate.  Eq. 11 is an expectation over p(theta|tau) and the
                    iterates are Monte-Carlo samples of it, so subsampling costs
                    variance and not bias.  See Trajectory in meta.py.
  fp16 + GradScaler Tesla T4 is compute capability 7.5, which has fast fp16 and
                    no bf16.  Never bf16 on this hardware.
  Eq. 12 default    See the long note in meta.py.  `--exact` uses Eq. 11 in fp32.
  LayerNorm         stands in for the paper's BatchNorm inside warp-layers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from llm.bracket import build_floors, normalised_score
from llm.data import META_TEST, META_TRAIN, ByteSampler, build_corpus
from llm.meta import MetaLearner, evaluate_bpb
from llm.model import GPTConfig, WarpedGPT


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_samplers(corpus, langs, block_size, device, seed=0, split="train"):
    out = {}
    for i, l in enumerate(langs):
        arr = corpus[l].train if split == "train" else corpus[l].val
        s = ByteSampler(arr, block_size, device=device, seed=seed * 1000 + i)
        s.lang = l
        out[l] = s
    return out


def meta_test(learner, corpus, langs, cfg, args, floors, verbose=True):
    """Adapt to each held-out language from scratch and score it, bracketed."""
    rows = {}
    for l in langs:
        tr = ByteSampler(corpus[l].train, cfg.block_size, device=args.device, seed=7)
        va = ByteSampler(corpus[l].val, cfg.block_size, device=args.device, seed=8)
        tr.lang = va.lang = l

        before = evaluate_bpb(learner.model, va, learner.theta0, learner.phi,
                              learner.buffers, args.batch_size, args.eval_batches,
                              amp_dtype=learner.amp_dtype)
        traj, _ = learner.adapt(tr, args.batch_size, steps=args.inner_steps,
                                collect=False, track_leap=False)
        after = evaluate_bpb(learner.model, va, traj.final, learner.phi,
                             learner.buffers, args.batch_size, args.eval_batches,
                             amp_dtype=learner.amp_dtype)
        rows[l] = dict(
            script=corpus[l].script, bpb_before=before, bpb_after=after,
            bigram_floor=floors[l]["bigram"], unigram_floor=floors[l]["unigram"],
            beats_bigram=bool(after < floors[l]["bigram"]),
            adaptation_curve=traj.losses[::5],
        )
        if verbose:
            print(f"    {l:<5}{rows[l]['script']:<11} bpb {before:6.3f} -> {after:6.3f}"
                  f"   bigram floor {floors[l]['bigram']:6.3f}"
                  f"   {'BEATS' if rows[l]['beats_bigram'] else 'below'} floor",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="warp_leap",
                    choices=["warp_leap", "leap", "reptile", "joint"])
    ap.add_argument("--warp-kind", default="linear")
    ap.add_argument("--algorithm", default="offline", choices=["offline", "online"])
    ap.add_argument("--exact", action="store_true", help="Eq. 11 in fp32, not Eq. 12")
    # model
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--warp-every", type=int, default=1)
    # meta
    ap.add_argument("--meta-steps", type=int, default=300)
    ap.add_argument("--meta-batch", type=int, default=5)
    ap.add_argument("--inner-steps", type=int, default=100)
    ap.add_argument("--inner-lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--theta-lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--buffer-stride", type=int, default=5)
    ap.add_argument("--eta", type=int, default=1, help="phi updates per buffer item")
    # eval / io
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=24)
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--train-mb", type=float, default=2.0)
    ap.add_argument("--val-mb", type=float, default=0.5)
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"{args.arm}_{args.warp_kind}_{args.algorithm}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    out_path = f"{args.out}/{tag}.json"
    if os.path.exists(out_path):
        print(f"[{tag}] already done, skipping (delete {out_path} to rerun)")
        return

    set_seed(args.seed)
    amp_dtype = None if (args.fp32 or args.exact) else torch.float16

    print(f"[{tag}] building corpus")
    corpus, manifest = build_corpus(args.data_dir, train_mb=args.train_mb,
                                    val_mb=args.val_mb, verbose=False)
    train_langs = [l for l in META_TRAIN if l in corpus]
    test_langs = [l for l in META_TEST if l in corpus]
    print(f"[{tag}] meta-train {len(train_langs)} langs, meta-test {len(test_langs)}")

    floors = build_floors(corpus, out_path=f"{args.out}/floors.json", verbose=False)

    # An arm with no geometry must literally have no warp-layers, or the
    # "no warp" control is not a control.  meta.py asserts this too.
    warp_kind = args.warp_kind if args.arm == "warp_leap" else "identity"
    cfg = GPTConfig(n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head,
                    block_size=args.block_size, warp_kind=warp_kind,
                    warp_every=args.warp_every, dropout=0.0)
    model = WarpedGPT(cfg).to(args.device)
    counts = model.n_params()
    print(f"[{tag}] params  task {counts['task'] / 1e6:.2f}M  "
          f"warp {counts['warp'] / 1e6:.3f}M  total {counts['total'] / 1e6:.2f}M")

    learner = MetaLearner(model, arm=args.arm, inner_lr=args.inner_lr,
                          inner_steps=args.inner_steps, lam=args.lam,
                          exact=args.exact, device=args.device, amp_dtype=amp_dtype)

    opt_phi = (torch.optim.Adam(list(learner.phi.values()), lr=args.meta_lr)
               if learner.phi else None)
    opt_theta = torch.optim.Adam(list(learner.theta0.values()), lr=args.theta_lr)

    samplers = make_samplers(corpus, train_langs, cfg.block_size, args.device,
                             seed=args.seed)
    hist = {"meta_step": [], "mean_task_loss": [], "meta_loss": [], "wallclock": [],
            "eval": []}
    t_start = time.time()

    for m in range(args.meta_steps):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"[{tag}] hit --max-minutes at meta-step {m}, stopping cleanly")
            break

        batch_langs = random.sample(train_langs, min(args.meta_batch, len(train_langs)))
        trajs, leap_accs = [], []

        # --- Algorithm 2, lines 5-11: collect trajectories -------------------
        # `joint` never adapts during meta-training (that is what makes it the
        # no-meta-learning control), so running the inner loop for it would
        # double its cost for nothing.  It still adapts at meta-TEST time, like
        # every other arm.
        for l in (batch_langs if args.arm != "joint" else []):
            collect = args.algorithm == "offline" and learner.phi
            traj, acc = learner.adapt(
                samplers[l], args.batch_size, steps=args.inner_steps,
                stride=args.buffer_stride, collect=collect,
                track_leap=(args.arm in ("warp_leap", "leap")),
            )
            trajs.append(traj)
            leap_accs.append(acc)

        # --- Algorithm 2, lines 12-23: mini-batched meta-updates on phi ------
        meta_loss, n_meta = 0.0, 0
        if learner.phi:
            items = [(t, p) for t in trajs for p in t.points]
            random.shuffle(items)              # "sample tau,k without replacement"
            if opt_phi is not None:
                opt_phi.zero_grad(set_to_none=True)
            for i, (t, point) in enumerate(items, 1):
                meta_loss += learner.warp_meta_grad(point, samplers[t.lang],
                                                    args.batch_size)
                n_meta += 1
                if i % args.eta == 0:
                    torch.nn.utils.clip_grad_norm_(list(learner.phi.values()), 1.0)
                    opt_phi.step()
                    opt_phi.zero_grad(set_to_none=True)

        # --- update theta0 ----------------------------------------------------
        if args.arm == "reptile":
            learner.reptile_step(trajs, args.theta_lr * 100)
        elif args.arm in ("warp_leap", "leap"):
            opt_theta.zero_grad(set_to_none=True)
            for t in trajs:
                learner.leap_grad(t, None)
            for p in learner.theta0.values():
                if p.grad is not None:
                    p.grad.div_(len(trajs))
            torch.nn.utils.clip_grad_norm_(list(learner.theta0.values()), 1.0)
            opt_theta.step()
        elif args.arm == "joint":
            # Plain multi-task training of theta0: the "Finetuning" row of
            # Table 1, and the no-meta-learning control.
            #
            # FAIRNESS.  It must get a COMPUTE budget comparable to the other
            # arms, or it is a strawman and every C1 number is inflated.  Each
            # meta-step of Warp-Leap costs meta_batch * inner_steps forward and
            # backward passes for adaptation, plus the meta-gradient passes.  An
            # earlier version of this file gave `joint` a SINGLE gradient step
            # per meta-step, which is roughly 500x less signal, and it would have
            # lost for that reason alone.
            #
            # So `joint` now takes `inner_steps` gradient steps per meta-step, on
            # batches drawn from randomly chosen meta-training languages, which
            # is exactly what ordinary multi-task pretraining looks like.
            from llm.meta import functional_loss
            for _ in range(args.inner_steps):
                opt_theta.zero_grad(set_to_none=True)
                for l in random.sample(train_langs,
                                       min(args.meta_batch, len(train_langs))):
                    x, y = samplers[l].batch(args.batch_size)
                    for p in learner.theta0.values():
                        p.requires_grad_(True)
                    with torch.autocast("cuda", dtype=amp_dtype,
                                        enabled=amp_dtype is not None):
                        loss = functional_loss(model, learner.theta0, learner.phi,
                                               x, y, learner.buffers)
                    (loss / args.meta_batch).backward()
                torch.nn.utils.clip_grad_norm_(list(learner.theta0.values()), 1.0)
                opt_theta.step()

        mean_tl = (float(np.mean([np.mean(t.losses[-10:]) for t in trajs]))
                   if trajs else float("nan"))
        hist["meta_step"].append(m)
        hist["mean_task_loss"].append(mean_tl)
        hist["meta_loss"].append(meta_loss / max(n_meta, 1))
        hist["wallclock"].append(time.time() - t_start)

        if m % 10 == 0 or m == args.meta_steps - 1:
            print(f"[{tag}] meta {m:4d}  task_loss {mean_tl:7.4f}"
                  f"  meta_loss {hist['meta_loss'][-1]:7.4f}"
                  f"  phi_updates {n_meta:4d}"
                  f"  {(time.time() - t_start) / 60:5.1f}min", flush=True)

        if args.eval_every and (m + 1) % args.eval_every == 0:
            print(f"[{tag}] --- meta-test at step {m + 1} ---", flush=True)
            rows = meta_test(learner, corpus, test_langs, cfg, args, floors)
            hist["eval"].append(dict(meta_step=m + 1, rows=rows))

    print(f"[{tag}] === FINAL meta-test ===", flush=True)
    final_rows = meta_test(learner, corpus, test_langs, cfg, args, floors)

    rec = dict(
        tag=tag, args=vars(args), params=counts,
        meta_train_langs=train_langs, meta_test_langs=test_langs,
        history=hist, final=final_rows,
        wallclock_min=(time.time() - t_start) / 60,
    )
    json.dump(rec, open(out_path, "w"), indent=2)
    torch.save({"theta0": {k: v.detach().cpu() for k, v in learner.theta0.items()},
                "phi": {k: v.detach().cpu() for k, v in learner.phi.items()},
                "cfg": vars(cfg)}, f"{args.out}/{tag}.pt")
    print(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
