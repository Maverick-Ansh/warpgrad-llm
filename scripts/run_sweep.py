"""Launch the language sweep across both T4s, detached, resumable.

One process per GPU.  Running two processes on one T4 gives no throughput gain
and roughly a 2.7x per-step slowdown, so the scheduler here keeps exactly one
job in flight per device.

Every run writes results/llm/<tag>.json and llm/train.py skips a tag whose file
already exists, so a killed sweep resumes instead of restarting.

PHASE A tests C1 and C2, the abstract's claims.  Four arms, identical in every
respect except what is meta-learned:

    warp_leap  geometry (phi) AND initialisation (theta0)
    leap       initialisation only, warp-layers are literally absent
    reptile    initialisation only, by moving it toward where training ended
    joint      no meta-learning at all, plain multi-task training of theta0

`leap` is the arm that isolates the contribution of the warp, because it shares
warp_leap's initialisation objective exactly and differs only by having no
geometry.  The distance between those two IS claim C1.

PHASE B tests C4, the capacity ladder of Table 3, ported to a transformer.
The prediction under test is that the ORDER survives the port, and in particular
that "mlp" (two layers, input-dependent Jacobian, beyond block-diagonal) beats
"linear" (constant Jacobian, block-diagonal, the T-Net equivalent).
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

COMMON = [
    "--n-layer", "6", "--d-model", "256", "--n-head", "8", "--block-size", "256",
    "--inner-steps", "100", "--inner-lr", "0.1", "--batch-size", "16",
    "--meta-batch", "5", "--buffer-stride", "5", "--eval-batches", "24",
]


def phase_a(seeds, meta_steps):
    jobs = []
    for seed, (arm, kind) in itertools.product(
        seeds,
        [("warp_leap", "linear"), ("leap", "identity"),
         ("reptile", "identity"), ("joint", "identity")],
    ):
        tag = f"A_{arm}_{kind}_s{seed}"
        jobs.append((tag, ["--arm", arm, "--warp-kind", kind,
                           "--seed", str(seed), "--meta-steps", str(meta_steps),
                           "--tag", tag]))
    return jobs


def phase_b(seeds, meta_steps):
    jobs = []
    for seed, kind in itertools.product(
        seeds, ["scale", "lowrank", "norm_act", "residual", "mlp"]
    ):
        tag = f"B_warp_leap_{kind}_s{seed}"
        jobs.append((tag, ["--arm", "warp_leap", "--warp-kind", kind,
                           "--seed", str(seed), "--meta-steps", str(meta_steps),
                           "--tag", tag]))
    return jobs


def phase_c(seeds, meta_steps):
    """C6 (offline vs online) and C7 (Eq. 11 vs Eq. 12)."""
    jobs = []
    for seed in seeds:
        jobs.append((f"C_online_s{seed}",
                     ["--arm", "warp_leap", "--warp-kind", "linear",
                      "--algorithm", "online", "--seed", str(seed),
                      "--meta-steps", str(meta_steps), "--tag", f"C_online_s{seed}"]))
        jobs.append((f"C_exact_s{seed}",
                     ["--arm", "warp_leap", "--warp-kind", "linear",
                      "--exact", "--seed", str(seed),
                      "--meta-steps", str(meta_steps), "--tag", f"C_exact_s{seed}"]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["A"], choices=["A", "B", "C"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--meta-steps", type=int, default=120)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--logdir", default="results/llm/logs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.logdir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    builders = {"A": phase_a, "B": phase_b, "C": phase_c}
    jobs = []
    for p in a.phases:
        jobs += builders[p](a.seeds, a.meta_steps)
    jobs = [(t, c) for t, c in jobs
            if not os.path.exists(os.path.join(a.out, f"{t}.json"))]

    print(f"{len(jobs)} job(s) to run on GPUs {a.gpus}")
    for t, _ in jobs:
        print("   ", t)
    if a.dry_run or not jobs:
        return

    running, queue, t0 = {}, list(jobs), time.time()
    while queue or running:
        for g in a.gpus:
            if g in running:
                continue
            if not queue:
                continue
            tag, extra = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log = open(os.path.join(a.logdir, f"{tag}.log"), "w")
            cmd = ([sys.executable, "-u", "llm/train.py"] + COMMON + extra
                   + ["--out", a.out, "--max-minutes", str(a.max_minutes)])
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
            running[g] = (tag, p, log)
            print(f"[{(time.time() - t0) / 60:6.1f}m] GPU{g} START {tag}", flush=True)

        for g, (tag, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"[{(time.time() - t0) / 60:6.1f}m] GPU{g} DONE  {tag}  {ok}",
                      flush=True)
                del running[g]

        time.sleep(10)

    print(f"all jobs finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
