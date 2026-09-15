"""One detached pipeline: corpus -> ceilings -> Phase A -> Phase B -> summary.

WHY THIS FILE EXISTS
====================
The first attempt at the language sweep ran interactively from notebook cells.
The Colab session was then reset, which wiped /content, and roughly two hours of
finished GPU work went with it because the results had only ever existed on that
machine.

Two changes, both cheap, both learned the hard way:

  1. EVERYTHING RUNS FROM ONE DETACHED PROCESS.  Launch it once, and it survives
     the notebook kernel dying, the MCP connection dropping, and the browser
     being closed.  It does not survive the VM being recycled, but nothing does.

  2. EVERY STAGE IS SKIPPABLE AND EVERY RESULT IS SMALL.  Each stage checks for
     its own output file first, so a re-launch after any interruption resumes
     instead of restarting.  A compact STATUS.json is rewritten after every
     stage, so progress can be polled with a one-line cell that cannot itself
     time out.

The results files are a few kilobytes each, so they can be read back out through
the notebook and committed to git from the developer machine.  Never leave the
only copy of a finished result on ephemeral storage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = "/content/warpgrad-llm"
STATUS = os.path.join(ROOT, "results", "STATUS.json")


def write_status(stage, detail=""):
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    prev = {}
    if os.path.exists(STATUS):
        try:
            prev = json.load(open(STATUS))
        except Exception:
            prev = {}
    prev.setdefault("log", []).append(
        dict(t=time.strftime("%H:%M:%S"), stage=stage, detail=detail))
    prev["stage"] = stage
    prev["detail"] = detail
    prev["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(prev, open(STATUS, "w"), indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] STAGE {stage}: {detail}", flush=True)


def ceilings_complete():
    """True only when every held-out language has an oracle ceiling.

    Returns False for a missing file, a corrupt file, or a partial one.  The
    partial case is the one that matters: it looks exactly like success to an
    existence check.
    """
    path = os.path.join(ROOT, "results", "llm", "ceilings.json")
    if not os.path.exists(path):
        return False
    try:
        got = json.load(open(path))
        sys.path.insert(0, ROOT)
        from llm.data import META_TEST
        manifest = json.load(open("/content/data/manifest.json"))
        want = [l for l in META_TEST if l in manifest.get("meta_test", META_TEST)]
        missing = [l for l in want if l not in got]
        if missing:
            print(f"    ceilings.json is PARTIAL, missing {missing}", flush=True)
            return False
        return True
    except Exception as e:
        print(f"    ceilings.json unreadable ({type(e).__name__}), redoing", flush=True)
        return False


def run(cmd, tag, check_file=None):
    """Run a stage.  When `check_file` is given, that file existing is the real
    success criterion, not the exit code.

    This matters because the `datasets` library aborts at interpreter shutdown
    on this image:

        terminate called without an active exception   (SIGABRT, exit -6)

    That happens AFTER every byte has been written to disk, so the exit code
    reports a failure that did not occur.  Judging a stage by its output rather
    than its return code is the robust thing to do anyway: the question is
    whether the artefact exists, not how the process felt about exiting.
    """
    print(f"\n{'=' * 70}\n>>> {tag}\n{'=' * 70}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    produced = bool(check_file) and os.path.exists(
        check_file if os.path.isabs(check_file) else os.path.join(ROOT, check_file))
    if r.returncode != 0 and not produced:
        write_status("FAILED", f"{tag} exited {r.returncode}")
        sys.exit(r.returncode or 1)
    if r.returncode != 0:
        print(f"    (exit {r.returncode} ignored: {check_file} was written)",
              flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ladder-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--oracle-steps", type=int, default=3000)
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()

    os.chdir(ROOT)
    t0 = time.time()

    # ---------------------------------------------------------------- corpus
    write_status("corpus", "streaming 32 Wikipedia languages")
    if not os.path.exists("/content/data/manifest.json"):
        run([sys.executable, "-c",
             "import sys, os; sys.path.insert(0,'.');"
             "from llm.data import build_corpus;"
             "build_corpus('/content/data', train_mb=2.0, val_mb=0.5, verbose=True);"
             "sys.stdout.flush(); os._exit(0)"],
            "build corpus", check_file="/content/data/manifest.json")
    write_status("corpus", "done")

    # -------------------------------------------------------------- ceilings
    # GATE 2.  Must pass before any sweep: if the span between the bigram floor
    # and the from-scratch oracle is narrow, no meta-learner can show anything.
    #
    # "Exists" is NOT the right completeness test here.  llm/oracle.py rewrites
    # ceilings.json after every language so a long run is not lost, which means a
    # partially finished oracle leaves a file containing two languages out of
    # seven.  Skipping on existence would silently sweep against a ceiling set
    # that is missing most of the held-out languages.  Check the contents.
    if not ceilings_complete():
        write_status("ceilings", f"oracle at {a.oracle_steps} steps per language")
        run([sys.executable, "-u", "llm/oracle.py",
             "--oracle-steps", str(a.oracle_steps)], "oracle ceilings")
        if not ceilings_complete():
            write_status("FAILED", "oracle finished but ceilings.json is incomplete")
            sys.exit(1)
    write_status("ceilings", "done")

    # --------------------------------------------------------------- phase A
    write_status("phaseA", f"4 arms x {len(a.seeds)} seeds, C1 and C2")
    run([sys.executable, "-u", "scripts/run_sweep.py", "--phases", "A",
         "--seeds", *map(str, a.seeds), "--meta-steps", str(a.meta_steps),
         "--max-minutes", str(a.max_minutes), "--gpus", "0", "1"], "phase A")
    write_status("phaseA", "done")

    # --------------------------------------------------------------- phase B
    if not a.skip_b:
        write_status("phaseB", f"capacity ladder, C4, seeds {a.ladder_seeds}")
        run([sys.executable, "-u", "scripts/run_sweep.py", "--phases", "B",
             "--seeds", *map(str, a.ladder_seeds), "--meta-steps", str(a.meta_steps),
             "--max-minutes", str(a.max_minutes), "--gpus", "0", "1"], "phase B")
        write_status("phaseB", "done")

    # ------------------------------------------------------------------- C5
    ck = "results/llm/A_warp_leap_linear_s0.pt"
    if os.path.exists(ck):
        write_status("fisher", "C5, is the learned geometry Fisher-like")
        subprocess.run([sys.executable, "-u", "llm/fisher_check.py",
                        "--ckpt", ck], cwd=ROOT)
    write_status("fisher", "done")

    # -------------------------------------------------------------- analysis
    write_status("analyse", "building tables and figures")
    subprocess.run([sys.executable, "-u", "scripts/analyse.py"], cwd=ROOT)
    write_status("DONE", f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
