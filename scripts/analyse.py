"""Turn results/llm/*.json into the tables and figures that go in REPORT.md.

Every number printed here is a NORMALISED score:

    score = (bigram_floor - achieved_bpb) / (bigram_floor - oracle_ceiling)

    1.0   matched a model trained on that language from scratch at 30x the
          adaptation budget
    0.0   no better than a smoothed bigram lookup table fit on that language
   <0.0   worse than that lookup table

Both ends are measured on that language's own data (llm/bracket.py,
llm/oracle.py), which is what makes the score comparable across writing systems.
Raw bits per byte is not, because UTF-8 spends 1 to 3 bytes per character
depending on the script.

THE SPLIT THAT MATTERS
======================
Held-out languages are reported in two groups, never pooled into one mean:

  SCRIPT SEEN     uk ca fa mr   new language, writing system present in
                                meta-training
  SCRIPT UNSEEN   ka ta ko      new language AND a writing system that appears
                                nowhere in meta-training

If a meta-learned geometry transfers only within a script, those groups
separate.  Pooling them would hide exactly the thing worth knowing, and it is a
question the paper's Omniglot setup cannot ask because all its alphabets are
rendered identically.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

SCRIPT_SEEN = ["uk", "ca", "fa", "mr"]
SCRIPT_UNSEEN = ["ka", "ta", "ko"]

ARM_LABEL = {
    "warp_leap": "Warp-Leap", "leap": "Leap", "reptile": "Reptile",
    "joint": "Joint (no meta)",
}


def load(results_dir):
    ceilings = json.load(open(os.path.join(results_dir, "ceilings.json")))
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(p)
        if base in ("ceilings.json", "floors.json") or base.startswith(("SMOKE", "TIMING")):
            continue
        r = json.load(open(p))
        if "final" not in r:
            continue
        runs.append(r)
    return runs, ceilings


def score_run(run, ceilings):
    """Per-language normalised score for one run."""
    out = {}
    for lang, row in run["final"].items():
        c = ceilings.get(lang)
        if c is None:
            continue
        floor, ceil = row["bigram_floor"], c["oracle"]
        span = floor - ceil
        if span <= 1e-9:
            continue
        out[lang] = dict(
            score=(floor - row["bpb_after"]) / span,
            bpb=row["bpb_after"],
            bpb_before=row["bpb_before"],
            script=row["script"],
            beats_bigram=row["beats_bigram"],
        )
    return out


def group_stats(per_lang, langs):
    v = [per_lang[l]["score"] for l in langs if l in per_lang]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"),) * 2 + (0,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/llm")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--out", default="results/llm/summary.json")
    a = ap.parse_args()

    runs, ceilings = load(a.results)
    if not runs:
        print(f"no runs found in {a.results}")
        return
    print(f"loaded {len(runs)} run(s)\n")

    # ------------------------------------------------------------ scratch ref
    scratch = {}
    for l, c in ceilings.items():
        span = c["bigram"] - c["oracle"]
        scratch[l] = (c["bigram"] - c["scratch_k"]) / span if span > 0 else float("nan")

    # ------------------------------------------------------------------ table
    by_cfg = defaultdict(list)
    for r in runs:
        args = r["args"]
        key = (args["arm"], args["warp_kind"], args["algorithm"], bool(args["exact"]))
        by_cfg[key].append(score_run(r, ceilings))

    rows = []
    for key, per_lang_list in by_cfg.items():
        arm, kind, algo, exact = key
        seen_m, unseen_m, all_m = [], [], []
        for pl in per_lang_list:
            seen_m.append(group_stats(pl, SCRIPT_SEEN)[0])
            unseen_m.append(group_stats(pl, SCRIPT_UNSEEN)[0])
            all_m.append(group_stats(pl, SCRIPT_SEEN + SCRIPT_UNSEEN)[0])
        n = len(per_lang_list)
        rows.append(dict(
            arm=arm, warp_kind=kind, algorithm=algo, exact=exact, n_seeds=n,
            all_mean=float(np.mean(all_m)),
            all_sem=float(np.std(all_m) / max(np.sqrt(n), 1)),
            seen_mean=float(np.mean(seen_m)),
            unseen_mean=float(np.mean(unseen_m)),
            per_lang=per_lang_list,
        ))
    rows.sort(key=lambda r: -r["all_mean"])

    sref = np.mean([scratch[l] for l in SCRIPT_SEEN + SCRIPT_UNSEEN])
    print("=" * 84)
    print("NORMALISED SCORE on held-out languages")
    print("  1.0 = from-scratch oracle at 30x budget | 0.0 = smoothed bigram table")
    print("=" * 84)
    print(f"{'arm':<16}{'warp':<10}{'algo':<9}{'seeds':>6}"
          f"{'all':>10}{'sem':>8}{'seen':>9}{'unseen':>9}")
    print("-" * 84)
    for r in rows:
        print(f"{ARM_LABEL.get(r['arm'], r['arm']):<16}{r['warp_kind']:<10}"
              f"{r['algorithm']:<9}{r['n_seeds']:>6}"
              f"{r['all_mean']:>10.4f}{r['all_sem']:>8.4f}"
              f"{r['seen_mean']:>9.4f}{r['unseen_mean']:>9.4f}")
    print("-" * 84)
    print(f"{'SCRATCH-100':<16}{'-':<10}{'none':<9}{'-':>6}{sref:>10.4f}"
          f"{'-':>8}{np.mean([scratch[l] for l in SCRIPT_SEEN]):>9.4f}"
          f"{np.mean([scratch[l] for l in SCRIPT_UNSEEN]):>9.4f}")
    print(f"{'BIGRAM TABLE':<16}{'-':<10}{'none':<9}{'-':>6}{0.0:>10.4f}")
    print("=" * 84)

    # ------------------------------------------------- the claim C1 comparison
    def find(arm, kind=None):
        for r in rows:
            if r["arm"] == arm and (kind is None or r["warp_kind"] == kind) \
                    and r["algorithm"] == "offline" and not r["exact"]:
                return r
        return None

    wl, lp = find("warp_leap", "linear"), find("leap")
    if wl and lp:
        d = wl["all_mean"] - lp["all_mean"]
        sem = float(np.sqrt(wl["all_sem"] ** 2 + lp["all_sem"] ** 2))
        print("\nC1  Warp-Leap minus Leap (the arm that isolates the warp)")
        print(f"      all scripts     {d:+.4f} +/- {sem:.4f}"
              f"   ({d / max(sem, 1e-9):+.1f} sem)")
        print(f"      script seen     {wl['seen_mean'] - lp['seen_mean']:+.4f}")
        print(f"      script unseen   {wl['unseen_mean'] - lp['unseen_mean']:+.4f}")
        if abs(d) < 2 * sem:
            print("      NOT separable from zero at 2 sem with this many seeds.")

    # ---------------------------------------------------------------- figures
    os.makedirs(a.figdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # per-language bars, grouped by whether the script was seen
        arms = [r for r in rows if r["algorithm"] == "offline" and not r["exact"]
                and r["warp_kind"] in ("linear", "identity")]
        if arms:
            langs = SCRIPT_SEEN + SCRIPT_UNSEEN
            fig, ax = plt.subplots(figsize=(11, 4.6))
            w = 0.8 / max(len(arms), 1)
            for i, r in enumerate(arms):
                vals = [np.mean([pl[l]["score"] for pl in r["per_lang"] if l in pl])
                        if any(l in pl for pl in r["per_lang"]) else np.nan
                        for l in langs]
                ax.bar(np.arange(len(langs)) + i * w, vals, w,
                       label=ARM_LABEL.get(r["arm"], r["arm"]))
            ax.axhline(0, color="#333", lw=1)
            ax.plot(np.arange(len(langs)) + 0.4 - w / 2,
                    [scratch[l] for l in langs], "k*--", ms=9, lw=1,
                    label="scratch-100 (no meta)")
            ax.axvline(len(SCRIPT_SEEN) - 0.5 + 0.4 - w / 2, color="#999", ls=":")
            ax.text(len(SCRIPT_SEEN) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
                    "script SEEN in meta-training", ha="center", fontsize=9,
                    color="#555")
            ax.text(len(SCRIPT_SEEN) + len(SCRIPT_UNSEEN) / 2 - 0.5,
                    ax.get_ylim()[1] * 0.95, "script UNSEEN", ha="center",
                    fontsize=9, color="#555")
            ax.set_xticks(np.arange(len(langs)) + 0.4 - w / 2)
            ax.set_xticklabels([f"{l}\n{ceilings[l]['script']}" for l in langs],
                               fontsize=8)
            ax.set_ylabel("normalised score\n(1 = oracle, 0 = bigram table)")
            ax.set_title("Adaptation to held-out languages in 100 SGD steps")
            ax.legend(fontsize=8, ncol=3)
            fig.tight_layout()
            fig.savefig(f"{a.figdir}/llm_per_language.png", dpi=150)
            plt.close(fig)
            print(f"\nwrote {a.figdir}/llm_per_language.png")

        # capacity ladder (C4)
        ladder = ["scale", "lowrank", "linear", "norm_act", "residual", "mlp"]
        lr_rows = [(k, r) for k in ladder for r in rows
                   if r["arm"] == "warp_leap" and r["warp_kind"] == k]
        base = find("leap")
        if len(lr_rows) >= 3:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            xs = list(range(len(lr_rows)))
            ax.errorbar(xs, [r["all_mean"] for _, r in lr_rows],
                        yerr=[r["all_sem"] for _, r in lr_rows],
                        marker="o", color="#E0218A", capsize=3)
            if base:
                ax.axhline(base["all_mean"], color="#111", ls="--", lw=1,
                           label="no warp (Leap)")
            ax.set_xticks(xs)
            ax.set_xticklabels([k for k, _ in lr_rows], rotation=20, fontsize=9)
            ax.set_ylabel("normalised score")
            ax.set_title("C4: warp capacity ladder (Table 3 ported to a transformer)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{a.figdir}/llm_capacity_ladder.png", dpi=150)
            plt.close(fig)
            print(f"wrote {a.figdir}/llm_capacity_ladder.png")

        # adaptation curves
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for r in runs:
            args = r["args"]
            if args["algorithm"] != "offline" or args["exact"]:
                continue
            curves = [v["adaptation_curve"] for v in r["final"].values()]
            if not curves:
                continue
            m = np.mean(np.array([c[:min(map(len, curves))] for c in curves]), axis=0)
            ax.plot(np.arange(len(m)) * 5, m, lw=1.4,
                    label=f"{ARM_LABEL.get(args['arm'], args['arm'])}"
                          f"{'/' + args['warp_kind'] if args['arm'] == 'warp_leap' else ''}"
                          f" s{args['seed']}")
        ax.set_xlabel("adaptation step on a held-out language")
        ax.set_ylabel("task loss (nats/byte)")
        ax.set_title("C2: 100-step adaptation, no backprop through training")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{a.figdir}/llm_adaptation_curves.png", dpi=150)
        plt.close(fig)
        print(f"wrote {a.figdir}/llm_adaptation_curves.png")
    except Exception as e:
        print(f"figure generation skipped: {type(e).__name__}: {e}")

    json.dump(dict(rows=[{k: v for k, v in r.items() if k != "per_lang"}
                         for r in rows],
                   scratch_reference=scratch),
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
