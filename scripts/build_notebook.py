"""Generate notebooks/warpgrad_llm.ipynb from this file.

The notebook is a BUILD ARTIFACT, never edited by hand.  Two reasons:

  1. Prose and code cannot drift apart, because both live here and every code
     cell calls a script from the repository rather than duplicating it.
  2. A Colab session reset cannot destroy it.  This project already lost a
     fully annotated live notebook, plus about two hours of GPU results, to one
     session recycle.  A notebook that lives in git cannot be lost that way.

Run:  python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ===========================================================================
md(r"""
# WarpGrad from scratch, and applied to language models

Reproducing **[Meta-Learning with Warped Gradient Descent](https://arxiv.org/abs/1909.00025)**
(Flennerhag, Rusu, Pascanu, Visin, Yin, Hadsell — ICLR 2020), then porting it to
language modelling.

Code: **https://github.com/Maverick-Ansh/warpgrad-llm** — every code cell below
runs a script from that repo, so the notebook and the code cannot drift apart.
Nothing is imported from the authors' implementation.

Hardware: 2× Tesla T4 (compute capability 7.5, so **fp16 and never bf16**).

---

## The idea, in plain words

You want a network to learn a new task fast. MAML learns a good *starting point*
for the weights. WarpGrad instead learns a better *update rule*, but cheaply.

It inserts extra layers — **warp-layers** — between the normal layers. On the
backward pass the gradient must travel back through them, and each one multiplies
it by that layer's Jacobian. So the gradient arriving at a normal layer has
already been multiplied by a product of matrices the meta-learner controls. That
product **is** the preconditioner. You never build it, never store it, never
invert it:

> **You add some layers and call `.backward()`.**

The second idea is what makes it scale. MAML differentiates through the whole
training run, which breaks down past a few steps. WarpGrad does not. It treats
points visited during training as *samples*, takes one step from a sampled point,
asks whether that step helped, and nudges the warp to make it help more. The
training run is only a device for generating points to ask the question at. That
is why cost does not grow with the number of adaptation steps (Eq. 11).

## The seven claims under test

| # | Claim | Kind |
|---|---|---|
| C1 | Warp-layers beat the same task-learner with no warp, on held-out tasks | headline |
| C2 | Works at 100+ adaptation steps, where backprop-through-training cannot | headline |
| C3 | Warping turns a surface where gradient descent struggles into one where it does not *(the mountain)* | mechanism |
| C4 | More warp capacity is better; beyond-block-diagonal helps | mechanism |
| C5 | The learned geometry is **not** the Fisher information matrix | mechanism |
| C6 | Offline meta-training beats online | mechanism |
| C7 | The first-order objective (Eq. 12) costs little vs the exact one (Eq. 11) | mechanism |

## Two warnings before you read any number

**The evaluation broke before the model did — twice.** The first complete run
reported that WarpGrad *loses* to plain gradient descent (win rate 0.17). That was
an artefact: the literal Appendix D warp reaches only **25.4%** of the domain
that starting points are drawn from, so for three quarters of comparisons the two
optimisers began at *different places on the surface*. Both gate scripts exist to
catch that class of mistake and refuse to run a sweep on an instrument that
cannot measure.

**Bits per byte is not comparable across writing systems.** UTF-8 spends 1 byte
per character on Latin and 3 on Georgian, Tamil and Devanagari, and two of every
three bytes in a 3-byte script are near-free continuation bytes. A model can
score well on Tamil while knowing no Tamil. Every number here is bracketed
against that language's own floor and ceiling.

> Full write-up with verdicts: **REPORT.md**. Section 5, *What broke*, is the
> part worth reading.
""")

# ---------------------------------------------------------------------------
md("## Part 0 — Environment, and the real paper\n\n"
   "`WebFetch` on an arXiv PDF returns a summary that silently drops the "
   "equations and the appendix hyper-parameters, which are the only two things a "
   "reproduction needs. So we download the PDF and extract the text properly.")

code(r"""
import subprocess, sys, os, platform, torch
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  cc={p.major}.{p.minor}  {p.total_memory/1e9:.1f}GB")
print("cc 7.5 means fp16 + GradScaler, and NEVER bf16.")
""")

code(r"""
subprocess.run([sys.executable,"-m","pip","install","-q","pypdf"], check=True)
os.makedirs("/content/paper", exist_ok=True)
if not os.path.exists("/content/paper/warpgrad.pdf"):
    subprocess.run(["wget","-q","-O","/content/paper/warpgrad.pdf",
                    "https://arxiv.org/pdf/1909.00025v2"], check=True)
from pypdf import PdfReader
r = PdfReader("/content/paper/warpgrad.pdf")
txt = "\n\n".join(f"<<<PAGE {i+1}>>>\n{(p.extract_text() or '')}" for i,p in enumerate(r.pages))
open("/content/paper/warpgrad.txt","w").write(txt)
print(f"{len(r.pages)} pages, {len(txt)} chars extracted")
print("Appendix D (the mountain) is on page 18, Table 3 (the ladder) on page 22.")
""")

# ---------------------------------------------------------------------------
md("## Part 1 — The code, and 19 tests that assert the paper's rules\n\n"
   "These do not check tensor shapes. They check claims.\n\n"
   "- `test_family_is_randomised_peaks` — Appendix D's task family must reduce to "
   "the MATLAB `peaks` function *exactly* at $s=5$, $a=(1,1,1)$, "
   "$b=(3,10,\\tfrac13)$. If any sign or exponent were mistranscribed this fails "
   "even though every shape stays right.\n"
   "- `test_first_order_equivalence_error_scales_as_alpha_squared` — Eq. 9 claims "
   "a step in $P$-space equals the ideal step in $W$-space up to $O(\\alpha^2)$. "
   "We halve $\\alpha$ six times and check the gap falls ~4× each time. Linear "
   "scaling would mean the paper's justification for descending in $P$ does not "
   "hold.\n"
   "- `test_linear_warp_is_block_diagonal_and_mlp_warp_is_not` — the paper's "
   "stated contribution over T-Nets, tested on the **Jacobian itself** rather "
   "than inferred from accuracy.\n"
   "- `test_leap_gradient_is_not_the_reptile_direction` — exists because an "
   "earlier version of this repo computed Reptile's update and called it Leap. "
   "Nothing crashed. Two of four baselines would have been the same algorithm.")

code(r"""
REPO, ROOT = "https://github.com/Maverick-Ansh/warpgrad-llm.git", "/content/warpgrad-llm"
if os.path.exists(ROOT):
    print(subprocess.run(["git","-C",ROOT,"pull","--ff-only"],capture_output=True,text=True).stdout.strip())
else:
    subprocess.run(["git","clone","-q",REPO,ROOT], check=True); print("cloned")
os.chdir(ROOT); sys.path.insert(0, ROOT)
subprocess.run([sys.executable,"-m","pip","install","-q","datasets"], check=False)
for t in ["tests/test_mountain.py","tests/test_warp.py","tests/test_meta.py"]:
    r = subprocess.run([sys.executable, t], capture_output=True, text=True)
    print(f"{t:26s} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]}")
""")

# ---------------------------------------------------------------------------
md(r"""
## Part 2 — The mountain (Appendix D)

This is the figure everyone remembers, and it is cheap enough to run verbatim
with no resizing at all.

$$f_\tau(x_1,x_2) = b_1(a_1-x_1)^2 e^{-x_1^2-(x_2+a_2)^2} - b_2\!\left(\tfrac{x_1}{s}-x_1^3-x_2^5\right)e^{-x_1^2-x_2^2} - b_3 e^{-(x_1+a_3)^2-x_2^2}$$

with $s\sim\mathrm{Cat}(1..10)$, $a_i\sim\mathrm{Cat}(-1,0,1)$,
$b_i\sim\mathrm{Cat}(-5..5)$, starts $x\sim U(-3,3)$, 100 task steps at lr 0.1,
and a warp $\Omega$ that is a 2-layer tanh net with 30 hidden units.

**This family is a randomised MATLAB `peaks`.**

### Two problems in Appendix D that had to be solved

**A typo.** The third term is printed with $x_1$ in both exponent slots, which
turns a localised bump into an infinite ridge along $x_2$ and changes every
surface in the family. `peaks` has $x_2$ there, and the paper's own figures show
bumps, not ridges. Both readings are implemented; we default to `peaks` and say so.

**An omission that decides the experiment.** The warp's initialisation is never
stated. A freshly initialised 2-layer tanh net is an arbitrary squashing map
whose *image is a bounded blob*, and there is no reason that blob contains
$[-3,3]^2$. Measured: the literal reading reaches **25.4%** of the domain. The
rest is genuinely outside the range of $\Omega$, so WarpGrad cannot be started
there and no comparison exists.

So we also implement $\Omega(\theta)=\theta+g(\theta)$ with $g$'s output layer
zeroed, making $\Omega$ **exactly the identity at step 0**. Coverage becomes
100%, and we get a free correctness check: the measured advantage before any
meta-update must be exactly `0.0000`. It is.

### What "the same initialisation" costs

Gradient descent starts at $x_0$ on the surface. WarpGrad starts at $\theta_0$ in
its own coordinates, so it stands at $\Omega(\theta_0)$. Those coincide only if
$\Omega$ is the identity, which after meta-training it is not. So we **solve**
$\Omega(\theta_0)=x_0$ numerically and print the residual on every panel. A panel
with a large residual compares two different starting points on the mountain.

That residual is what exposed the first wrong answer this project produced.
""")

code(r"""
# Meta-train both readings of the warp, then GATE each one before believing it.
for init in ["plain", "residual"]:
    print("#"*74); print(f"#  warp init = {init}"); print("#"*74)
    subprocess.run([sys.executable,"synthetic/train_mountain.py","--meta-steps","100",
                    "--seed","0","--warp-init",init,"--third-term","peaks"])
    print(f"\n>>> GATE: is a comparison against gradient descent even DEFINED here?")
    subprocess.run([sys.executable,"synthetic/check_warp.py",
                    "--warp",f"results/mountain/warp_{init}_peaks_exact_s0.pt",
                    "--warp-init",init,"--out",f"results/mountain/check_{init}.json"])
""")

code(r"""
# Figures. Also emits fig_unselected_*.png from uniformly random tasks, because
# Appendix D selects its panels "such that standard gradient descent struggles"
# and a selected panel is an illustration, not evidence.
for init in ["residual","plain"]:
    subprocess.run([sys.executable,"synthetic/make_figures.py",
                    "--warp",f"results/mountain/warp_{init}_peaks_exact_s0.pt",
                    "--warp-init",init,"--third-term","peaks","--n-agg","150"])
print(sorted(os.listdir("figures")))
""")

code(r"""
from IPython.display import display, Markdown, Image as IPyImage
from PIL import Image
def show(path, width=900, note=""):
    im = Image.open(path).convert("RGB"); im.thumbnail((width,10000), Image.LANCZOS)
    small = "/tmp/"+os.path.basename(path).replace(".png","_s.jpg")
    im.save(small, quality=88, optimize=True)
    display(Markdown(f"**{os.path.basename(path)}** — {note}")); display(IPyImage(filename=small))

show("figures/fig3_mountain_residual_peaks_exact_s0.png",
     note="Figure 3. Top row is P-space, the warped surface WarpGrad actually descends. "
          "Bottom is the real mountain: black = plain gradient descent, magenta = the same "
          "WarpGrad run mapped back onto it. Inversion residual ~1e-7, so both genuinely start together.")
show("figures/fig_unselected_residual_peaks_exact_s0.png",
     note="The SAME warp on uniformly random tasks. This is the typical case and it is much "
          "less dramatic. Both panels are true.")
show("figures/fig7_trajectories_plain_peaks_exact_s0.png",
     note="THE BROKEN INSTRUMENT, kept on purpose. The literal Appendix D warp. The warped "
          "surface is a tiny patch and the magenta trajectory floats OFF it. Inversion residuals "
          "1.7 and 1.9 mean the two optimisers started in different places. This produced the "
          "confident wrong answer in REPORT section 5.1.")
""")

code(r"""
# The number the figure does NOT give you. Two different questions:
#   (a) does WarpGrad beat GD on a RANDOM task?
#   (b) does WarpGrad beat GD WHERE GD STRUGGLES (the regime the figure is drawn from)?
# The stratified table answers (b). Bins use the GD score ONLY and never look at
# WarpGrad, so this cannot manufacture a result.
subprocess.run([sys.executable,"-u","synthetic/evaluate.py",
                "--warp","results/mountain/warp_residual_peaks_exact_s0.pt",
                "--warp-init","residual","--n-pairs","200"])
""")

# ---------------------------------------------------------------------------
md(r"""
## Part 3 — The language tier

Each **language** is a task. Each task is next-byte prediction. Whole languages,
and whole **scripts**, are held out.

| | Paper (Omniglot) | Here |
|---|---|---|
| task | one alphabet, 20-way classification | one language, next-byte prediction |
| task-learner | 4 conv blocks, 64 filters | 6-layer byte-level GPT, d_model 256 |
| warp placement | after each conv block | after each transformer block |
| adaptation | 100 SGD steps | 100 SGD steps |
| meta-train | 25 alphabets | 25 languages |
| held out | 10 alphabets | 7 languages, 7 scripts |

The held-out split is deliberately mixed so two questions stay separable:
`uk ca fa mr` are new languages in a **seen** script, while `ka ta ko` are new
languages in a script seen **nowhere** in meta-training. If a learned geometry
transfers only within a writing system, those groups separate. Omniglot cannot
ask this, because all its alphabets are rendered identically.

### Why raw bytes and not a tokenizer

A BPE tokenizer trained on the meta-training languages would leak information
about them into the held-out languages. Omniglot shares no such preprocessing
across alphabets, so bytes are the faithful choice. The cost is real and stated:
byte sequences run 1.5–4× longer, and much longer for non-Latin scripts.

### The gate you must pass before believing anything

Per language, measured on that language's own data:

- **floor** — a smoothed bigram byte table. Achievable with no transfer at all.
- **ceiling** — the same architecture trained on that one language with Adam for
  30× the adaptation budget.

Everything is reported as `score = (floor − achieved) / (floor − ceiling)`, where
**1.0 = oracle** and **0.0 = bigram table**. That scale is comparable across
writing systems. Raw bits per byte is not.
""")

code(r"""
# ONE detached pipeline: corpus -> ceilings (GATE) -> Phase A -> Phase B -> C5 -> analysis.
# Every stage is skippable, so re-running this cell after any interruption RESUMES.
# This project already lost ~2h of finished GPU work to a Colab session reset
# because results lived only on /content. Never block this kernel: there is no interrupt.
os.makedirs("results/llm/logs", exist_ok=True)
import time
p = subprocess.Popen([sys.executable,"-u","scripts/run_all.py",
                      "--meta-steps","120","--seeds","0","1","--ladder-seeds","0",
                      "--max-minutes","40"],
                     stdout=open("results/llm/logs/PIPELINE.log","w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
print("pipeline pid", p.pid, "- detached"); time.sleep(30)
print(open("results/llm/logs/PIPELINE.log").read()[-1500:])
""")

code(r"""
#@title Poll (cheap, safe to re-run, never blocks)
import json, glob
s = json.load(open("results/STATUS.json")) if os.path.exists("results/STATUS.json") else {}
print(f"stage: {s.get('stage','?')} | {s.get('detail','')} | {s.get('updated','')}")
done = sorted(os.path.basename(x)[:-5] for x in glob.glob("results/llm/[AB]_*.json"))
print(f"finished runs: {len(done)}  {done}")
for lg in sorted(glob.glob("results/llm/logs/[AB]_*.log")):
    ls = [l for l in open(lg).read().splitlines() if "meta " in l]
    if ls: print(f"  {os.path.basename(lg)[:-4]:<26} {ls[-1][:96]}")
print(open("results/llm/logs/PIPELINE.log").read()[-800:])
""")

md("### Results\n\n"
   "`scripts/analyse.py` prints every arm as a bracketed, normalised score, split "
   "by whether the held-out language's **script** appeared in meta-training. The "
   "two groups are never pooled into one mean, because pooling would hide exactly "
   "the thing worth knowing.")

code(r"""
subprocess.run([sys.executable,"scripts/analyse.py"])
""")

code(r"""
for f in ["llm_per_language.png","llm_capacity_ladder.png",
          "llm_adaptation_curves.png","llm_fisher_check.png"]:
    p = os.path.join("figures", f)
    if os.path.exists(p): show(p, width=980, note=f)
""")

md("---\n\n"
   "## Where the verdicts live\n\n"
   "**REPORT.md** in the repo. It records, per claim: whether it reproduced, what "
   "the number was, and what was *not* tested. It never blurs **untestable** into "
   "**refuted** — if the instrument lacked the range to detect an effect, that is "
   "a different statement about the world, and the honest one.")

# ===========================================================================
if __name__ == "__main__":
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "warpgrad_llm.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}  ({len(CELLS)} cells, "
          f"{sum(1 for c in CELLS if c['cell_type'] == 'code')} code)")
