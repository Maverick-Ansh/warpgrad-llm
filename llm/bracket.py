"""GATE for the language tier: bracket every language before comparing anything.

This is the LLM counterpart of synthetic/check_warp.py, and it exists for the
same reason: on a resized reproduction the evaluation breaks more often than the
model does, and a headline number computed on a broken instrument looks exactly
like a real result.

THE PROBLEM WITH BITS PER BYTE
==============================
UTF-8 spends a different number of bytes per character in different scripts:

    Latin        1 byte/char
    Cyrillic, Greek, Hebrew, Arabic   2 bytes/char
    Devanagari, Georgian, Tamil, CJK, Hangul   3 bytes/char

In a 3-byte script, two out of every three bytes are continuation bytes whose
value is almost fully determined by the byte before them.  A model can therefore
score a very low bits-per-byte on Tamil while understanding nothing about Tamil,
purely by learning UTF-8.  Conversely a genuinely good English model will look
"worse" on the same scale.

So a table of raw bits-per-byte averaged over languages ranks meta-learners
mostly by which scripts landed in the held-out split.  That is a property of the
split, not of the method.  Every number in REPORT.md is bracketed instead.

THE BRACKET
===========
For each language we measure three reference points on the SAME validation data
the models are scored on:

  FLOOR-0  uniform over 256 byte values = 8.000 bits/byte exactly.
           Knowing nothing at all.

  FLOOR-1  the unigram byte model: fit byte frequencies on the training stream,
           score the validation stream.  This is "knowing the script and nothing
           else".  For a 3-byte script this is already a big drop from 8, which
           is precisely the effect we need to divide out.

  FLOOR-2  the bigram byte model with add-one smoothing, fit on training and
           scored on validation.  This is "knowing local byte co-occurrence and
           nothing else", and for UTF-8 continuation bytes it is very strong.
           A neural model that cannot beat FLOOR-2 has learned nothing a lookup
           table does not already contain.  This is the degenerate shortcut for
           this tier, and it is reported next to every result.

  CEILING  an oracle: the SAME architecture trained from scratch on that one
           language alone, for many times the adaptation budget.  No
           meta-learner adapting in 100 steps should beat this, so it sets the
           top of the achievable range.

We then report the normalised score

    score = (FLOOR2_bpb - achieved_bpb) / (FLOOR2_bpb - CEILING_bpb)

    1.0  matched a model trained on this language from scratch at full budget
    0.0  no better than a smoothed bigram table
   <0.0  worse than a smoothed bigram table

which IS comparable across languages, because both ends of the scale are
measured on that language's own data.

WHY FLOOR-2 AND NOT FLOOR-1 IS THE ZERO
---------------------------------------
Because FLOOR-2 is achievable without learning anything transferable, and the
claim under test is about transfer.  Using the weaker FLOOR-1 as the zero would
credit every method with the free win of learning UTF-8 continuation structure,
which all of them get and none of them get from meta-learning.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

LOG2 = math.log(2.0)


def unigram_bpb(train: np.ndarray, val: np.ndarray, alpha: float = 1.0) -> float:
    """Bits per byte of a smoothed unigram model fit on train, scored on val."""
    counts = np.bincount(train, minlength=256).astype(np.float64) + alpha
    logp = np.log(counts / counts.sum())
    return float(-logp[val].mean() / LOG2)


def bigram_bpb(train: np.ndarray, val: np.ndarray, alpha: float = 1.0) -> float:
    """Bits per byte of an add-alpha bigram model fit on train, scored on val.

    Built with a 256x256 count matrix, which is 64k floats, so this is exact
    rather than approximate and costs milliseconds.
    """
    counts = np.zeros((256, 256), dtype=np.float64)
    np.add.at(counts, (train[:-1], train[1:]), 1.0)
    counts += alpha
    logp = np.log(counts / counts.sum(axis=1, keepdims=True))
    # first validation byte has no predecessor: score it under the unigram model
    uni = np.bincount(train, minlength=256).astype(np.float64) + alpha
    first = -math.log(uni[val[0]] / uni.sum())
    rest = -logp[val[:-1], val[1:]].sum()
    return float((first + rest) / len(val) / LOG2)


def uniform_bpb() -> float:
    """8.0 exactly.  Present so the report never has to assert it in prose."""
    return 8.0


def normalised_score(achieved_bpb: float, floor_bpb: float, ceiling_bpb: float) -> float:
    span = floor_bpb - ceiling_bpb
    if span <= 1e-9:
        return float("nan")     # oracle no better than bigram: language unusable
    return (floor_bpb - achieved_bpb) / span


def build_floors(corpus, out_path=None, verbose=True):
    """Compute FLOOR-0/1/2 for every language.  Cheap, exact, no GPU."""
    floors = {}
    if verbose:
        print(f"{'lang':<8}{'script':<12}{'uniform':>9}{'unigram':>9}{'bigram':>9}")
        print("-" * 47)
    for lang, d in corpus.items():
        rec = dict(
            script=d.script,
            uniform=uniform_bpb(),
            unigram=unigram_bpb(d.train, d.val),
            bigram=bigram_bpb(d.train, d.val),
            n_train=int(len(d.train)),
            n_val=int(len(d.val)),
        )
        floors[lang] = rec
        if verbose:
            print(f"{lang:<8}{rec['script']:<12}{rec['uniform']:>9.3f}"
                  f"{rec['unigram']:>9.3f}{rec['bigram']:>9.3f}")
    if verbose:
        print("-" * 47)
        print("Note how far the bigram floor falls for 2- and 3-byte scripts.")
        print("That drop is UTF-8 structure, not language understanding, and it")
        print("is exactly what averaging raw bits-per-byte would reward.")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(floors, open(out_path, "w"), indent=2)
    return floors


def report_bracket(floors, ceilings, verbose=True):
    """Print the bracket per language and flag any that are too narrow to use."""
    rows, unusable = [], []
    for lang, f in floors.items():
        c = ceilings.get(lang)
        if c is None:
            continue
        span = f["bigram"] - c
        rows.append((lang, f["script"], f["uniform"], f["unigram"], f["bigram"], c, span))
        if span < 0.05:
            unusable.append(lang)
    if verbose:
        print(f"{'lang':<8}{'script':<12}{'unigram':>9}{'bigram':>9}"
              f"{'oracle':>9}{'span':>8}")
        print("-" * 55)
        for lang, sc, _u0, u1, b, c, span in sorted(rows, key=lambda r: -r[6]):
            flag = "  <-- TOO NARROW" if span < 0.05 else ""
            print(f"{lang:<8}{sc:<12}{u1:>9.3f}{b:>9.3f}{c:>9.3f}{span:>8.3f}{flag}")
        print("-" * 55)
        print("span = bigram floor - oracle ceiling, in bits/byte.  This is the")
        print("entire dynamic range available to any meta-learner on that")
        print("language.  A narrow span means the language cannot discriminate")
        print("between methods and must be excluded, not averaged in.")
        if unusable:
            print(f"\nEXCLUDE from all comparisons: {unusable}")
    return dict(rows=rows, unusable=unusable)
