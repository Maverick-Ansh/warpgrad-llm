"""The task distribution p(tau): one language-modelling task per language.

THE PORT
========
The paper's multi-shot benchmark is Omniglot under the protocol of Flennerhag
et al. (2019):

    "each of the 50 alphabets that comprise the dataset constitutes a distinct
     task.  Each task is treated as a 20-way classification problem. ... 10
     alphabets are held-out for final meta-testing"
                                                        -- Appendix E, page 19

The structure that matters there is: many tasks, each task is the same KIND of
problem over a different SYMBOL SYSTEM, and whole symbol systems are held out at
meta-test time so the meta-learner cannot have memorised them.

The language-modelling analogue is exact.  Each language is a task, each task is
next-byte prediction, and whole languages (including whole scripts) are held out.
A meta-learner that has never seen a single byte of Georgian must adapt to
Georgian in 100 gradient steps using only the geometry it learned elsewhere.

WHY THIS IS A FAIR TEST AND NOT A RESKIN
========================================
The thing under test is the warp, and the warp is shared across tasks.  For the
test to be meaningful, tasks must share structure the warp can capture while
differing in ways it cannot memorise.  Languages do exactly that: they share the
statistics of natural text (Zipfian token frequencies, strong local dependence,
whitespace and punctuation structure, long-range topical coherence) while
differing in vocabulary, morphology and script.

WHAT IS NOT COMPARABLE ACROSS THESE TASKS, AND WHY IT MATTERS
=============================================================
Bits per byte is NOT comparable across languages.  In UTF-8, a Latin character
is 1 byte, a Cyrillic or Greek or Hebrew character is 2, and a Devanagari,
Georgian, Tamil or CJK character is 3.  A model that predicts Hindi at 1.2
bits/byte is not twice as good as one predicting English at 2.4 bits/byte, it is
just being scored on a script whose bytes are more predictable because two out of
every three of them are continuation bytes with almost no entropy.

Averaging raw bits per byte across languages would therefore rank meta-learners
mostly by which scripts happened to be in the held-out split.  Every number this
repository reports is bracketed per language instead.  See `llm/bracket.py`.

DATA SOURCE
===========
Wikipedia (wikimedia/wikipedia, 20231101 snapshots) streamed per language, so
nothing is downloaded that is not used.  Cached to disk as raw uint8 arrays.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

# 30 languages spanning 12 scripts and 9 families.  The split is fixed here and
# never chosen by results.  Held-out languages deliberately include BOTH a script
# that appears in meta-training (Cyrillic: uk, seen via ru/bg) and scripts that
# appear nowhere in it (Georgian ka, Tamil ta, Korean ko), so the report can
# separate "adapting to a new language" from "adapting to a new writing system".
META_TRAIN = [
    # Latin
    "simple", "es", "fr", "de", "it", "pt", "nl", "sv", "pl", "cs",
    "tr", "id", "sw", "vi", "fi", "hu", "ro", "da",
    # Cyrillic
    "ru", "bg",
    # Greek / Hebrew / Arabic / Devanagari / CJK
    "el", "he", "ar", "hi", "ja",
]
META_TEST = [
    "uk",   # Cyrillic  -- script SEEN in meta-training (ru, bg)
    "ca",   # Latin     -- script seen, language unseen
    "fa",   # Arabic    -- script seen (ar), language unseen
    "mr",   # Devanagari-- script seen (hi), language unseen
    "ka",   # Georgian  -- script UNSEEN anywhere in meta-training
    "ta",   # Tamil     -- script UNSEEN anywhere in meta-training
    "ko",   # Hangul    -- script UNSEEN anywhere in meta-training
]

SCRIPT = {
    "simple": "Latin", "es": "Latin", "fr": "Latin", "de": "Latin", "it": "Latin",
    "pt": "Latin", "nl": "Latin", "sv": "Latin", "pl": "Latin", "cs": "Latin",
    "tr": "Latin", "id": "Latin", "sw": "Latin", "vi": "Latin", "fi": "Latin",
    "hu": "Latin", "ro": "Latin", "da": "Latin", "ca": "Latin",
    "ru": "Cyrillic", "bg": "Cyrillic", "uk": "Cyrillic",
    "el": "Greek", "he": "Hebrew", "ar": "Arabic", "fa": "Arabic",
    "hi": "Devanagari", "mr": "Devanagari",
    "ja": "Japanese", "ko": "Hangul", "ka": "Georgian", "ta": "Tamil",
}

SNAPSHOT = "20231101"


@dataclass
class LangData:
    """One task's data: a train byte stream and a disjoint validation stream."""

    lang: str
    train: np.ndarray        # uint8
    val: np.ndarray          # uint8

    @property
    def script(self) -> str:
        return SCRIPT.get(self.lang, "?")

    def __repr__(self):
        return (f"LangData({self.lang}, {self.script}, "
                f"train={len(self.train) / 1e6:.2f}MB, val={len(self.val) / 1e6:.2f}MB)")


def fetch_language(lang, cache_dir, train_mb=2.0, val_mb=0.5, max_articles=20000):
    """Stream Wikipedia for one language until the byte budget is met, then cache.

    Articles are concatenated with a blank line between them, which is the usual
    thing to do and keeps document boundaries visible to the model as a pattern
    rather than as a special token.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tr_p = os.path.join(cache_dir, f"{lang}_train.npy")
    va_p = os.path.join(cache_dir, f"{lang}_val.npy")
    if os.path.exists(tr_p) and os.path.exists(va_p):
        return LangData(lang, np.load(tr_p), np.load(va_p))

    from datasets import load_dataset

    need = int((train_mb + val_mb) * 1024 * 1024)
    ds = load_dataset("wikimedia/wikipedia", f"{SNAPSHOT}.{lang}",
                      split="train", streaming=True)
    chunks, total = [], 0
    for row in itertools.islice(ds, max_articles):
        b = (row["text"] + "\n\n").encode("utf-8")
        chunks.append(b)
        total += len(b)
        if total >= need:
            break
    if total < need * 0.5:
        raise RuntimeError(
            f"{lang}: only got {total / 1e6:.2f}MB of a requested "
            f"{need / 1e6:.2f}MB. Wikipedia for this language may be too small."
        )

    buf = np.frombuffer(b"".join(chunks), dtype=np.uint8)[:need]
    # A CONTIGUOUS split, not a random one.  Random windows would put sentences
    # from the same article on both sides of the split and leak validation text
    # into training, which would quietly inflate every number in the report.
    n_val = int(len(buf) * val_mb / (train_mb + val_mb))
    val, train = buf[:n_val].copy(), buf[n_val:].copy()
    np.save(tr_p, train)
    np.save(va_p, val)
    return LangData(lang, train, val)


def build_corpus(cache_dir="/content/data", langs=None, train_mb=2.0, val_mb=0.5,
                 verbose=True):
    """Fetch every language.  Returns {lang: LangData} and writes a manifest."""
    langs = langs or (META_TRAIN + META_TEST)
    out, failed = {}, []
    for i, lang in enumerate(langs):
        try:
            out[lang] = fetch_language(lang, cache_dir, train_mb, val_mb)
            if verbose:
                print(f"  [{i + 1:2d}/{len(langs)}] {out[lang]}", flush=True)
        except Exception as e:
            failed.append((lang, f"{type(e).__name__}: {str(e)[:120]}"))
            if verbose:
                print(f"  [{i + 1:2d}/{len(langs)}] FAILED {lang}: {failed[-1][1]}",
                      flush=True)
    manifest = dict(
        snapshot=SNAPSHOT, train_mb=train_mb, val_mb=val_mb,
        meta_train=[l for l in META_TRAIN if l in out],
        meta_test=[l for l in META_TEST if l in out],
        scripts={l: SCRIPT.get(l, "?") for l in out},
        failed=failed,
    )
    json.dump(manifest, open(os.path.join(cache_dir, "manifest.json"), "w"), indent=2)
    if failed:
        print(f"  WARNING: {len(failed)} languages unavailable: "
              f"{[f[0] for f in failed]}")
    return out, manifest


class ByteSampler:
    """Random crops of (block_size + 1) bytes from one language's stream.

    Returns (x, y) where y is x shifted by one, the standard next-token setup.
    A fresh torch.Generator per sampler keeps task data reproducible independently
    of whatever else is consuming the global RNG.
    """

    def __init__(self, data: np.ndarray, block_size: int, device="cpu", seed=0):
        import torch

        self.data = torch.from_numpy(np.ascontiguousarray(data)).long()
        self.block_size = block_size
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        if len(self.data) <= block_size + 1:
            raise ValueError(f"stream of {len(self.data)} bytes is too short "
                             f"for block_size {block_size}")

    def batch(self, batch_size):
        import torch

        hi = len(self.data) - self.block_size - 1
        ix = torch.randint(hi, (batch_size,), generator=self.gen)
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def sequential_batches(self, batch_size, max_batches=None):
        """Deterministic non-overlapping sweep, for validation.

        Validation must not be a random sample, or the validation number itself
        becomes noisy and small real differences get buried under sampling noise.
        """
        import torch

        n = (len(self.data) - 1) // self.block_size
        n = min(n, (max_batches or n) * batch_size)
        for s in range(0, n, batch_size):
            idx = [s + j for j in range(min(batch_size, n - s))]
            x = torch.stack([self.data[i * self.block_size:
                                       (i + 1) * self.block_size] for i in idx])
            y = torch.stack([self.data[i * self.block_size + 1:
                                       (i + 1) * self.block_size + 1] for i in idx])
            yield x.to(self.device), y.to(self.device)
