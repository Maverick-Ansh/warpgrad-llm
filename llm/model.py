"""A byte-level GPT written from scratch, with warp-layers interleaved.

No Hugging Face, no pretrained weights, no tokenizer file.  Every parameter in
this model is trained by this repository.

WHY BYTE LEVEL
==============
Two reasons, and the first is the important one.

1. IT IS THE HONEST ANALOGUE OF THE PAPER'S BENCHMARK.  The paper's multi-shot
   experiment is Omniglot: 50 alphabets, each alphabet a separate task, raw
   28x28 images with no learned preprocessing shared across tasks.  The
   corresponding language-modelling setup is: many languages, each language a
   separate task, raw bytes with no learned tokenizer shared across tasks.  A
   BPE tokenizer trained on the meta-training languages would leak information
   about them into the held-out languages, which is exactly the kind of shared
   preprocessing Omniglot avoids.  Bytes keep the split clean.

2. THERE IS NO TOKENIZER TO LOSE.  A vocabulary of 256 is defined by the ASCII
   table, not by an artefact that has to be saved alongside a checkpoint.  A
   checkpoint from this repo can always encode new text.

The cost is real and stated: byte sequences are roughly 1.5x to 4x longer than
BPE for the same content, and much longer for non-Latin scripts where one
character is 2 or 3 UTF-8 bytes.  So a 256-byte context is a short context in
Hindi and a medium one in English.  This is a deviation from anything a modern
LLM does and it is recorded in REPORT.md.

WHERE THE WARP-LAYERS GO
========================
    "Given a model, designate all layers as task-adaptable and interleave
     warp-layers.  Warp-layers can be relatively weak as backpropagation through
     non-linear activations ensures expressive gradient warping.  This was our
     approach to the Omniglot experiment; our main architecture interleaves
     linear warp-layers in a standard architecture."
                                       -- Appendix A, "Model augmentation", p.15

    "We create a Warp Leap meta-learner that inserts warp-layers between each
     convolutional block, W . omega4 . h4 . ... . omega1 . h1"
                                                        -- Appendix E, page 19

So: one warp-layer after each transformer block, and the language-model head
after the last one.  Transformer block plays the role of convolutional block.
That is the direct port, and `--warp-every` lets a reader place them more
sparsely to see whether the density matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpgrad.warp import WarpLayer, split_parameters

BYTE_VOCAB = 256


@dataclass
class GPTConfig:
    vocab_size: int = BYTE_VOCAB
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.0          # 0 during meta-learning: see note in train.py
    warp_kind: str = "linear"     # see WarpLayer.KINDS
    warp_every: int = 1           # a warp-layer after every k-th block
    warp_rank: int = 16
    warp_expansion: int = 2
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal attention, written out rather than imported."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must divide by n_head"
        self.n_head, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, d_head)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Pre-LayerNorm transformer block.  This is h(i) in Eq. 6."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class WarpedGPT(nn.Module):
    """fhat = head . omega(L) . h(L) . ... . omega(1) . h(1) . embed

    The task-learner is an ordinary GPT.  The only structural change is a
    WarpLayer after each block.  With warp_kind="identity" this class IS the
    baseline GPT, bit for bit, which is what makes the no-warp control exact
    rather than approximate.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.wte.weight

        self.apply(self._init)
        # GPT-2 scaled init for residual projections: keeps activation variance
        # stable with depth, which matters because we will be taking 100 SGD
        # steps from this initialisation with no warmup.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        # Warp-layers are built AFTER self.apply(self._init), deliberately.
        # _init rewrites every nn.Linear to normal(0, 0.02), which would destroy
        # the identity initialisation that each WarpLayer sets up for itself
        # (proj.weight = I for "linear", zeroed residual branches for "mlp" and
        # "residual").  Building them last keeps their own init authoritative.
        # This bug was live for one commit and is exactly what
        # tests/test_warp.py::test_identity_warp_model_is_bit_identical_to_a_plain_gpt
        # exists to catch.
        self.warps = nn.ModuleList([
            WarpLayer(
                cfg.d_model,
                kind=cfg.warp_kind if (i + 1) % cfg.warp_every == 0 else "identity",
                rank=cfg.warp_rank,
                expansion=cfg.warp_expansion,
            )
            for i in range(cfg.n_layer)
        ])

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        # h(1), omega(1), h(2), omega(2), ...  exactly the composition in Eq. 6
        for block, warp in zip(self.blocks, self.warps):
            x = warp(block(x))

        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    # ------------------------------------------------------------------ splits

    def param_split(self):
        """(task_names, warp_names).  theta and phi, by name."""
        return split_parameters(self)

    def n_params(self):
        task, warp = self.param_split()
        d = dict(self.named_parameters())
        # tied head shares storage with wte, so count unique tensors only
        seen, n_task, n_warp = set(), 0, 0
        for n in task:
            if id(d[n]) in seen:
                continue
            seen.add(id(d[n]))
            n_task += d[n].numel()
        for n in warp:
            if id(d[n]) in seen:
                continue
            seen.add(id(d[n]))
            n_warp += d[n].numel()
        return dict(task=n_task, warp=n_warp, total=n_task + n_warp)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=1.0, top_k=None):
        """Sample bytes.  Used only for qualitative sanity output in the report."""
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
