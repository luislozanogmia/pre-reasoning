#!/usr/bin/env python3
"""
inference.py -- Neural perception layer (12M MoE v4)
================================================

Learned successor to ReasoningEngineV2 (harness/rule-based).
12M MoE neural network brain (v4, 5 experts).

TWO-PASS PIPELINE:
  PASS A  F1-F3  : b8 auto-routed MoE generate -> dependencies / conflicts /
                   requirements / root-blocker / unlock-sequence.
  PASS B  F5     : dedicated E4 expert (family 5, fixed_family=5) ->
                   transitive closure (implicit assumptions) + HAS_YES/NO verdict.

Modes:
  ll = language in, language out (default)
  ls = language in, structural out
  ss = structural in, structural out
  sl = structural in, language out

Author: Dr. Shannon, Mia Labs
Date: 2026-06-23 (12M MoE v4)
"""

import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "pre-reasoning-12m-v3.safetensors"

# ── Vocabulary ────────────────────────────────────────────────────────────────

CONTROL = [
    "PAD", "BOS", "EOS", "INPUT", "OUTPUT", "SEP", "QUERY",
    "UNLOCK_SEQ", "ROOT_BLOCKER", "PARALLEL_WORK", "HAS_CYCLE",
]
STRUCTURAL_OPS = [
    "DEPENDS_ON", "CONTRADICTS", "BLOCKS", "UNLOCKS", "PARALLEL",
    "REQUIRES", "ENABLES", "CONFLICTS", "SEQUENCES",
]
ROLES = ["ROOT", "BLOCKER", "CHAIN", "CYCLE", "LEAF", "CONSTRAINT"]
ACTIONS = ["RESOLVE", "SKIP", "DEFER", "ESCALATE"]
ENTITIES = [f"ENT_{i}" for i in range(32)]

LANG_CONTROL = ["LANG_IN", "LANG_OUT"]
LANG_WORDS = [
    "the", "is", "a", "on", "by", "in", "and", "of", "or", "not",
    "depends", "needs", "requires",
    "blocked_w", "waiting", "stopped",
    "enables_w", "allows", "makes",
    "contradicts_w", "conflicts",
    "start", "happen", "proceed", "done", "missing",
    "task", "blocker_w",
    "root_w", "main_w", "first_w", "original_w",
    "cycle_w", "circular", "dependency_w", "loop_w",
    "possible", "indirectly", "both", "true_w",
    "cannot", "must", "can", "has_w",
    "before", "after", "until",
    "together", "with_w",
    "PERIOD",
]
LANG_ENTITIES = [f"TASK_{chr(65+i)}" if i < 26 else f"TASK_{i}" for i in range(32)]

CYCLE_TOKENS = [
    "CYCLE_CHECK", "CYCLE_NODES", "IN_CYCLE", "NO_CYCLE",
    "HAS_CYCLE_YES", "HAS_CYCLE_NO", "BACK_EDGE", "LOCAL_CYCLE", "NONE",
]
NEW_LANG_WORDS = [
    "relies", "waits", "for_w", "unlocks_w", "advance",
    "finish", "fixed", "at_w", "same", "time_w",
    "while_w", "required", "are", "because", "no_w",
    "there", "we_w", "only", "then_w", "still",
]
CF_NEW_TOKENS = ["NORMALIZED"]

# Conflict language alias tokens (synonyms for "conflicts with")
CONFLICT_LANG_TOKENS = ["clashes", "incompatible", "mutual", "exclusive", "opposes"]

# Requirement tokens (GEQ/LEQ operators, value bins, verdict tokens)
REQUIREMENT_TOKENS = [
    "REQUIREMENT", "REQUIREMENTS",  # relation token + section header
    "GEQ", "LEQ",                   # >= and <= operators
    "VALUE_0", "VALUE_5", "VALUE_10", "VALUE_50", "VALUE_60",
    "VALUE_70", "VALUE_75", "VALUE_80", "VALUE_85", "VALUE_90", "VALUE_95", "VALUE_100",
    "HAS_REQ_YES", "HAS_REQ_NO",   # requirement verdict tokens
]

ALL_TOKENS = (CONTROL + STRUCTURAL_OPS + ROLES + ACTIONS + ENTITIES +
              LANG_CONTROL + LANG_WORDS + LANG_ENTITIES + CYCLE_TOKENS +
              NEW_LANG_WORDS + CF_NEW_TOKENS + CONFLICT_LANG_TOKENS +
              REQUIREMENT_TOKENS)
TOKEN2ID = {tok: idx for idx, tok in enumerate(ALL_TOKENS)}
ID2TOKEN = {idx: tok for tok, idx in TOKEN2ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)  # 199

# -- CM23/CM25B: Sentinel tokens (input-only, never predicted) --
OP_GEQ_ID = 199
OP_LEQ_ID = 200
CM23_VOCAB_SIZE = 201  # Input embedding size (head stays at 199)

PAD_ID = TOKEN2ID["PAD"]
BOS_ID = TOKEN2ID["BOS"]
EOS_ID = TOKEN2ID["EOS"]
DEP_EDGES = {"DEPENDS_ON", "REQUIRES"}

ENT_TO_TASK = {f"ENT_{i}": LANG_ENTITIES[i] for i in range(32)}
TASK_TO_ENT = {v: k for k, v in ENT_TO_TASK.items()}

# ── Extended token IDs (from cm23) ──────────────────────────────────────────
NORMALIZED_ID = TOKEN2ID["NORMALIZED"]
DEPENDS_ON_ID = TOKEN2ID["DEPENDS_ON"]
ROOT_BLOCKER_ID = TOKEN2ID["ROOT_BLOCKER"]
BACK_EDGE_ID = TOKEN2ID["BACK_EDGE"]
CONFLICTS_ID = TOKEN2ID["CONFLICTS"]
REQUIREMENT_ID = TOKEN2ID["REQUIREMENT"]
REQUIREMENTS_ID = TOKEN2ID["REQUIREMENTS"]
GEQ_ID = TOKEN2ID["GEQ"]
LEQ_ID = TOKEN2ID["LEQ"]
HAS_REQ_YES_ID = TOKEN2ID["HAS_REQ_YES"]
HAS_REQ_NO_ID = TOKEN2ID["HAS_REQ_NO"]
PERIOD_ID = TOKEN2ID["PERIOD"]

# ── B8 conditional token IDs (201-210) ──────────────────────────────────────
CONDITIONALS_ID = 201
COND_IF_ID = 202
COND_THEN_ID = 203
COND_ELSE_ID = 204
IF_MET_ID = 205
IF_NOT_MET_ID = 206
HAS_COND_YES_ID = 207
HAS_COND_NO_ID = 208
COND_DEP_ID = 209
CD_IF_W_ID = 210

B1_INPUT_VOCAB_SIZE = 211
B1_OUTPUT_VOCAB_SIZE = 211

INPUT_ONLY_SENTINEL_IDS = {OP_GEQ_ID, OP_LEQ_ID}  # 199, 200

B1_TOKEN_NAMES = {
    CONDITIONALS_ID: "CONDITIONALS",
    COND_IF_ID: "COND_IF",
    COND_THEN_ID: "COND_THEN",
    COND_ELSE_ID: "COND_ELSE",
    IF_MET_ID: "IF_MET",
    IF_NOT_MET_ID: "IF_NOT_MET",
    HAS_COND_YES_ID: "HAS_COND_YES",
    HAS_COND_NO_ID: "HAS_COND_NO",
    COND_DEP_ID: "COND_DEP",
    CD_IF_W_ID: "if_w",
}

# ── H1 token extension (IDs 217, 218, 223) ──────────────────────────────────
HAS_ASSUMPTIONS_YES_ID = 217
HAS_ASSUMPTIONS_NO_ID = 218
CLOSURE_ASSUM_ID = 223
H1_INPUT_VOCAB_SIZE = 224
H1_OUTPUT_VOCAB_SIZE = 224
H1_TAIL_FAMILY = 5

# Extend TOKEN2ID/ID2TOKEN with H1 tokens
TOKEN2ID["HAS_ASSUMPTIONS_YES"] = HAS_ASSUMPTIONS_YES_ID
TOKEN2ID["HAS_ASSUMPTIONS_NO"] = HAS_ASSUMPTIONS_NO_ID
TOKEN2ID["CLOSURE_ASSUM"] = CLOSURE_ASSUM_ID
ID2TOKEN[HAS_ASSUMPTIONS_YES_ID] = "HAS_ASSUMPTIONS_YES"
ID2TOKEN[HAS_ASSUMPTIONS_NO_ID] = "HAS_ASSUMPTIONS_NO"
ID2TOKEN[CLOSURE_ASSUM_ID] = "CLOSURE_ASSUM"

# ── Architecture constants (12M MoE) ─────────────────────────────────────────
D_MODEL = 288
N_HEADS = 8
D_FF = 1152
N_LAYERS = 8
N_FAMILY_TYPES = 10
N_EXPERT_GROUPS = 5  # 5 expert groups
MAX_SEQ_LEN = 1024
DROPOUT = 0.1
MAX_LEN = 160

# ── Family->Expert routing (H1 patch: family 5 -> E4) ────────────────────────
FAMILY_EXPERT_MAP = {
    0: 0, 1: 0, 2: 1, 3: 1,
    4: 2, 5: 4,  # H1 patch: family 5 -> E4
    6: 2, 7: 2, 8: 3, 9: 3,
}

# ── Direction aliases (for NL parsing) ───────────────────────────────────────

# Reverse: subject is the enabler/root ("A enables B" -> B DEPENDS_ON A)
REVERSE_PATTERNS = [
    (r"\benables\b", "enables"),
    (r"\ballows\b", "allows"),
    (r"\bunlocks\b", "unlocks"),
    (r"\bmakes\s+possible\b", "makes_possible"),
    (r"\bmust\s+happen\s+before\b", "must_happen_before"),
    (r"\bneeds?\s+(?:to\s+)?happen\s+before\b", "needs_happen_before"),
    (r"\bcomes?\s+before\b", "comes_before"),
    (r"\brequired\s+before\b", "required_before"),
    (r"\blets?\s+.*\s+start\b", "lets_start"),
    (r"\bbefore\b(?!\s+(?:and|or|the)\b)", "before"),
]

# Forward: subject is the dependent ("B depends on A" -> B DEPENDS_ON A)
FORWARD_PATTERNS = [
    (r"\bdepends?\s+on\b", "depends_on"),
    (r"\bneeds\b", "needs"),
    (r"\brequires\b", "requires"),
    (r"\brelies\s+on\b", "relies_on"),
    (r"\bwaits?\s+for\b", "waits_for"),
    (r"\bis\s+waiting\s+for\b", "is_waiting_for"),
    (r"\bcannot\s+start\s+until\b", "cannot_start_until"),
    (r"\bcannot\s+proceed\s+until\b", "cannot_proceed_until"),
    (r"\bblocked\s+by\b", "blocked_by"),
    (r"\bmust\s+happen\s+after\b", "must_happen_after"),
    (r"\bafter\b", "after"),
]

# Token sequences for each alias (language tokens the model expects)
REVERSE_ALIAS_TOKENS = {
    "enables": ["enables_w"],
    "allows": ["allows"],
    "unlocks": ["unlocks_w"],
    "makes_possible": ["makes", "possible"],
    "must_happen_before": ["must", "happen", "before"],
    "needs_happen_before": ["needs", "happen", "before"],
    "comes_before": ["before"],
    "required_before": ["required", "before"],
    "lets_start": ["enables_w", "start"],
    "before": ["before"],
}

FORWARD_ALIAS_TOKENS = {
    "depends_on": ["depends", "on"],
    "needs": ["needs"],
    "requires": ["requires"],
    "relies_on": ["relies", "on"],
    "waits_for": ["waits", "for_w"],
    "is_waiting_for": ["is", "waiting", "for_w"],
    "cannot_start_until": ["cannot", "start", "until"],
    "cannot_proceed_until": ["cannot", "proceed", "until"],
    "blocked_by": ["blocked_w", "by"],
    "must_happen_after": ["must", "happen", "after"],
    "after": ["must", "happen", "after"],
}

# Conflict: symmetric ("A contradicts B" -> A CONTRADICTS B)
CONFLICT_PATTERNS = [
    (r"\bcontradicts\b", "contradicts"),
    (r"\bconflicts?\s+with\b", "conflicts_with"),
    (r"\bclashes\s+with\b", "clashes_with"),
    (r"\bis\s+incompatible\s+with\b", "incompatible_with"),
    (r"\bopposes\b", "opposes"),
]

# Requirement: "A requires B" with optional GEQ/LEQ/VALUE (parsed separately)
REQUIREMENT_PATTERNS = [
    (r"\brequires\b", "requires_req"),
]

# ── Phrasing aliases (from eval_unified.py) ──────────────────────────────────

FORWARD_PHRASINGS = [
    ["depends", "on"],
    ["needs"],
    ["requires"],
    ["blocked_w", "by"],
]

REVERSE_PHRASINGS = [
    ["enables_w"],
    ["allows"],
    ["unlocks_w"],
    ["must", "happen", "before"],
]

TEMPORAL_BEFORE_PHRASINGS = [
    ["before"],
    ["must", "happen", "before"],
]

TEMPORAL_AFTER_PHRASINGS = [
    ["after"],
    ["must", "happen", "after"],
]

CONFLICT_ALIASES = [
    ["conflicts", "with_w"],
    ["contradicts_w"],
    ["clashes", "with_w"],
    ["is", "incompatible", "with_w"],
    ["mutual", "exclusive"],
    ["opposes"],
]

REQUIREMENT_GEQ_LANG_ALIASES = [
    ["must", "reach"],
    ["must", "be"],
    ["needs"],
    ["required"],
]

REQUIREMENT_LEQ_LANG_ALIASES = [
    ["must", "stay"],
    ["cannot", "reach"],
    ["must", "not"],
    ["cannot", "start"],
]


# ── Model Architecture (12M MoE v4) ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ExpertBlock(nn.Module):
    """Transformer block with shared attention and family-routed expert FFNs."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int, n_experts: int, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        residual = x
        x_normed = self.ln2(x)

        B, T, D = x_normed.shape
        output = torch.zeros_like(x_normed)

        for expert_id in range(self.n_experts):
            mask = (expert_indices == expert_id)
            if not mask.any():
                continue
            expert_input = x_normed[mask]
            expert_output = self.experts[expert_id](expert_input)
            output[mask] = expert_output.to(output.dtype)

        return residual + output


class V4ReasoningModel(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 d_ff: int = D_FF, n_layers: int = N_LAYERS,
                 n_family_types: int = N_FAMILY_TYPES,
                 n_experts: int = N_EXPERT_GROUPS,
                 max_seq_len: int = MAX_SEQ_LEN,
                 input_vocab_size: int = H1_INPUT_VOCAB_SIZE,
                 output_vocab_size: int = H1_OUTPUT_VOCAB_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_family_types = n_family_types
        self.n_experts = n_experts
        self.max_seq_len = max_seq_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.family_type_emb = nn.Embedding(n_family_types, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # L0-L3: Shared backbone
        self.backbone = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(4)
        ])

        # L4-L5: Expert blocks
        self.expert_blocks = nn.ModuleList([
            ExpertBlock(d_model, n_heads, d_ff, max_seq_len, n_experts, dropout)
            for _ in range(2)
        ])

        # L6-L7: Fusion layers
        self.fusion = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(2)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, family_ids: torch.Tensor = None) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)

        h = self.tok_emb(x) + self.pos_emb(pos)
        if family_ids is not None:
            h = h + self.family_type_emb(family_ids)
        h = self.drop(h)

        for block in self.backbone:
            h = block(h)

        if family_ids is not None:
            expert_indices = torch.zeros(B, T, dtype=torch.long, device=x.device)
            for fam_id, expert_id in FAMILY_EXPERT_MAP.items():
                expert_indices[family_ids == fam_id] = expert_id
        else:
            expert_indices = torch.zeros(B, T, dtype=torch.long, device=x.device)

        for expert_block in self.expert_blocks:
            h = expert_block(h, expert_indices)

        for block in self.fusion:
            h = block(h)

        logits = self.head(self.ln_f(h))
        return logits

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, family_ids: torch.Tensor = None,
                 max_new_tokens: int = 200, temperature: float = 0.0,
                 fixed_family: int = None) -> torch.Tensor:
        self.eval()
        idx = input_ids.clone()
        fids = family_ids.clone() if family_ids is not None else None

        _normalized_id = NORMALIZED_ID
        _conflicts_id = CONFLICTS_ID
        _contradicts_id = TOKEN2ID.get("CONTRADICTS")
        _requirements_id = REQUIREMENTS_ID
        _cycle_nodes_id = TOKEN2ID.get("CYCLE_NODES")
        _root_blocker_id = ROOT_BLOCKER_ID
        _unlock_seq_id = TOKEN2ID.get("UNLOCK_SEQ")
        _conditionals_id = CONDITIONALS_ID

        current_family = 0

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            fids_cond = fids[:, -self.max_seq_len:] if fids is not None else None
            logits = self.forward(idx_cond, fids_cond)
            logits = logits[:, -1, :]

            # Mask input-only sentinel IDs
            for sid in INPUT_ONLY_SENTINEL_IDS:
                if sid < logits.shape[-1]:
                    logits[:, sid] = float('-inf')

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if fids is not None:
                if fixed_family is not None:
                    current_family = fixed_family
                else:
                    tok_val = next_token.item()
                    if tok_val == _normalized_id:
                        current_family = 1
                    elif tok_val == _conflicts_id or tok_val == _contradicts_id:
                        current_family = 2
                    elif tok_val == _requirements_id:
                        current_family = 3
                    elif tok_val == _conditionals_id:
                        current_family = 8
                    elif tok_val == _cycle_nodes_id:
                        current_family = 4
                    elif tok_val == _root_blocker_id or tok_val == _unlock_seq_id:
                        current_family = 1
                fids = torch.cat([fids, torch.full((1, 1), current_family,
                                                   dtype=torch.long, device=idx.device)], dim=1)

            if next_token.item() == EOS_ID:
                break
        return idx


# ── Self-module shim (lets encode_input_ls/compute_family_ids use module globals)
import sys as _sys
_TV = _sys.modules.get(__name__)
if _TV is None:
    # Loaded via importlib with a custom name; create a minimal shim
    import types as _types
    _TV = _types.SimpleNamespace(
        TOKEN2ID=TOKEN2ID,
        ID2TOKEN=ID2TOKEN,
        BOS_ID=BOS_ID,
        EOS_ID=EOS_ID,
        PAD_ID=PAD_ID,
        ENT_TO_TASK=ENT_TO_TASK,
        OP_GEQ_ID=OP_GEQ_ID,
        OP_LEQ_ID=OP_LEQ_ID,
        CD_IF_W_ID=CD_IF_W_ID,
        HAS_COND_YES_ID=HAS_COND_YES_ID,
        HAS_COND_NO_ID=HAS_COND_NO_ID,
    )


# ── Checkpoint loading ───────────────────────────────────────────────────────

def load_model(checkpoint_path=None, device="cpu"):
    """Load the 12M MoE model from safetensors checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("PRE_REASONING_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(
            "Public inference only loads weights-only .safetensors files. "
            "Convert training .pt checkpoints before use."
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Full neural mode requires safetensors. Install with "
            "`pip install pre-reasoning[neural]`."
        ) from exc

    sd = load_file(str(checkpoint_path), device=device)
    model = V4ReasoningModel(
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS,
        n_family_types=N_FAMILY_TYPES, n_experts=N_EXPERT_GROUPS,
        max_seq_len=MAX_SEQ_LEN, input_vocab_size=H1_INPUT_VOCAB_SIZE,
        output_vocab_size=H1_OUTPUT_VOCAB_SIZE, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Simple config object for engine_info compatibility
    class _Cfg:
        max_seq_len = MAX_SEQ_LEN

    meta = {
        "variant_id": "12m_v4",
        "params": model.count_parameters(),
    }
    return model, _Cfg(), meta


# ── Tokenization helpers ────────────────────────────────────────────────────

def decode_tokens(token_ids):
    return [ID2TOKEN.get(tid, f"UNK_{tid}") for tid in token_ids
            if tid != PAD_ID and tid != -100]


def _ent_to_task_id(ent):
    task = ENT_TO_TASK.get(ent)
    if task and task in TOKEN2ID:
        return TOKEN2ID[task]
    return TOKEN2ID.get(ent, PAD_ID)


# ── NL Parsing ──────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "if", "then", "so", "than", "too",
    "very", "can", "will", "just", "should", "would", "could", "may",
    "might", "shall", "do", "does", "did", "has", "have", "had",
    "our", "we", "us", "my", "your", "his", "her", "their", "all",
    "no", "not", "when", "while", "also", "about", "up", "out",
    "get", "got", "getting", "want", "wants", "warn", "warns",
    "whole", "any", "each", "every", "some",
}


def _clean_noun_phrase(text: str) -> str:
    """Clean a raw text fragment into a noun phrase entity name."""
    text = text.strip().rstrip(".,;!?:").lstrip(".,;!?:").strip()
    words = text.split()
    cleaned = [w for w in words if w.lower() not in _STOPWORDS]
    if not cleaned:
        cleaned = [w for w in words if w.strip()]
    return " ".join(cleaned).strip()


def _extract_entities_and_relations(text: str) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str, str]]]:
    """Relation-first parsing: find relations per sentence, extract noun
    phrases on each side as entities. Returns (entity_list, name->ENT_x map, relations).
    Handles DEPENDS_ON, CONTRADICTS, and REQUIRES relations."""
    sentences = re.split(r"[.;!?\n]+", text)
    entity_order = []
    seen_entities = set()
    raw_relations = []

    def _register(name: str) -> str:
        key = name.lower()
        if key and key not in seen_entities:
            seen_entities.add(key)
            entity_order.append(name)
        return key

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        matched = False

        # Check conflicts first (symmetric: A CONTRADICTS B)
        for pattern, _ in CONFLICT_PATTERNS:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                left = _clean_noun_phrase(sentence[:m.start()])
                right = _clean_noun_phrase(sentence[m.end():])
                if left and right:
                    _register(left)
                    _register(right)
                    raw_relations.append((left.lower(), "CONTRADICTS", right.lower(), "conflict"))
                    matched = True
                    break

        if matched:
            continue

        # Check reverse dependency patterns
        for pattern, _ in REVERSE_PATTERNS:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                left = _clean_noun_phrase(sentence[:m.start()])
                right = _clean_noun_phrase(sentence[m.end():])
                if left and right:
                    _register(left)
                    _register(right)
                    raw_relations.append((right.lower(), "DEPENDS_ON", left.lower(), "reverse"))
                    matched = True
                    break

        if matched:
            continue

        # Check forward dependency patterns
        for pattern, _ in FORWARD_PATTERNS:
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                left = _clean_noun_phrase(sentence[:m.start()])
                right = _clean_noun_phrase(sentence[m.end():])
                if left and right:
                    _register(left)
                    _register(right)
                    raw_relations.append((left.lower(), "DEPENDS_ON", right.lower(), "forward"))
                    matched = True
                    break

    ent_map = {}
    for i, name in enumerate(entity_order[:32]):
        ent_map[name.lower()] = f"ENT_{i}"

    relations = []
    for src_name, rel, tgt_name, _ in raw_relations:
        src_ent = ent_map.get(src_name)
        tgt_ent = ent_map.get(tgt_name)
        if src_ent and tgt_ent:
            relations.append((src_ent, rel, tgt_ent))

    return entity_order, ent_map, relations


def _extract_entities(text: str) -> Tuple[List[str], Dict[str, str]]:
    """Extract unique entity names from text and map to ENT_0, ENT_1, etc."""
    entity_order, ent_map, _ = _extract_entities_and_relations(text)
    return entity_order, ent_map


def _parse_relations(text: str, ent_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Parse natural language text into (src, DEPENDS_ON, tgt) triples."""
    _, _, relations = _extract_entities_and_relations(text)
    return relations


def _resolve_entity(text: str, ent_map: Dict[str, str]) -> Optional[str]:
    """Resolve a text fragment to an ENT_x token."""
    cleaned = _clean_noun_phrase(text)
    key = cleaned.lower()
    if key in ent_map:
        return ent_map[key]
    for name, ent_id in ent_map.items():
        if name in key or key in name:
            return ent_id
    return None


# ── Tokenize input (legacy, kept for analyze_blocks compatibility) ────────────

def tokenize_input_structural(relations: List[Tuple[str, str, str]]) -> List[int]:
    """Tokenize structural relations into model input (ss/sl mode)."""
    tokens = [BOS_ID, TOKEN2ID["INPUT"]]
    for src, rel, tgt in relations:
        if src in TOKEN2ID and rel in TOKEN2ID and tgt in TOKEN2ID:
            tokens.extend([TOKEN2ID[src], TOKEN2ID[rel], TOKEN2ID[tgt], TOKEN2ID["SEP"]])
    tokens.extend([TOKEN2ID["QUERY"], TOKEN2ID["ROOT_BLOCKER"]])
    tokens.append(TOKEN2ID["OUTPUT"])
    tokens.append(EOS_ID)
    return tokens


def tokenize_input_language(relations: List[Tuple[str, str, str]],
                            direction="mixed") -> List[int]:
    """Tokenize relations into language input tokens (ls/ll mode)."""
    tokens = [BOS_ID, TOKEN2ID["LANG_IN"]]
    period_id = TOKEN2ID["PERIOD"]

    for src, rel, tgt in relations:
        src_id = _ent_to_task_id(src)
        tgt_id = _ent_to_task_id(tgt)
        if src_id == PAD_ID or tgt_id == PAD_ID:
            continue

        if rel in ("DEPENDS_ON", "REQUIRES"):
            tokens.append(tgt_id)
            tokens.append(TOKEN2ID["enables_w"])
            tokens.append(src_id)
        elif rel == "CONTRADICTS":
            tokens.append(src_id)
            tokens.append(TOKEN2ID["contradicts_w"])
            tokens.append(tgt_id)
        else:
            tokens.extend([src_id, TOKEN2ID.get(rel, PAD_ID), tgt_id])

        tokens.append(period_id)

    tokens.extend([TOKEN2ID["QUERY"], TOKEN2ID["ROOT_BLOCKER"]])
    tokens.append(TOKEN2ID["OUTPUT"])
    tokens.append(EOS_ID)
    return tokens[:1024]


def tokenize_input(relations: List[Tuple[str, str, str]], mode: str = "ls") -> List[int]:
    """Tokenize relations based on mode."""
    if mode in ("ss", "sl"):
        return tokenize_input_structural(relations)
    else:  # ls, ll
        return tokenize_input_language(relations)


# ── Output parsing ──────────────────────────────────────────────────────────

def parse_output_tokens(tokens: List[str]) -> dict:
    """Parse structural output tokens into result dict."""
    result = {
        "root_blockers": [], "unlock_sequence": [],
        "parallel_work": [], "has_cycle": False, "cycle_nodes": [],
        "normalized_relations": [],
        "conflicts": [],
        "requirements": [],
        "dependencies": [],
    }
    mode = None
    _norm_triple = []
    _req_accum = []
    for tok in tokens:
        if tok in ("BOS", "OUTPUT"):
            continue
        elif tok == "EOS":
            break
        elif tok == "NORMALIZED":
            mode = "normalized"
            _norm_triple = []
        elif tok == "ROOT_BLOCKER":
            mode = "root"
        elif tok == "HAS_CYCLE":
            result["has_cycle"] = True
            mode = None
        elif tok == "HAS_CYCLE_YES":
            result["has_cycle"] = True
            mode = None
        elif tok == "HAS_CYCLE_NO":
            result["has_cycle"] = False
            mode = None
        elif tok == "CYCLE_NODES":
            mode = "cycle_nodes"
        elif tok == "UNLOCK_SEQ":
            mode = "unlock"
        elif tok == "PARALLEL_WORK":
            mode = "parallel"
        elif tok in ("REQUIREMENT", "REQUIREMENTS"):
            mode = "requirement"
            _req_accum = []
        elif tok == "NONE":
            if mode == "cycle_nodes":
                pass
            mode = None
        elif tok == "SEP":
            if mode == "normalized" and len(_norm_triple) >= 3:
                src, op, tgt = _norm_triple[0], _norm_triple[1], _norm_triple[2]
                if op == "DEPENDS_ON":
                    result["dependencies"].append((src, tgt))
                elif op in ("CONTRADICTS", "CONFLICTS"):
                    result["conflicts"].append((src, tgt))
                _norm_triple = []
            elif mode == "requirement" and _req_accum:
                ent = _req_accum[0] if _req_accum else None
                op = _req_accum[1] if len(_req_accum) > 1 else None
                val = _req_accum[2] if len(_req_accum) > 2 else None
                if ent:
                    result["requirements"].append((ent, op, val))
                _req_accum = []
            elif mode in ("root", "unlock", "cycle_nodes"):
                pass
        elif tok.startswith("ENT_"):
            if mode == "root":
                result["root_blockers"].append(tok)
            elif mode == "unlock":
                result["unlock_sequence"].append(tok)
            elif mode == "parallel":
                result["parallel_work"].append(tok)
            elif mode == "cycle_nodes":
                result["cycle_nodes"].append(tok)
            elif mode == "normalized":
                _norm_triple.append(tok)
                result["normalized_relations"].append(tok)
                if len(_norm_triple) >= 3:
                    src, op, tgt = _norm_triple[0], _norm_triple[1], _norm_triple[2]
                    if op == "DEPENDS_ON":
                        result["dependencies"].append((src, tgt))
                    elif op in ("CONTRADICTS", "CONFLICTS"):
                        result["conflicts"].append((src, tgt))
                    _norm_triple = []
            elif mode == "requirement":
                _req_accum.append(tok)
        elif tok in ("DEPENDS_ON", "CONTRADICTS", "CONFLICTS", "REQUIRES",
                      "BLOCKS", "UNLOCKS", "ENABLES", "PARALLEL", "SEQUENCES"):
            if mode == "normalized":
                _norm_triple.append(tok)
                result["normalized_relations"].append(tok)
        elif tok in ("GEQ", "LEQ"):
            if mode == "requirement":
                _req_accum.append(tok)
        elif tok.startswith("VALUE_"):
            if mode == "requirement":
                _req_accum.append(tok)
        elif tok in ("HAS_REQ_YES", "HAS_REQ_NO"):
            pass

    if mode == "normalized" and len(_norm_triple) >= 3:
        src, op, tgt = _norm_triple[0], _norm_triple[1], _norm_triple[2]
        if op == "DEPENDS_ON":
            result["dependencies"].append((src, tgt))
        elif op in ("CONTRADICTS", "CONFLICTS"):
            result["conflicts"].append((src, tgt))
    if mode == "requirement" and _req_accum:
        ent = _req_accum[0] if _req_accum else None
        op = _req_accum[1] if len(_req_accum) > 1 else None
        val = _req_accum[2] if len(_req_accum) > 2 else None
        if ent:
            result["requirements"].append((ent, op, val))

    return result


def parse_lang_output_tokens(tokens: List[str]) -> dict:
    """Parse language-format output tokens."""
    result = {
        "root_blockers": [], "unlock_sequence": [],
        "parallel_work": [], "has_cycle": False, "cycle_nodes": [],
        "normalized_relations": [],
        "conflicts": [], "requirements": [], "dependencies": [],
    }
    section = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("BOS", "LANG_OUT"):
            i += 1
            continue
        elif tok == "EOS":
            break

        if tok in ("root_w", "main_w", "first_w") and i + 2 < len(tokens) \
                and tokens[i + 1] == "blocker_w" and tokens[i + 2] == "is":
            section = "root"
            i += 3
            continue

        if tok == "has_w" and i + 2 < len(tokens) \
                and tokens[i + 1] == "a" and tokens[i + 2] == "cycle_w":
            result["has_cycle"] = True
            section = None
            i += 3
            continue

        if tok == "not" and i + 2 < len(tokens) \
                and tokens[i + 1] == "a" and tokens[i + 2] == "cycle_w":
            result["has_cycle"] = False
            section = None
            i += 3
            continue

        if tok == "cycle_w" and section is None and i + 1 < len(tokens) \
                and tokens[i + 1].startswith("TASK_"):
            section = "cycle_nodes"
            i += 1
            continue

        if tok == "together":
            section = "parallel_end"
            i += 1
            continue

        if tok == "PERIOD":
            section = None
            i += 1
            continue

        if tok.startswith("TASK_") and tok in TASK_TO_ENT:
            ent = TASK_TO_ENT[tok]
            if section == "root":
                result["root_blockers"].append(ent)
            elif section == "cycle_nodes":
                result["cycle_nodes"].append(ent)
            elif section == "unlock":
                result["unlock_sequence"].append(ent)
            elif section == "parallel":
                result["parallel_work"].append(ent)
            elif section is None:
                remaining = tokens[i:]
                has_before = "before" in remaining[:50] or "then_w" in remaining[:50]
                has_together = "together" in remaining[:50]
                if has_before:
                    section = "unlock"
                    result["unlock_sequence"].append(ent)
                elif has_together:
                    section = "parallel"
                    result["parallel_work"].append(ent)
                else:
                    section = "unlock"
                    result["unlock_sequence"].append(ent)

        elif tok in ("before", "then_w"):
            if section is None:
                section = "unlock"

        i += 1

    return result


def auto_parse_output(tokens: List[str]) -> dict:
    """Auto-detect structural vs language output and parse."""
    if "LANG_OUT" in tokens:
        return parse_lang_output_tokens(tokens)
    return parse_output_tokens(tokens)


# ── Graph algorithms ─────────────────────────────────────────────────────────

def _find_root_blockers_from_edges(entities, edges):
    has_incoming, has_outgoing = set(), set()
    for src, rel, tgt in edges:
        if rel in DEP_EDGES:
            has_incoming.add(src)
            has_outgoing.add(tgt)
    roots = [e for e in entities if e in has_outgoing and e not in has_incoming]
    if not roots:
        all_dep = has_incoming | has_outgoing
        roots = [e for e in entities if e not in has_incoming and e in all_dep]
    return sorted(roots, key=lambda x: int(x.split("_")[1]) if x.startswith("ENT_") else 0)


def _compute_impact(root, adj_forward):
    """Count how many nodes are transitively unblocked by resolving root."""
    visited = set()
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nb in adj_forward.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited)


# ── encode_input_ls (from eval_unified.py, verbatim) ─────────────────────────

def encode_input_ls(tv, example: dict, max_len: int, sentinel_mode: bool = False) -> List[int]:
    """Encode example using language input (ls mode).
    sentinel_mode: if True, insert OP_GEQ/OP_LEQ sentinel tokens for requirement edges (CM23+).
    """
    TOKEN2ID = tv.TOKEN2ID
    BOS_ID = tv.BOS_ID
    EOS_ID = tv.EOS_ID
    PAD_ID = tv.PAD_ID
    ENT_TO_TASK = tv.ENT_TO_TASK

    # Sentinel token IDs (only used when sentinel_mode=True)
    # CD models define CD_OP_GEQ_ID/CD_OP_LEQ_ID (207/208); fall back to OP_GEQ_ID/OP_LEQ_ID (199/200)
    op_geq_id = getattr(tv, "CD_OP_GEQ_ID", getattr(tv, "OP_GEQ_ID", 199)) if sentinel_mode else None
    op_leq_id = getattr(tv, "CD_OP_LEQ_ID", getattr(tv, "OP_LEQ_ID", 200)) if sentinel_mode else None

    tokens = [BOS_ID, TOKEN2ID["LANG_IN"]]
    period_id = TOKEN2ID["PERIOD"]

    phrasing_type = example.get("phrasing_type", "forward")

    def _task_id(ent):
        task = ENT_TO_TASK.get(ent)
        if task and task in TOKEN2ID:
            return TOKEN2ID[task]
        return TOKEN2ID.get(ent, PAD_ID)

    # Encode dependency and conflict edges
    for edge_idx, (src, rel, tgt) in enumerate(example["edges"]):
        if src not in ENT_TO_TASK or tgt not in ENT_TO_TASK:
            continue

        if rel in ("DEPENDS_ON", "REQUIRES"):
            if phrasing_type == "reverse":
                phr = REVERSE_PHRASINGS[edge_idx % len(REVERSE_PHRASINGS)]
                tokens.append(_task_id(tgt))
                for word in phr:
                    tid = TOKEN2ID.get(word)
                    if tid is not None:
                        tokens.append(tid)
                tokens.append(_task_id(src))
            elif phrasing_type == "temporal_before":
                phr = TEMPORAL_BEFORE_PHRASINGS[edge_idx % len(TEMPORAL_BEFORE_PHRASINGS)]
                tokens.append(_task_id(tgt))
                for word in phr:
                    tid = TOKEN2ID.get(word)
                    if tid is not None:
                        tokens.append(tid)
                tokens.append(_task_id(src))
            elif phrasing_type == "temporal_after":
                phr = TEMPORAL_AFTER_PHRASINGS[edge_idx % len(TEMPORAL_AFTER_PHRASINGS)]
                tokens.append(_task_id(src))
                for word in phr:
                    tid = TOKEN2ID.get(word)
                    if tid is not None:
                        tokens.append(tid)
                tokens.append(_task_id(tgt))
            else:
                phr = FORWARD_PHRASINGS[edge_idx % len(FORWARD_PHRASINGS)]
                tokens.append(_task_id(src))
                for word in phr:
                    tid = TOKEN2ID.get(word)
                    if tid is not None:
                        tokens.append(tid)
                tokens.append(_task_id(tgt))

        elif rel in ("CONFLICTS", "CONTRADICTS"):
            alias_idx = example.get("conflict_alias_idx", edge_idx % len(CONFLICT_ALIASES))
            phr = CONFLICT_ALIASES[alias_idx % len(CONFLICT_ALIASES)]
            tokens.append(_task_id(src))
            for word in phr:
                tid = TOKEN2ID.get(word)
                if tid is not None:
                    tokens.append(tid)
            tokens.append(_task_id(tgt))
        else:
            tokens.append(_task_id(src))
            tid = TOKEN2ID.get(rel)
            if tid is not None:
                tokens.append(tid)
            tokens.append(_task_id(tgt))

        tokens.append(period_id)

    # Encode requirements in language form (VALUE included in input)
    for ent, op, val in example.get("expected_requirements", []):
        if ent not in ENT_TO_TASK:
            continue
        tokens.append(_task_id(ent))
        # CM23: Insert sentinel token AFTER entity, BEFORE alias phrase
        if sentinel_mode:
            if op == "GEQ":
                tokens.append(op_geq_id)
            else:
                tokens.append(op_leq_id)
        if op == "GEQ":
            alias = REQUIREMENT_GEQ_LANG_ALIASES[random.randint(0, len(REQUIREMENT_GEQ_LANG_ALIASES) - 1)]
        else:
            alias = REQUIREMENT_LEQ_LANG_ALIASES[random.randint(0, len(REQUIREMENT_LEQ_LANG_ALIASES) - 1)]
        for word in alias:
            tid = TOKEN2ID.get(word)
            if tid is not None:
                tokens.append(tid)
        val_id = TOKEN2ID.get(val)
        if val_id is not None:
            tokens.append(val_id)
        tokens.append(period_id)

    tokens.extend([TOKEN2ID["QUERY"], TOKEN2ID["ROOT_BLOCKER"]])
    tokens.append(TOKEN2ID["OUTPUT"])

    return tokens[:max_len]


# ── compute_family_ids_for_input (from eval_unified.py, verbatim) ─────────────

def compute_family_ids_for_input(input_ids: List[int], tv) -> List[int]:
    """Compute family_ids for input tokens (V4 expert routing).

    Matches the training script's assign_family_ids behavior exactly:
    - Training used LS-only inputs (language alias tokens, not structural tokens)
    - Dependency relation tokens in LS are language aliases (depends, on, needs,
      requires, enables_w, allows, unlocks_w) which are NOT in dep_tokens, so
      they received family=0 during training
    - Conflict relation tokens in LS are language aliases (conflicts, with_w,
      contradicts_w, clashes) which are NOT in conflict_tokens, so they also
      received family=0 during training
    - Requirement tokens (OP_GEQ_ID, OP_LEQ_ID, VALUE_*) DO appear in LS inputs
      and ARE in req_tokens, so they correctly received family=3
    - Cycle tokens similarly appear in both modes
    - For cross-modal consistency, dep_tokens and conflict_tokens are NOT checked
      for input positions, since the model was only trained on LS inputs where
      those structural tokens never appeared. This ensures SS and LS inputs get
      the same family_ids for equivalent positions.
    """
    TOKEN2ID = tv.TOKEN2ID

    req_tokens = set()
    for name in ("REQUIREMENTS", "GEQ", "LEQ", "HAS_REQ_YES", "HAS_REQ_NO"):
        tid = TOKEN2ID.get(name)
        if tid is not None:
            req_tokens.add(tid)
    # Operator sentinel tokens
    for attr in ("OP_GEQ_ID", "OP_LEQ_ID", "CD_OP_GEQ_ID", "CD_OP_LEQ_ID"):
        tid = getattr(tv, attr, None)
        if tid is not None:
            req_tokens.add(tid)
    # VALUE tokens
    for v in (0, 5, 10, 50, 60, 70, 75, 80, 85, 90, 95, 100):
        tid = TOKEN2ID.get(f"VALUE_{v}")
        if tid is not None:
            req_tokens.add(tid)

    cycle_tokens = set()
    for name in ("CYCLE_NODES", "HAS_CYCLE_YES", "HAS_CYCLE_NO",
                  "IN_CYCLE", "BACK_EDGE", "LOCAL_CYCLE", "NONE"):
        tid = TOKEN2ID.get(name)
        if tid is not None:
            cycle_tokens.add(tid)

    cond_input_tokens = set()
    for attr in ("CD_IF_W_ID", "HAS_COND_YES_ID", "HAS_COND_NO_ID"):
        tid = getattr(tv, attr, None)
        if tid is not None:
            cond_input_tokens.add(tid)

    fam_ids = []
    for tok in input_ids:
        if tok in req_tokens:
            fam_ids.append(3)
        elif tok in cycle_tokens:
            fam_ids.append(4)
        elif tok in cond_input_tokens:
            fam_ids.append(9)
        else:
            fam_ids.append(0)  # structural (includes dep/conflict aliases)

    return fam_ids


# ── F5 closure helpers ────────────────────────────────────────────────────────

def _f5_input_graph_only_h1(entities: List[str], dep_edges: List[tuple]) -> List[int]:
    """Input encoder for F5 closure pass.

    Format: BOS LANG_IN [dep edges via FORWARD_PHRASINGS cycle] QUERY CLOSURE_ASSUM OUTPUT EOS
    """
    tokens = [BOS_ID, TOKEN2ID["LANG_IN"]]
    period_id = TOKEN2ID.get("PERIOD", PERIOD_ID)
    edge_counter = 0
    for edge in dep_edges:
        src, rel, tgt = edge[0], edge[1], edge[2]
        if src not in ENT_TO_TASK or tgt not in ENT_TO_TASK:
            continue
        if rel in ("DEPENDS_ON", "REQUIRES"):
            phr = FORWARD_PHRASINGS[edge_counter % len(FORWARD_PHRASINGS)]
            tokens.append(_ent_to_task_id(src))
            for word in phr:
                if word in TOKEN2ID:
                    tokens.append(TOKEN2ID[word])
            tokens.append(_ent_to_task_id(tgt))
            tokens.append(period_id)
            edge_counter += 1
    query_id = TOKEN2ID.get("QUERY")
    if query_id is not None:
        tokens.append(query_id)
    tokens.append(CLOSURE_ASSUM_ID)
    output_id = TOKEN2ID.get("OUTPUT")
    if output_id is not None:
        tokens.append(output_id)
    tokens.append(EOS_ID)
    return tokens[:MAX_LEN]


def _parse_f13_output(toks, id2token, eos_id, pad_id):
    """Parse Pass A output tokens into (dep_pairs, conflicts, root_blockers, unlock_seq)."""
    names = [id2token.get(t, str(t)) for t in toks if t not in (eos_id, pad_id, 0)]
    dep_pairs, conflicts, root_blockers, unlock = [], [], [], []
    section = None
    i = 0
    while i < len(names):
        tk = names[i]
        if tk in ("NORMALIZED", "UNLOCK_SEQ", "ROOT_BLOCKER", "CONFLICTS",
                  "CONTRADICTS", "REQUIREMENTS", "CYCLE_NODES", "SEP", "OUTPUT",
                  "HAS_REQ_NO", "HAS_REQ_YES", "HAS_CYCLE_NO", "HAS_CYCLE_YES", "NONE"):
            section = tk
            i += 1
            continue
        if tk.startswith("ENT_"):
            if section == "ROOT_BLOCKER":
                root_blockers.append(tk)
            elif section == "UNLOCK_SEQ":
                unlock.append(tk)
            elif section in ("CONFLICTS", "CONTRADICTS") and i + 2 < len(names) and names[i + 2].startswith("ENT_"):
                conflicts.append((tk, names[i + 2]))
                i += 3
                continue
            elif i + 2 < len(names) and names[i + 1] in ("DEPENDS_ON", "REQUIRES") and names[i + 2].startswith("ENT_"):
                dep_pairs.append((tk, names[i + 2]))
                i += 3
                continue
        i += 1
    return dep_pairs, conflicts, root_blockers, unlock


def _topo_renumber(pairs):
    """Renumber nodes to ENT_0.. in topological order."""
    nodes = {e for p in pairs for e in p}
    adj = defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for s, t in pairs:
        adj[s].append(t)
        indeg[t] += 1
    q = deque(sorted(n for n in nodes if indeg[n] == 0))
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    is_dag = (len(order) == len(nodes))
    if not is_dag:
        return pairs, {n: n for n in nodes}, False
    orig2canon = {n: f"ENT_{i}" for i, n in enumerate(order)}
    canon2orig = {v: k for k, v in orig2canon.items()}
    return [(orig2canon[s], orig2canon[t]) for s, t in pairs], canon2orig, True


def _f5_generate(model, dep_edges, device):
    """Run E4 closure on dep_edges. Returns (verdict, pairs)."""
    ents = sorted({e for (s, r, t) in dep_edges for e in (s, t)})
    inp = _f5_input_graph_only_h1(ents, dep_edges)
    it = torch.tensor([inp], dtype=torch.long, device=device)
    fam = torch.full_like(it, H1_TAIL_FAMILY)
    with torch.no_grad():
        out = model.generate(it, fam, max_new_tokens=140, temperature=0.0,
                             fixed_family=H1_TAIL_FAMILY)
    toks = out[0].tolist()[len(inp):]
    verdict, pairs = None, set()
    if CLOSURE_ASSUM_ID in toks:
        j = toks.index(CLOSURE_ASSUM_ID) + 1
        if j < len(toks) and toks[j] == HAS_ASSUMPTIONS_YES_ID:
            verdict = "YES"
            j += 1
        elif j < len(toks) and toks[j] == HAS_ASSUMPTIONS_NO_ID:
            verdict = "NO"
            j += 1
        body = [t for t in toks[j:] if t not in (EOS_ID, 0)]
        ents_out = [TASK_TO_ENT.get(ID2TOKEN.get(t, str(t)), ID2TOKEN.get(t, str(t))) for t in body]
        for k in range(0, len(ents_out) - 1, 2):
            pairs.add((ents_out[k], ents_out[k + 1]))
    return verdict, pairs


def _f5_window(model, window, device, _cache):
    """Run E4 on one minimal window, canonicalized to E4's training distribution.

    The window is topologically renumbered to ENT_0.. so every query lands inside
    the distribution E4 was trained on (<=3 entities / 2 edges here). Identical
    canonical inputs are cached, so the model is consulted once per distinct shape.
    Returns the closure pairs in the window's ORIGINAL entity names.
    """
    canon_pairs, canon2orig, is_dag = _topo_renumber(window)
    if not is_dag:
        return set()
    sig = tuple(sorted(canon_pairs))
    if sig not in _cache:
        _v, cl = _f5_generate(model, [(s, "DEPENDS_ON", t) for s, t in canon_pairs], device)
        _cache[sig] = cl
    return {(canon2orig.get(a, a), canon2orig.get(b, b)) for a, b in _cache[sig]}


def _f5_pass(model, edges, device, max_window_calls=4000):
    """Pass B: F5 transitive closure via spoon-fed windows (E4).

    Instead of feeding the whole graph to E4 (which goes out-of-distribution and
    hallucinates on large/disjoint graphs), the harness chunks the graph into
    minimal 2-hop windows [(A->B),(B->C)], confirms each candidate closure pair
    with E4 (always in-distribution), and iterates rounds until no new pairs
    appear -- yielding the full transitive closure with zero hallucination. This
    mirrors the original spoon-feeding harness.
    """
    dep_edges = [(s, t) for (s, r, t) in edges if r in ("DEPENDS_ON", "REQUIRES")]
    if not dep_edges:
        return None, set()
    direct = set(dep_edges)
    nodes = {e for p in dep_edges for e in p}
    all_edges = set(dep_edges)
    derived = set()
    cache = {}
    calls = 0
    for _ in range(max(1, len(nodes))):
        adj = defaultdict(set)
        for s, t in all_edges:
            adj[s].add(t)
        new_pairs = set()
        for src, mid in list(all_edges):
            for tgt in adj.get(mid, ()):
                if src == tgt:
                    continue
                pair = (src, tgt)
                if pair in direct or pair in derived or pair in new_pairs:
                    continue
                if calls >= max_window_calls:
                    break
                calls += 1
                if pair in _f5_window(model, [(src, mid), (mid, tgt)], device, cache):
                    new_pairs.add(pair)
            if calls >= max_window_calls:
                break
        if not new_pairs:
            break
        derived |= new_pairs
        all_edges = set(dep_edges) | derived
    return ("YES" if derived else "NO"), derived


# ── ReasoningEngineV3 ───────────────────────────────────────────────────────

class ReasoningEngineV3:
    """Neural Reasoning Engine -- 12M MoE v4.

    Full v3 interface with two new result keys:
      derived_assumptions  -- list of {assuming, premise} dicts from F5 closure
      transitive_closure   -- list of {src, tgt, src_name, tgt_name} dicts
      assumption_verdict   -- "YES" / "NO" / None (F5 E4 verdict)
    """

    VERSION = "3.1.0"

    def __init__(self, checkpoint_path=None, device="cpu"):
        self.device = device
        self._checkpoint_path = str(checkpoint_path or DEFAULT_CHECKPOINT)
        self.model, self.cfg, self.meta = load_model(
            self._checkpoint_path, device=device
        )

    def get_form(self) -> dict:
        """Return form describing how to use the engine."""
        return {
            "engine": "ReasoningEngineV3",
            "version": self.VERSION,
            "type": "neural",
            "model": self.meta.get("variant_id", "12m_v4"),
            "params": self.meta.get("params", 0),
            "input": {
                "text": "Natural language describing dependencies between entities.",
                "signal_words": {
                    "forward": ["depends on", "needs", "requires", "relies on",
                                "waits for", "blocked by", "must happen after"],
                    "reverse": ["enables", "allows", "unlocks", "must happen before",
                                "makes possible", "comes before"],
                },
                "examples": [
                    "A enables B. B enables C.",
                    "X depends on Y. Y depends on Z.",
                    "Database setup must happen before API deployment.",
                ],
            },
            "output": {
                "root_blockers": "Entities that must be resolved first (no dependencies).",
                "unlock_sequence": "Topological order for resolution.",
                "has_cycle": "Whether circular dependencies exist.",
                "cycle_nodes": "Entities participating in cycles.",
                "parallel_work": "Independent entities that can proceed in parallel.",
                "derived_assumptions": "Implicit transitive dependencies (F5/E4 closure).",
                "assumption_verdict": "YES if implicit assumptions exist, NO otherwise.",
            },
            "modes": {
                "ll": "Language in, language out (default)",
                "ls": "Language in, structural out",
                "ss": "Structural in, structural out",
                "sl": "Structural in, language out",
            },
        }

    def analyze(self, text: str, mode: str = "ll") -> dict:
        """Full two-pass pipeline: text -> Pass A (F1-F3) + Pass B (F5) -> result."""
        t0 = time.perf_counter()

        entity_order, ent_map, relations = _extract_entities_and_relations(text)

        reverse_map = {}
        for i, name in enumerate(entity_order[:32]):
            ent_id = f"ENT_{i}"
            reverse_map[ent_id] = name
        for name_lower, ent_id in ent_map.items():
            if ent_id not in reverse_map:
                for n in entity_order:
                    if n.lower() == name_lower:
                        reverse_map[ent_id] = n
                        break

        if not relations:
            return self._empty_result(text, mode, time.perf_counter() - t0)

        # PASS A: F1-F3 (b8 auto-routed MoE)
        example = {"edges": relations, "phrasing_type": "forward", "expected_requirements": []}
        inp = encode_input_ls(_TV, example, MAX_LEN, sentinel_mode=True)
        fam = compute_family_ids_for_input(inp, _TV)
        it = torch.tensor([inp], dtype=torch.long, device=self.device)
        ft = torch.tensor([fam], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(it, family_ids=ft, max_new_tokens=160, temperature=0.0)
        toks_a = out[0].tolist()[len(inp):]
        dep_pairs, conflicts_pairs, root_blockers, unlock_seq = _parse_f13_output(
            toks_a, ID2TOKEN, EOS_ID, PAD_ID
        )

        # PASS B: F5 (dedicated E4, family 5)
        verdict, closure_pairs = _f5_pass(self.model, relations, self.device)

        inference_ms = (time.perf_counter() - t0) * 1000

        def _name(ent_id):
            return reverse_map.get(ent_id, ent_id)

        def _name_token(tok):
            if tok.startswith("ENT_") and tok in reverse_map:
                return f"{reverse_map[tok]} [{tok}]"
            if tok.startswith("TASK_") and tok in TASK_TO_ENT:
                ent = TASK_TO_ENT[tok]
                if ent in reverse_map:
                    return f"{reverse_map[ent]} [{tok}]"
            return tok

        # Build adjacency for impact scoring
        adj_forward = defaultdict(list)
        for src, rel, tgt in relations:
            if rel in DEP_EDGES:
                adj_forward[tgt].append(src)

        rb_list = []
        for rb in root_blockers:
            impact = _compute_impact(rb, adj_forward)
            rb_list.append({"entity": rb, "name": _name(rb), "impact": impact})

        parsed = {
            "root_blockers": root_blockers,
            "unlock_sequence": unlock_seq,
            "has_cycle": False,
            "cycle_nodes": [],
            "parallel_work": [],
            "conflicts": conflicts_pairs,
            "requirements": [],
            "dependencies": dep_pairs,
        }

        derived_assumptions = [
            {"assuming": _name(a), "premise": _name(b)}
            for a, b in sorted(closure_pairs)
        ]
        transitive_closure = [
            {"src": a, "tgt": b, "src_name": _name(a), "tgt_name": _name(b)}
            for a, b in sorted(closure_pairs)
        ]

        variant = "12m_v4"
        params = self.model.count_parameters()
        pred_tokens_a = [ID2TOKEN.get(t, str(t)) for t in toks_a if t not in (EOS_ID, PAD_ID, 0)]

        if mode in ("ls", "ss"):
            trace = self._format_structural_trace(
                parsed, reverse_map, entity_order, relations,
                rb_list, root_blockers, unlock_seq,
                False, [], [],
                conflicts_pairs, [], dep_pairs,
                mode, inference_ms, variant, params,
                pred_tokens_a, _name, _name_token,
                derived_assumptions, verdict,
            )
        else:
            trace = self._format_nl_trace(
                rb_list, root_blockers, unlock_seq,
                False, [], [],
                mode, inference_ms, variant, params, _name,
            )

        return {
            "trace": trace,
            "root_blockers": rb_list,
            "unlock_sequence": [{"entity": e, "name": _name(e)} for e in unlock_seq],
            "has_cycle": False,
            "cycle_nodes": [],
            "parallel_work": [],
            "conflicts": [{"a": a, "b": b, "a_name": _name(a), "b_name": _name(b)}
                          for a, b in conflicts_pairs],
            "requirements": [],
            "dependencies": [{"src": s, "tgt": t, "src_name": _name(s), "tgt_name": _name(t)}
                             for s, t in dep_pairs],
            "mode": mode,
            "inference_ms": round(inference_ms, 1),
            "model": variant,
            "params": params,
            "pred_tokens": pred_tokens_a,
            "derived_assumptions": derived_assumptions,
            "transitive_closure": transitive_closure,
            "assumption_verdict": verdict,
        }

    def derive_assumptions(self, edges):
        """F5/E4 transitive closure on a list of (src, tgt) NAME pairs (or (s, rel, t)).

        This is the neural E4 closure pass built into the 12M model.
        It maps entity names to ENT ids, runs the dedicated E4 closure pass
        (with topological canonicalization), and maps the closure back to names.
        Returns (closure_name_pairs, verdict): closure_name_pairs is a set of
        (src_name, tgt_name) transitive pairs; verdict is "YES"/"NO"/None.
        """
        norm = []
        for e in edges:
            if len(e) == 3:
                s, _r, t = e
            else:
                s, t = e
            if s and t:
                norm.append((str(s), str(t)))
        if not norm:
            return set(), None

        # Map unique names -> ENT_i by appearance order (vocab supports ENT_0..31).
        name2ent = {}
        for s, t in norm:
            for n in (s, t):
                if n not in name2ent and len(name2ent) < 32:
                    name2ent[n] = f"ENT_{len(name2ent)}"
        ent2name = {v: k for k, v in name2ent.items()}

        dep_edges = [(name2ent[s], "DEPENDS_ON", name2ent[t])
                     for s, t in norm if s in name2ent and t in name2ent]
        if not dep_edges:
            return set(), None

        verdict, ent_pairs = _f5_pass(self.model, dep_edges, self.device)
        valid = set(name2ent)
        closure = {(ent2name.get(a, a), ent2name.get(b, b)) for a, b in ent_pairs}
        closure = {(a, b) for a, b in closure if a in valid and b in valid}
        return closure, verdict

    def analyze_blocks(self, blocks: list, mode: str = "ll") -> dict:
        """Direct block input (V2-compatible).
        Blocks are dicts with 'family', 'source', 'entities' keys.
        Converts to relations and runs through the neural model.
        """
        t0 = time.perf_counter()

        relations = []
        all_entities = set()
        ent_map = {}
        ent_counter = 0

        for block in blocks:
            entities = block.get("entities", [])
            for e in entities:
                e_clean = e.strip()
                if e_clean and e_clean.lower() not in ent_map:
                    ent_map[e_clean.lower()] = f"ENT_{ent_counter}"
                    ent_counter += 1
                    all_entities.add(e_clean)

        for block in blocks:
            family = block.get("family", "")
            source = block.get("source", block.get("source_clause", ""))
            entities = block.get("entities", [])

            if family in ("dependency", "prereq") and len(entities) >= 2:
                e0 = ent_map.get(entities[0].strip().lower())
                e1 = ent_map.get(entities[1].strip().lower())
                if e0 and e1:
                    relations.append((e0, "DEPENDS_ON", e1))
            elif source:
                parsed = _parse_relations(source, ent_map)
                relations.extend(parsed)

        reverse_map = {v: k for k, v in ent_map.items()}
        entity_names = list(all_entities)

        if not relations:
            return self._empty_result(str(blocks), mode, time.perf_counter() - t0)

        # Use legacy tokenize path for analyze_blocks (blocks don't have NL text)
        token_ids = tokenize_input(relations, mode=mode)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen = self.model.generate(input_tensor, max_new_tokens=250, temperature=0.0)
        pred_ids = gen[0].cpu().tolist()[len(token_ids):]
        pred_tokens = decode_tokens(pred_ids)
        parsed_out = auto_parse_output(pred_tokens)

        inference_ms = (time.perf_counter() - t0) * 1000

        def _name(ent_id):
            return reverse_map.get(ent_id, ent_id)

        adj_forward = defaultdict(list)
        for src, rel, tgt in relations:
            if rel in DEP_EDGES:
                adj_forward[tgt].append(src)

        root_blockers = parsed_out.get("root_blockers", [])
        rb_list = []
        for rb in root_blockers:
            impact = _compute_impact(rb, adj_forward)
            rb_list.append({"entity": rb, "name": _name(rb), "impact": impact})

        variant = "12m_v4"
        params = self.model.count_parameters()
        unlock_sequence = parsed_out.get("unlock_sequence", [])
        has_cycle = parsed_out.get("has_cycle", False)
        cycle_nodes = parsed_out.get("cycle_nodes", [])
        parallel_work = parsed_out.get("parallel_work", [])
        conflicts = parsed_out.get("conflicts", [])
        requirements = parsed_out.get("requirements", [])
        dependencies = parsed_out.get("dependencies", [])

        if mode in ("ls", "ss"):
            def _name_token(tok):
                if tok.startswith("ENT_") and tok in reverse_map:
                    return f"{reverse_map[tok]} [{tok}]"
                if tok.startswith("TASK_") and tok in TASK_TO_ENT:
                    ent = TASK_TO_ENT[tok]
                    if ent in reverse_map:
                        return f"{reverse_map[ent]} [{tok}]"
                return tok
            trace = self._format_structural_trace(
                parsed_out, reverse_map, entity_names, relations,
                rb_list, root_blockers, unlock_sequence,
                has_cycle, cycle_nodes, parallel_work,
                conflicts, requirements, dependencies,
                mode, inference_ms, variant, params,
                pred_tokens, _name, _name_token,
                [], None,
            )
        else:
            trace = self._format_nl_trace(
                rb_list, root_blockers, unlock_sequence,
                has_cycle, cycle_nodes, parallel_work,
                mode, inference_ms, variant, params, _name,
            )

        return {
            "trace": trace,
            "root_blockers": rb_list,
            "unlock_sequence": [{"entity": e, "name": _name(e)} for e in unlock_sequence],
            "has_cycle": has_cycle,
            "cycle_nodes": [{"entity": cn, "name": _name(cn)} for cn in cycle_nodes],
            "parallel_work": [{"entity": e, "name": _name(e)} for e in parallel_work],
            "conflicts": [{"a": a, "b": b, "a_name": _name(a), "b_name": _name(b)}
                          for a, b in conflicts],
            "requirements": [{"entity": e, "operator": op, "value": val,
                              "name": _name(e)} for e, op, val in requirements],
            "dependencies": [{"src": s, "tgt": t, "src_name": _name(s), "tgt_name": _name(t)}
                             for s, t in dependencies],
            "mode": mode,
            "inference_ms": round(inference_ms, 1),
            "model": variant,
            "params": params,
            "pred_tokens": pred_tokens,
            "derived_assumptions": [],
            "transitive_closure": [],
            "assumption_verdict": None,
        }

    def pulse(self, original_problem: str, response: str) -> dict:
        """Backward check: does the response address root blockers?"""
        analysis = self.analyze(original_problem, mode="ll")
        root_blockers = analysis.get("root_blockers", [])

        if not root_blockers:
            return {
                "status": "COMPLETE",
                "message": "No root blockers identified.",
                "gaps": [],
            }

        response_lower = response.lower()
        addressed = []
        gaps = []

        for rb in root_blockers:
            name = rb.get("name", rb.get("entity", ""))
            if name.lower() in response_lower:
                addressed.append(name)
            else:
                gaps.append(name)

        if not gaps:
            return {
                "status": "COMPLETE",
                "message": f"All {len(root_blockers)} root blockers addressed.",
                "addressed": addressed,
                "gaps": [],
            }
        else:
            return {
                "status": "CONTINUE",
                "message": f"{len(gaps)} root blocker(s) not addressed in response.",
                "addressed": addressed,
                "gaps": gaps,
                "suggestion": f"Address: {', '.join(gaps)}",
            }

    def engine_info(self) -> dict:
        """Return engine metadata."""
        return {
            "engine": "ReasoningEngineV3",
            "version": self.VERSION,
            "type": "neural",
            "model": self.meta.get("variant_id", "12m_v4"),
            "params": self.meta.get("params", 0),
            "checkpoint": self._checkpoint_path,
            "device": self.device,
            "vocab_size": H1_OUTPUT_VOCAB_SIZE,
            "max_seq_len": self.cfg.max_seq_len,
            "modes": ["ll", "ls", "ss", "sl"],
            "architecture": "V4ReasoningModel (MoE, 5 experts)",
            "passes": ["Pass A: F1-F3 b8 auto-routed", "Pass B: F5 E4 closure"],
        }

    # ── Internal methods ────────────────────────────────────────────────────

    def _empty_result(self, text, mode, elapsed):
        return {
            "trace": "--- STRUCTURAL TRACE (v3, neural) ---\nNo relations found in input.\n",
            "root_blockers": [],
            "unlock_sequence": [],
            "has_cycle": False,
            "cycle_nodes": [],
            "parallel_work": [],
            "conflicts": [],
            "requirements": [],
            "dependencies": [],
            "pred_tokens": [],
            "raw_input": text[:200],
            "mode": mode,
            "inference_ms": round(elapsed * 1000, 1),
            "model": "12m_v4",
            "derived_assumptions": [],
            "transitive_closure": [],
            "assumption_verdict": None,
        }

    def _format_nl_trace(self, rb_list, root_blockers, unlock_sequence,
                         has_cycle, cycle_nodes, parallel_work,
                         mode, inference_ms, variant, params, _name):
        """Build the NL-formatted trace string (ll/sl modes)."""
        lines = ["--- STRUCTURAL TRACE (v3, neural) ---"]

        if rb_list:
            lines.append("ROOT BLOCKERS (must resolve FIRST):")
            for i, rb in enumerate(rb_list):
                lines.append(f"  [{i+1}] {rb['name']}")
                if rb["impact"] > 0:
                    lines.append(f"    Impact: unblocks {rb['impact']} downstream step(s)")
        else:
            lines.append("ROOT BLOCKERS: None identified")

        lines.append("")
        if unlock_sequence:
            lines.append("UNLOCK SEQUENCE (optimal order):")
            for i, ent in enumerate(unlock_sequence):
                action = "Resolve" if ent in root_blockers else "Unblocked"
                lines.append(f"  Step {i+1}: {action} {_name(ent)}")
        else:
            lines.append("UNLOCK SEQUENCE: N/A (cycle or single node)")

        if parallel_work:
            lines.append("")
            lines.append("PARALLEL WORK (independent):")
            for ent in parallel_work:
                lines.append(f"  {_name(ent)} (no dependencies)")

        lines.append("")
        if has_cycle:
            cycle_names = [_name(cn) for cn in cycle_nodes]
            lines.append(f"CYCLES DETECTED: {', '.join(cycle_names) if cycle_names else 'Yes'}")
            lines.append("  These depend on each other -- use alternative resolution strategy.")
        else:
            lines.append("CYCLES DETECTED: None")

        lines.append("")
        lines.append(f"Model: {variant} ({params:,} params) | Mode: {mode} | Inference: {inference_ms:.0f}ms")

        return "\n".join(lines)

    def _format_structural_trace(self, parsed, reverse_map, entity_names,
                                 relations, rb_list, root_blockers,
                                 unlock_sequence, has_cycle, cycle_nodes,
                                 parallel_work, conflicts, requirements,
                                 dependencies, mode, inference_ms,
                                 variant, params, pred_tokens, _name, _name_token,
                                 derived_assumptions=None, verdict=None):
        """Build the structural trace string (ls/ss modes).
        Shows ALL families the model detected with entity name reverse-mapping.
        Includes F5 closure section at the end."""
        lines = ["=== STRUCTURAL OUTPUT (v3, neural) ===", ""]

        # ── Entity map ──
        lines.append("ENTITIES:")
        for i, name in enumerate(entity_names[:32]):
            lines.append(f"  ENT_{i} = {name}")
        lines.append("")

        # ── Input relations ──
        lines.append("INPUT RELATIONS:")
        for src, rel, tgt in relations:
            lines.append(f"  {_name(src)} [{src}] {rel} {_name(tgt)} [{tgt}]")
        lines.append("")

        # ── Dependencies ──
        if dependencies:
            lines.append("DEPENDENCIES:")
            for src, tgt in dependencies:
                lines.append(f"  {_name(src)} DEPENDS_ON {_name(tgt)}")
        else:
            lines.append("DEPENDENCIES: (none in model output)")
        lines.append("")

        # ── Conflicts ──
        if conflicts:
            lines.append("CONFLICTS:")
            for a, b in conflicts:
                lines.append(f"  {_name(a)} CONTRADICTS {_name(b)}")
        else:
            lines.append("CONFLICTS: (none detected)")
        lines.append("")

        # ── Requirements ──
        if requirements:
            lines.append("REQUIREMENTS:")
            for ent, op, val in requirements:
                op_str = ">=" if op == "GEQ" else "<=" if op == "LEQ" else (op or "?")
                val_str = val.replace("VALUE_", "") if val and val.startswith("VALUE_") else (val or "?")
                lines.append(f"  {_name(ent)} {op_str} {val_str}")
        else:
            lines.append("REQUIREMENTS: (none detected)")
        lines.append("")

        # ── Cycles ──
        if has_cycle:
            cycle_names = [f"{_name(cn)} [{cn}]" for cn in cycle_nodes]
            lines.append("CYCLES: YES")
            if cycle_names:
                lines.append(f"  Nodes: {', '.join(cycle_names)}")
        else:
            lines.append("CYCLES: NO")
        lines.append("")

        # ── Root blockers ──
        if rb_list:
            lines.append("ROOT_BLOCKERS:")
            for i, rb in enumerate(rb_list):
                impact_str = f" (unblocks {rb['impact']})" if rb["impact"] > 0 else ""
                lines.append(f"  [{i+1}] {rb['name']} [{rb['entity']}]{impact_str}")
        else:
            lines.append("ROOT_BLOCKERS: (none)")
        lines.append("")

        # ── Unlock sequence ──
        if unlock_sequence:
            lines.append("UNLOCK_SEQUENCE:")
            for i, ent in enumerate(unlock_sequence):
                marker = " *ROOT*" if ent in root_blockers else ""
                lines.append(f"  {i+1}. {_name(ent)} [{ent}]{marker}")
        else:
            lines.append("UNLOCK_SEQUENCE: (none)")
        lines.append("")

        # ── Parallel work ──
        if parallel_work:
            lines.append("PARALLEL_WORK:")
            for ent in parallel_work:
                lines.append(f"  {_name(ent)} [{ent}]")
        else:
            lines.append("PARALLEL_WORK: (none)")
        lines.append("")

        # ── F5 closure section ──
        if derived_assumptions:
            lines.append("DERIVED ASSUMPTIONS (F5/E4 closure):")
            lines.append(f"  Verdict: {verdict}")
            for da in derived_assumptions:
                lines.append(f"  {da['assuming']} implicitly assumes {da['premise']}")
        else:
            lines.append(f"DERIVED ASSUMPTIONS (F5/E4 closure): none (verdict={verdict})")
        lines.append("")

        # ── Raw token sequence (debugging) ──
        lines.append("--- RAW MODEL OUTPUT TOKENS (Pass A) ---")
        if pred_tokens:
            mapped = [_name_token(t) for t in pred_tokens]
            lines.append("  " + " ".join(mapped))
        else:
            lines.append("  (no tokens)")
        lines.append("")

        lines.append(f"Model: {variant} ({params:,} params) | Mode: {mode} | Inference: {inference_ms:.0f}ms")

        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ReasoningEngineV3 -- 12M MoE Neural Reasoning")
    parser.add_argument("text", nargs="?", help="Problem text to analyze")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, default="ll", choices=["ll", "ls", "ss", "sl"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--info", action="store_true", help="Print engine info")
    parser.add_argument("--form", action="store_true", help="Print form")
    args = parser.parse_args()

    engine = ReasoningEngineV3(checkpoint_path=args.checkpoint, device=args.device)

    if args.info:
        import json
        print(json.dumps(engine.engine_info(), indent=2))
        return

    if args.form:
        import json
        print(json.dumps(engine.get_form(), indent=2))
        return

    if not args.text:
        print("Usage: python -m pre_reasoning.inference 'A enables B. B enables C.'")
        print("       python -m pre_reasoning.inference --info")
        print("       python -m pre_reasoning.inference --form")
        return

    result = engine.analyze(args.text, mode=args.mode)
    print(result["trace"])


if __name__ == "__main__":
    main()
