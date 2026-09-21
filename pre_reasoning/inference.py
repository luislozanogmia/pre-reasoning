"""Inference-only runtime for the bundled Pre-Reasoning 1M checkpoint.

This module intentionally contains only the architecture required to reconstruct
the released weights and deterministic greedy decoding. Training, optimizer, data,
and checkpoint-management code are not part of the public package.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 259
MODEL_PARAMS = 1_019_580
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pre-reasoning-1m.safetensors"
)


@dataclass(frozen=True)
class ModelConfig:
    sequence_len: int = 256
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 120


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class _Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


def _has_value_embedding(layer: int, layers: int) -> bool:
    return layer % 2 == (layers - 1) % 2


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class _Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer: int):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = _Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = _Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = _Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = _Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            _Linear(self.ve_gate_channels, config.n_head, bias=False)
            if _has_value_embedding(layer, config.n_layer)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        q = self.c_q(x).view(batch, length, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, length, self.n_head, self.head_dim)
        v = self.c_v(x).view(batch, length, self.n_head, self.head_dim)
        if value_embedding is not None:
            value_embedding = value_embedding.view(
                batch, length, self.n_head, self.head_dim
            )
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * value_embedding
        q = 1.2 * _norm(_rotary(q, cos, sin))
        k = 1.2 * _norm(_rotary(k, cos, sin))
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
        return self.c_proj(y.contiguous().view(batch, length, width))


class _MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = _Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = _Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class _Block(nn.Module):
    def __init__(self, config: ModelConfig, layer: int):
        super().__init__()
        self.attn = _Attention(config, layer)
        self.mlp = _MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(_norm(x), value_embedding, cos, sin)
        return x + self.mlp(_norm(x))


class PreReasoningModel(nn.Module):
    """Exact inference architecture for the released 1M weights."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [_Block(config, layer) for layer in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = _Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = _Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.zeros(1))
        value_width = config.n_embd
        self.value_embeds = nn.ModuleDict(
            {
                str(layer): nn.Embedding(config.vocab_size, value_width)
                for layer in range(config.n_layer)
                if _has_value_embedding(layer, config.n_layer)
            }
        )
        cos, sin = self._rotary_cache(config.sequence_len * 10)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rotary_cache(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.n_embd // self.config.n_head
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (100000 ** (channels / head_dim))
        positions = torch.arange(length, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        cos = frequencies.cos()[None, :, None, :]
        sin = frequencies.sin()[None, :, None, :]
        return cos, sin

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.cos.size(1):
            raise ValueError(
                f"Input length {length} exceeds model limit {self.cos.size(1)}"
            )
        x = _norm(self.transformer.wte(token_ids).float())
        if length > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24])
            )
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x0 = x
        middle = self.config.n_layer // 2
        middle_state = None
        cos = self.cos[:, :length].to(device=x.device, dtype=x.dtype)
        sin = self.sin[:, :length].to(device=x.device, dtype=x.dtype)
        for layer, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer] * x + self.x0_lambdas[layer] * x0
            value_embedding = (
                self.value_embeds[str(layer)](token_ids).to(x.dtype)
                if str(layer) in self.value_embeds
                else None
            )
            x = block(x, value_embedding, cos, sin)
            if layer == middle:
                middle_state = x
        if middle_state is not None:
            x = x - self.backout_lambda.to(x.dtype) * middle_state
        logits = self.lm_head(_norm(x))[..., : self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    checkpoint_path: str | os.PathLike[str] | None = None,
    device: str = "auto",
) -> tuple[PreReasoningModel, ModelConfig, dict]:
    """Strictly load the released architecture; incompatible weights fail loudly."""
    from safetensors.torch import load_file

    resolved_device = resolve_device(device)
    path = Path(
        checkpoint_path
        or os.environ.get("PRE_REASONING_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    )
    if not path.is_file():
        raise FileNotFoundError(f"Pre-Reasoning checkpoint not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError("Only weights-only .safetensors checkpoints are accepted")
    config = ModelConfig()
    model = PreReasoningModel(config)
    weights = load_file(str(path), device="cpu")
    model.load_state_dict(weights, strict=True)
    model = model.to(resolved_device).eval()
    params = model.count_parameters()
    if params != MODEL_PARAMS:
        raise RuntimeError(f"Unexpected parameter count: {params} != {MODEL_PARAMS}")
    return model, config, {
        "variant_id": "pre-reasoning-1m",
        "params": params,
        "checkpoint": str(path),
        "strict_load": True,
    }


def encode_prompt(text: str) -> list[int]:
    return [BOS_ID, *(byte + BYTE_OFFSET for byte in text.encode("utf-8"))]


def decode_completion(token_ids: Iterable[int]) -> str:
    raw = bytearray()
    for token in token_ids:
        if token == EOS_ID:
            break
        if token < BYTE_OFFSET:
            raise ValueError(f"Unexpected special token {token} in completion")
        raw.append(token - BYTE_OFFSET)
    return raw.decode("utf-8")


@torch.inference_mode()
def generate_completion(
    model: PreReasoningModel,
    prompt: str,
    max_new_tokens: int = 192,
) -> str:
    tokens = encode_prompt(prompt)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [tokens], dtype=torch.long, device=model.get_device()
        )
        next_token = int(model(input_ids)[:, -1].argmax(dim=-1).item())
        if next_token == EOS_ID:
            return decode_completion(generated)
        generated.append(next_token)
        tokens.append(next_token)
    raise RuntimeError("Model did not terminate within the generation limit")


__all__ = [
    "BOS_ID",
    "BYTE_OFFSET",
    "DEFAULT_CHECKPOINT",
    "EOS_ID",
    "MODEL_PARAMS",
    "PAD_ID",
    "ModelConfig",
    "PreReasoningModel",
    "decode_completion",
    "encode_prompt",
    "generate_completion",
    "load_model",
    "resolve_device",
]
