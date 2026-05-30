import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TinyGPTConfig:
    vocab_size: int = 128
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1


LayerKVCache = tuple[torch.Tensor, torch.Tensor]
KVCache = list[LayerKVCache]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(
        self,
        x: torch.Tensor,
        past_kv: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKVCache]:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)

        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        total_len = past_len + seq_len
        if total_len > self.causal_mask.size(-1):
            raise ValueError("sequence length exceeds block_size")

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, past_len:total_len, :total_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_dropout(self.proj(out))
        if use_cache:
            return out, (k, v)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKVCache]:
        attn_out = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        present_kv = None
        if use_cache:
            attn_out, present_kv = attn_out
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if use_cache:
            return x, present_kv
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_kvs: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, KVCache]:
        _batch, seq_len = input_ids.shape
        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].size(-2)
        if past_len + seq_len > self.config.block_size:
            raise ValueError("sequence length exceeds block_size")
        if labels is not None and use_cache:
            raise ValueError("labels are only supported when use_cache=False")

        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        present_kvs = []
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        for block, past_kv in zip(self.blocks, past_kvs, strict=True):
            if use_cache:
                x, present_kv = block(x, past_kv=past_kv, use_cache=True)
                present_kvs.append(present_kv)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        if use_cache:
            return logits, loss, present_kvs
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    @torch.no_grad()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        self.eval()
        if input_ids.size(1) + max_new_tokens > self.config.block_size:
            raise ValueError("generate_with_cache requires prompt length + new tokens <= block_size")

        logits, _loss, past_kvs = self(input_ids, use_cache=True)
        for step in range(max_new_tokens):
            next_token_logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if step < max_new_tokens - 1:
                logits, _loss, past_kvs = self(next_id, past_kvs=past_kvs, use_cache=True)
        return input_ids
