import argparse
import math
from pathlib import Path

import torch

from model import TinyGPT, TinyGPTConfig


TEXT = """
Large language models predict the next token.
The transformer decoder uses causal self attention.
Queries keys and values are projected from hidden states.
The causal mask prevents each position from seeing future tokens.
LoRA trains low rank adapters while the base model stays frozen.
Preference optimization compares chosen answers with rejected answers.
数据清洗、tokenizer、packing 和评估共同决定微调质量。
"""


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i : i + block_size] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model: TinyGPT, data: torch.Tensor, batch_size: int, block_size: int) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        x, y = get_batch(data, batch_size, block_size)
        _logits, loss = model(x, labels=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    stoi, itos = build_vocab(TEXT)
    data = encode(TEXT, stoi)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    valid_data = data[split - args.block_size :]

    config = TinyGPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.1,
    )
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        _logits, loss = model(x, labels=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.steps:
            valid_loss = estimate_loss(model, valid_data, args.batch_size, args.block_size)
            ppl = math.exp(valid_loss)
            print(
                f"step={step:03d} "
                f"train_loss={loss.item():.4f} "
                f"valid_loss={valid_loss:.4f} "
                f"valid_ppl={ppl:.2f}"
            )

    prompt = "The "
    input_ids = encode(prompt, stoi).unsqueeze(0)
    generated = model.generate(input_ids, max_new_tokens=80, temperature=0.8, do_sample=True)[0]
    out_path = Path(__file__).resolve().parent / "generated_sample.txt"
    out_path.write_text(decode(generated, itos), encoding="utf-8")
    print(f"generated sample: {out_path}")
    print(decode(generated, itos))


if __name__ == "__main__":
    main()
