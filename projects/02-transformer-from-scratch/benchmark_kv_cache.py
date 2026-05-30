import time

import torch

from model import TinyGPT, TinyGPTConfig


def timed_generate(model: TinyGPT, prompt: torch.Tensor, max_new_tokens: int, use_cache: bool) -> float:
    start = time.perf_counter()
    if use_cache:
        model.generate_with_cache(prompt.clone(), max_new_tokens=max_new_tokens)
    else:
        model.generate(prompt.clone(), max_new_tokens=max_new_tokens)
    return time.perf_counter() - start


def main() -> None:
    torch.manual_seed(42)
    config = TinyGPTConfig(
        vocab_size=128,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
    )
    model = TinyGPT(config)
    prompt = torch.randint(0, config.vocab_size, (1, 16))
    max_new_tokens = 32

    no_cache_ids = model.generate(prompt.clone(), max_new_tokens=max_new_tokens)
    cache_ids = model.generate_with_cache(prompt.clone(), max_new_tokens=max_new_tokens)
    print(f"outputs match: {torch.equal(no_cache_ids, cache_ids)}")

    logits, _loss, past_kvs = model(prompt, use_cache=True)
    print(f"prefill logits shape: {tuple(logits.shape)}")
    for layer_idx, (k, v) in enumerate(past_kvs):
        print(f"layer {layer_idx} k shape: {tuple(k.shape)}, v shape: {tuple(v.shape)}")

    no_cache_time = min(timed_generate(model, prompt, max_new_tokens, use_cache=False) for _ in range(5))
    cache_time = min(timed_generate(model, prompt, max_new_tokens, use_cache=True) for _ in range(5))
    speedup = no_cache_time / cache_time if cache_time > 0 else float("inf")

    print(f"no-cache time: {no_cache_time:.4f}s")
    print(f"kv-cache time: {cache_time:.4f}s")
    print(f"speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
