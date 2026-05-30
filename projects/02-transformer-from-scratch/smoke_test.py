import torch

from model import TinyGPT, TinyGPTConfig


def main() -> None:
    torch.manual_seed(42)
    config = TinyGPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32)
    model = TinyGPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(input_ids, labels=input_ids)
    prompt = input_ids[:, :4]
    generated = model.generate(prompt, max_new_tokens=4)
    generated_cached = model.generate_with_cache(prompt, max_new_tokens=4)

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"generated shape: {tuple(generated.shape)}")
    print(f"generated_cached shape: {tuple(generated_cached.shape)}")
    print(f"cache matches no-cache: {torch.equal(generated, generated_cached)}")
    print(f"generated ids[0]: {generated[0].tolist()}")


if __name__ == "__main__":
    main()
