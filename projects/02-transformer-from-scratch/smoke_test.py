import torch

from model import TinyGPT, TinyGPTConfig


def print_eos_progress(batch_size: int = 8, eos_token_id: int = 63) -> None:
    finished = torch.zeros(batch_size, dtype=torch.bool)
    non_eos_token_id = 0

    print("eos progress:")
    for step in range(batch_size):
        next_id = torch.full((batch_size, 1), non_eos_token_id, dtype=torch.long)
        next_id[step, 0] = eos_token_id

        finished_mask = finished[:, None]
        eos_ids = torch.full_like(next_id, eos_token_id)
        next_id = torch.where(finished_mask, eos_ids, next_id)
        finished = finished | (next_id[:, 0] == eos_token_id)

        finished_flags = [int(v) for v in finished.tolist()]
        print(f"  step {step + 1}: next_id={next_id[:, 0].tolist()} finished={finished_flags}")


def main() -> None:
    torch.manual_seed(42)
    config = TinyGPTConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4, n_embd=32)
    model = TinyGPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(input_ids, labels=input_ids)
    prompt = input_ids[:, :4]
    generated = model.generate(prompt, max_new_tokens=4)
    generated_cached = model.generate_with_cache(prompt, max_new_tokens=4)
    eos_prompt = torch.randint(0, config.vocab_size, (8, 4))
    eos_token_id = int(eos_prompt[0, -1])
    eos_prompt[:, -1] = eos_token_id
    stopped = model.generate(eos_prompt, max_new_tokens=32, eos_token_id=eos_token_id, min_new_tokens=4)
    stopped_cached = model.generate_with_cache(eos_prompt, max_new_tokens=32, eos_token_id=eos_token_id, min_new_tokens=4)

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"generated shape: {tuple(generated.shape)}")
    print(f"generated_cached shape: {tuple(generated_cached.shape)}")
    print(f"cache matches no-cache: {torch.equal(generated, generated_cached)}")
    print(f"stops on eos: {stopped.size(0) == 8 and stopped.size(1) >= eos_prompt.size(1) + 4}")
    print(f"cached stops on eos: {stopped_cached.size(0) == 8 and stopped_cached.size(1) >= eos_prompt.size(1) + 4}")
    print(f"generated ids[0]: {generated[0].tolist()}")
    print_eos_progress(batch_size=8, eos_token_id=config.vocab_size - 1)


if __name__ == "__main__":
    main()
