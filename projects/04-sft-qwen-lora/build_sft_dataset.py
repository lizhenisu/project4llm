import json
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "tiny_sft.jsonl"


EXAMPLES = [
    {
        "instruction": "解释 causal mask 的作用。",
        "input": "",
        "output": "causal mask 用于阻止当前位置看到未来 token，使 decoder-only 模型只能基于当前位置及其之前的上下文预测下一个 token。",
    },
    {
        "instruction": "LoRA 为什么能节省显存？",
        "input": "",
        "output": "LoRA 冻结原模型参数，只训练低秩增量矩阵，因此反向传播需要保存和更新的参数更少，优化器状态也更小。",
    },
    {
        "instruction": "给出 PyTorch 训练循环的核心步骤。",
        "input": "",
        "output": "核心步骤是前向计算 logits，计算 loss，清空梯度，执行 loss.backward 计算梯度，最后调用 optimizer.step 更新参数。",
    },
]


def to_messages(example: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    user_content = example["instruction"]
    if example["input"]:
        user_content += "\n" + example["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def main() -> None:
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for example in EXAMPLES:
            f.write(json.dumps(to_messages(example), ensure_ascii=False) + "\n")

    print(f"wrote: {OUT_PATH}")
    print(OUT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
