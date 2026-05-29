import json
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent / "tiny_preference.jsonl"


EXAMPLES = [
    {
        "prompt": "解释 KV cache 的作用。",
        "chosen": "KV cache 在自回归推理中缓存历史 token 的 key 和 value，生成新 token 时只需计算新位置的 K/V，从而减少重复计算并提升推理速度。",
        "rejected": "KV cache 是训练时用来让模型记住所有答案的缓存。",
    },
    {
        "prompt": "什么是 DPO？",
        "chosen": "DPO 是一种直接偏好优化方法，使用 chosen/rejected 偏好对训练模型，不需要显式训练 reward model，也避免了 PPO 的复杂在线强化学习流程。",
        "rejected": "DPO 是一种 tokenizer 压缩算法，可以减少词表大小。",
    },
    {
        "prompt": "为什么要做数据清洗？",
        "chosen": "数据清洗可以去除重复、低质量、格式错误和敏感内容，降低训练噪声，提高模型学习效率和最终输出质量。",
        "rejected": "数据清洗主要是为了让文件名更短。",
    },
]


def main() -> None:
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for example in EXAMPLES:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"wrote: {OUT_PATH}")
    print(OUT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
