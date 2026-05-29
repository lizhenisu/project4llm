import json
import re
from pathlib import Path
from statistics import mean

from tokenizers import Tokenizer


PROJECT_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = PROJECT_DIR / "tiny_bpe_tokenizer.json"
RAW_PATH = PROJECT_DIR / "raw_sft.jsonl"
CLEAN_PATH = PROJECT_DIR / "clean_sft.jsonl"
PACKED_PATH = PROJECT_DIR / "packed_token_ids.jsonl"


RAW_EXAMPLES = [
    {
        "instruction": "解释 causal mask 的作用。",
        "input": "",
        "output": "causal mask 会阻止当前位置看到未来 token，使模型只能基于历史上下文预测下一个 token。",
    },
    {
        "instruction": "解释 causal mask 的作用。",
        "input": "",
        "output": "causal mask 会阻止当前位置看到未来 token，使模型只能基于历史上下文预测下一个 token。",
    },
    {
        "instruction": "LoRA 为什么省显存？",
        "input": "",
        "output": "LoRA 只训练低秩增量矩阵，减少可训练参数、梯度和优化器状态。",
    },
    {
        "instruction": "我的手机号是 13800138000，请记住。",
        "input": "",
        "output": "好的。",
    },
    {
        "instruction": "hi",
        "input": "",
        "output": "ok",
    },
]


def write_raw() -> None:
    with RAW_PATH.open("w", encoding="utf-8") as f:
        for item in RAW_EXAMPLES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def has_pii(text: str) -> bool:
    phone_pattern = r"1[3-9]\d{9}"
    email_pattern = r"[\w.-]+@[\w.-]+\.\w+"
    return re.search(phone_pattern, text) is not None or re.search(email_pattern, text) is not None


def quality_ok(item: dict[str, str]) -> bool:
    merged = item["instruction"] + item.get("input", "") + item["output"]
    if len(item["instruction"]) < 4 or len(item["output"]) < 8:
        return False
    return not has_pii(merged)


def to_chat_text(item: dict[str, str]) -> str:
    user = item["instruction"]
    if item.get("input"):
        user += "\n" + item["input"]
    return f"<|user|>\n{user}\n<|assistant|>\n{item['output']}\n"


def clean_examples() -> list[dict[str, str]]:
    seen = set()
    cleaned = []
    for item in RAW_EXAMPLES:
        key = (item["instruction"], item.get("input", ""), item["output"])
        if key in seen:
            continue
        seen.add(key)
        if quality_ok(item):
            cleaned.append(item)
    return cleaned


def pack_sequences(sequences: list[list[int]], max_length: int, eos_id: int) -> list[list[int]]:
    packed = []
    current: list[int] = []
    for seq in sequences:
        seq = seq + [eos_id]
        if len(current) + len(seq) > max_length and current:
            packed.append(current)
            current = []
        current.extend(seq[:max_length])
    if current:
        packed.append(current)
    return packed


def main() -> None:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("先运行 train_bpe_tokenizer.py 生成 tiny_bpe_tokenizer.json")

    write_raw()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("[EOS]")
    cleaned = clean_examples()
    chat_texts = [to_chat_text(item) for item in cleaned]
    token_ids = [tokenizer.encode(text).ids for text in chat_texts]
    lengths = [len(ids) for ids in token_ids]
    packed = pack_sequences(token_ids, max_length=64, eos_id=eos_id)

    with CLEAN_PATH.open("w", encoding="utf-8") as f:
        for item, text, ids in zip(cleaned, chat_texts, token_ids):
            row = {"messages_text": text, "token_count": len(ids), **item}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with PACKED_PATH.open("w", encoding="utf-8") as f:
        for ids in packed:
            f.write(json.dumps({"input_ids": ids, "labels": ids.copy()}, ensure_ascii=False) + "\n")

    print(f"raw_count={len(RAW_EXAMPLES)}")
    print(f"clean_count={len(cleaned)}")
    print(f"removed_count={len(RAW_EXAMPLES) - len(cleaned)}")
    print(f"token_length_min={min(lengths)}")
    print(f"token_length_mean={mean(lengths):.2f}")
    print(f"token_length_max={max(lengths)}")
    print(f"packed_sequences={len(packed)}")
    print(f"wrote: {CLEAN_PATH}")
    print(f"wrote: {PACKED_PATH}")


if __name__ == "__main__":
    main()
