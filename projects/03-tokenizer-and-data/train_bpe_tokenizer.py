from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_DIR / "tiny_corpus.txt"
TOKENIZER_PATH = PROJECT_DIR / "tiny_bpe_tokenizer.json"


def write_tiny_corpus() -> None:
    lines = [
        "Large language models predict the next token.",
        "A tokenizer maps text into token ids.",
        "Causal language modeling uses a mask to block future tokens.",
        "Transformer attention computes queries keys and values.",
        "LoRA fine tuning trains small low rank adapters.",
        "大模型通过预测下一个 token 学习语言模式。",
        "数据清洗和 tokenizer 会影响训练效率。",
    ]
    CORPUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=120,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train([str(CORPUS_PATH)], trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer


def pad_to_length(ids: list[int], pad_id: int, max_length: int) -> tuple[list[int], list[int]]:
    ids = ids[:max_length]
    attention_mask = [1] * len(ids)
    while len(ids) < max_length:
        ids.append(pad_id)
        attention_mask.append(0)
    return ids, attention_mask


def main() -> None:
    write_tiny_corpus()
    tokenizer = train_tokenizer()

    text = "Transformer models predict the next token."
    encoded = tokenizer.encode(text)
    pad_id = tokenizer.token_to_id("[PAD]")
    input_ids, attention_mask = pad_to_length(encoded.ids, pad_id, max_length=16)
    labels = input_ids.copy()

    print(f"text: {text}")
    print(f"tokens: {encoded.tokens}")
    print(f"ids: {encoded.ids}")
    print(f"input_ids: {input_ids}")
    print(f"attention_mask: {attention_mask}")
    print(f"labels: {labels}")
    print(f"saved tokenizer: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
