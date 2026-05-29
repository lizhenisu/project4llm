# Tokenizer 与 Causal LM 数据

## 文本为什么要 tokenize

神经网络不能直接处理字符串，只能处理数字。Tokenizer 的任务是把文本切成 token，再把 token 映射成整数 ID。

常见粒度：

- 字符级：简单，但序列长。
- 词级：直观，但词表巨大，遇到新词困难。
- subword：在词和字符之间折中，LLM 主流使用 BPE、WordPiece 或 SentencePiece。

## BPE 的直觉

BPE 从字符开始，反复合并最常一起出现的相邻片段。常见词会变成较长 token，低频词会拆成更小片段。

优点：

- 词表大小可控。
- 能处理未见过的新词。
- 对多语言和代码比较实用。

## Causal LM 的训练样本

语言模型训练目标是预测下一个 token。

假设文本 token 是：

```text
[10, 20, 30, 40]
```

模型输入可以是：

```text
input_ids = [10, 20, 30]
labels    = [20, 30, 40]
```

在 Hugging Face 的 causal LM 中，常见做法是让：

```python
labels = input_ids.copy()
```

模型内部会自动把 logits 和 labels 错位计算 next-token loss。

## attention_mask

`attention_mask` 表示哪些位置是真实 token，哪些位置是 padding：

- `1`：真实 token。
- `0`：padding。

注意：causal mask 和 attention mask 不是同一个东西。

- causal mask：不允许当前位置看未来 token。
- attention mask：不让模型关注 padding token。

## 必须掌握的 shape

通常：

```text
input_ids:      [batch_size, seq_len]
attention_mask: [batch_size, seq_len]
labels:         [batch_size, seq_len]
logits:         [batch_size, seq_len, vocab_size]
```

交叉熵会在 `vocab_size` 维度上计算每个位置预测下一个 token 的损失。

## 面试口述版

Tokenizer 决定了文本如何映射为 token 序列，会影响序列长度、词表覆盖、训练效率和模型效果。Causal LM 的目标是自回归预测下一个 token，训练时虽然每个位置不能看未来，但可以通过 causal mask 并行计算所有位置的 loss。
