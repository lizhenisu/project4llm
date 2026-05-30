# Decoder-only Transformer

## 一句话理解

Decoder-only Transformer 是一个反复堆叠的模块，每层都做两件事：

1. 用 causal self-attention 汇总当前位置之前的信息。
2. 用 MLP/FFN 对每个位置的表示做非线性变换。

## 数据流

```text
token ids
  -> token embedding
  -> position information
  -> Transformer block x N
  -> final norm
  -> lm head
  -> logits over vocabulary
```

## Self-Attention

输入 `x` 的 shape：

```text
[batch, seq_len, hidden]
```

线性投影得到：

```text
q, k, v: [batch, heads, seq_len, head_dim]
```

注意力分数：

```text
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
scores: [batch, heads, seq_len, seq_len]
```

加 causal mask 后，每个位置只能关注自己和之前的位置。

## 为什么要 Multi-Head

不同 head 可以学习不同关系，例如局部依赖、长距离依赖、语法关系、格式关系。最后把多个 head 的结果拼接，再做一次线性投影。

## Pre-Norm

现代 LLM 常用 Pre-Norm：

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

Pre-Norm 通常比 Post-Norm 更容易稳定训练深层模型。

## KV Cache

自回归推理一次只生成一个新 token。没有 KV cache 时，每一步都要重新计算全部历史 token 的 K/V。KV cache 会缓存历史 K/V，只对新 token 计算一次，从而减少重复计算。

KV cache 主要提升推理，不改变训练目标。

## 面试口述版

Decoder-only 模型通过 causal mask 做自回归建模，每个位置预测下一个 token。核心模块是 masked multi-head self-attention 和 MLP，配合残差连接和归一化稳定训练。推理时使用 KV cache 复用历史 K/V，降低逐 token 生成的计算开销。

## 关键追问参考答案

1. Q、K、V 的 shape 怎么变化？

输入 hidden state 通常是 `[batch, seq_len, hidden]`。经过线性层投影后得到 Q/K/V，仍可看成 `[batch, seq_len, hidden]`，随后按 head 拆分成 `[batch, seq_len, n_head, head_dim]`，再转置为 `[batch, n_head, seq_len, head_dim]`。attention score 是 `q @ k.transpose(-2, -1)`，shape 为 `[batch, n_head, seq_len, seq_len]`。最后 `attn @ v` 得到每个 head 的输出，再拼回 `[batch, seq_len, hidden]`。

2. causal mask 为什么训练时仍能并行？

训练时整段序列一次性输入模型，GPU 可以同时计算所有位置的 Q/K/V 和 attention score。causal mask 只是在 score 上把未来位置置为 `-inf`，softmax 后未来位置概率为 0。因此每个位置的信息约束仍满足自回归，但计算不需要像推理一样逐 token 循环。

3. KV cache 为什么主要用于推理？

推理时每一步只新增一个 token，如果每步都重新前向完整上下文，历史 token 的 K/V 会反复计算。KV cache 把每层历史 K/V 存下来，新一步只算新 token 的 K/V 并追加到 cache。训练时 teacher forcing 一次处理完整序列，历史 K/V 没有跨 step 重复计算，所以标准训练中收益不明显。
