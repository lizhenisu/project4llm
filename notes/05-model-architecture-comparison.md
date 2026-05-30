# GPT、Llama、Qwen、DeepSeek 结构对比

这份笔记用于面试前快速复盘。具体模型版本会变化，面试时应以你实际阅读的 config 和源码为准。

## 共同基础

主流 LLM 大多采用 decoder-only causal language model：

- 输入 token ids。
- 经过 embedding 和位置信息。
- 堆叠多层 Transformer decoder block。
- 每个位置只能看自己及之前 token。
- 输出 vocabulary 上的 logits，训练目标是 next-token prediction。

## GPT 系列

重点理解：

- decoder-only causal LM 范式。
- 自回归生成：每次预测一个 next token。
- 指令能力通常来自 SFT、RLHF 或其他对齐阶段。

面试表达：

GPT 的核心不是“聊天结构”，而是 causal LM。聊天能力来自把多轮对话格式化成 token 序列，并通过 SFT/RLHF/DPO 等方法让模型学会遵循指令。

## Llama 系列

常见结构特征：

- RMSNorm。
- RoPE 位置编码。
- SwiGLU/门控 FFN。
- GQA 或 MQA 用于减少 KV cache 体积。
- decoder-only causal LM。

面试表达：

Llama 相比原始 Transformer decoder，更偏现代 LLM 工程设计：使用 RoPE 表示相对位置信息，RMSNorm 简化归一化，SwiGLU 提升 FFN 表达能力，GQA/MQA 改善长上下文推理时的 KV cache 成本。

## Qwen 系列

常见关注点：

- 多语言和代码能力。
- tokenizer 与 chat template。
- 长上下文支持。
- Hugging Face/Transformers 生态适配较完整。

面试表达：

用 Qwen 做实习项目时，不要只说“调用模型”。应该能展示你读过 config，知道 hidden size、layer 数、attention head、max position、special tokens、chat template，以及这些参数如何影响显存和序列长度。

## DeepSeek 系列

常见关注点：

- MoE：每个 token 只激活部分 expert，提高参数规模和计算效率之间的平衡。
- MLA 等注意力优化思路：降低长上下文推理的 KV cache 压力。
- 推理模型路线：强化学习、蒸馏、偏好优化等对齐方法。

面试表达：

MoE 的优势是总参数量大但每个 token 的激活参数较少，难点包括 expert load balancing、通信开销、训练稳定性和部署复杂度。不能只说“参数多所以强”，要能讲出稀疏激活的计算逻辑。

## MHA、MQA、GQA

- MHA：每个 query head 都有自己的 key/value head。
- MQA：多个 query head 共享一组 key/value head。
- GQA：多个 query head 分成若干组，每组共享 key/value head，是 MHA 和 MQA 的折中。

核心取舍：

- MHA 表达能力强，但 KV cache 大。
- MQA KV cache 小，但共享更强。
- GQA 常用于在效果和推理效率之间折中。

## 对比表

| 模型 | 基本范式 | 重点结构 | 工程关注 |
| --- | --- | --- | --- |
| GPT | Decoder-only | causal self-attention | SFT/RLHF 后形成聊天能力 |
| Llama | Decoder-only | RoPE、RMSNorm、SwiGLU、GQA/MQA | 开源生态、推理效率 |
| Qwen | Decoder-only | tokenizer/chat template、长上下文 | 中文/多语言、微调生态 |
| DeepSeek | Decoder-only/MoE 路线 | MoE、注意力/KV 优化、强化学习路线 | 稀疏激活、推理成本、蒸馏 |

## 自查问题

1. 你能从 config 里找到 vocab size、hidden size、layer 数、head 数吗？
2. 你能解释 attention head 数和 KV head 数不一致意味着什么吗？
3. 你能说明 RoPE、RMSNorm、SwiGLU 分别替代或改进了什么吗？
4. 你能解释 MoE 为什么不是“所有参数每次都参与计算”吗？

## 参考答案

1. 在 Hugging Face config 里，常见字段包括 `vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`intermediate_size`、`max_position_embeddings`、`rope_theta` 等。`vocab_size` 决定 embedding 和 lm head 的词表维度；`hidden_size` 是每个 token 表示的宽度；`num_hidden_layers` 是 decoder block 层数；`num_attention_heads` 决定 query head 数。读 config 时要能把这些字段和参数量、显存、KV cache、上下文长度联系起来。

2. attention head 数和 KV head 数不一致通常表示使用了 MQA 或 GQA。query head 仍然很多，用来保持表达能力；key/value head 更少，被多个 query head 共享，从而减少推理时 KV cache 的大小和带宽开销。如果 `num_attention_heads=32`、`num_key_value_heads=8`，可以理解为每 4 个 query head 共享一组 K/V。代价是 K/V 表达被共享，理论上自由度低于完整 MHA，但工程上常能在效果和推理效率之间取得好平衡。

3. RoPE 改进的是位置信息注入方式：它不再简单给 token embedding 加绝对位置向量，而是在 Q/K 上做旋转，使 attention 分数天然包含相对位置信息，更适合长度外推和现代 decoder-only 模型。RMSNorm 简化 LayerNorm，主要按均方根缩放，不减均值，计算更简洁，训练大模型时常见。SwiGLU 改进传统 FFN 的激活结构，用门控分支增强非线性表达，通常比普通 GELU FFN 有更好的效果/参数效率。

4. MoE 的核心是稀疏激活：模型有很多 expert，但每个 token 经过 router 选择 top-k 个 expert 参与计算，而不是所有 expert 都跑一遍。例如 64 个 expert 中每个 token 只激活 2 个，则总参数量很大，但单 token 的计算量接近少数 expert 的成本。它的优势是扩大模型容量同时控制计算量，难点是 router 负载均衡、expert 间通信、训练稳定性、推理部署和小 batch 下的硬件利用率。
