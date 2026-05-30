# 02-transformer-from-scratch

目标：从零实现 Decoder-only Transformer 的核心模块。

## 运行

```bash
python projects/02-transformer-from-scratch/smoke_test.py
python projects/02-transformer-from-scratch/train_tiny_gpt.py --steps 50
python projects/02-transformer-from-scratch/benchmark_kv_cache.py
```

脚本会验证：

1. tiny GPT 前向传播。
2. logits shape 是否正确。
3. causal LM loss 是否能计算。
4. generate 是否能自回归生成 token。
5. KV cache 版本 generate 是否和普通 generate 结果一致。

`train_tiny_gpt.py` 会在本地小语料上训练一个字符级 tiny GPT，并打印 train loss、valid loss、perplexity 和生成样例。
`benchmark_kv_cache.py` 会比较普通生成和 KV cache 生成的耗时，并打印每层 K/V cache 的 shape。

## 任务

1. 实现 token embedding。
2. 实现 causal self-attention。
3. 实现 multi-head attention。
4. 实现 Transformer block。
5. 实现 autoregressive generate。
6. 实现 KV cache，并比较无 cache 和有 cache 的生成耗时。

## KV cache 参考答案

自回归生成每一步只新增一个 token，但普通 `generate` 为了得到最后一个位置的 logits，会把完整上下文重新送进模型。这样历史 token 的 K/V 会被重复计算：生成第 1 个 token 计算 prompt 的 K/V，生成第 2 个 token 又重新计算 prompt+第 1 个 token 的 K/V，序列越长重复越多。

KV cache 的做法是在每层 attention 中缓存历史 token 投影后的 key/value，下一步只对新 token 计算 Q/K/V，再把新 K/V 拼到 cache 后面。新 token 的 query 会和“历史 K/V + 当前 K/V”做 attention，因此输出和完整重算应保持一致。当前实现中 cache shape 是：

```text
k/v: [batch, n_head, cached_seq_len, head_dim]
```

KV cache 主要提升推理，因为推理是逐 token 生成，历史上下文会被反复使用。训练时通常一次输入完整序列，并用 causal mask 并行计算所有位置的 loss；每个 token 的 K/V 本来只在这次 forward 中计算一次，不存在逐步生成里的大量重复重算，所以 KV cache 对标准 teacher-forcing 训练没有明显收益，反而会增加状态管理复杂度。

## 推荐文件

- `attention.py`
- `model.py`
- `train_tiny_gpt.py`
- `benchmark_kv_cache.py`
- `notes.md`
