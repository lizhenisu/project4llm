# 02-transformer-from-scratch

目标：从零实现 Decoder-only Transformer 的核心模块。

## 运行

```bash
python projects/02-transformer-from-scratch/smoke_test.py
python projects/02-transformer-from-scratch/train_tiny_gpt.py --steps 50
```

脚本会验证：

1. tiny GPT 前向传播。
2. logits shape 是否正确。
3. causal LM loss 是否能计算。
4. generate 是否能自回归生成 token。

`train_tiny_gpt.py` 会在本地小语料上训练一个字符级 tiny GPT，并打印 train loss、valid loss、perplexity 和生成样例。

## 任务

1. 实现 token embedding。
2. 实现 causal self-attention。
3. 实现 multi-head attention。
4. 实现 Transformer block。
5. 实现 autoregressive generate。
6. 可选：实现 KV cache。

## 推荐文件

- `attention.py`
- `model.py`
- `train_tiny_gpt.py`
- `generate.py`
- `notes.md`
