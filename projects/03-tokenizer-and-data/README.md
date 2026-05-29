# 03-tokenizer-and-data

目标：理解文本如何变成 LLM 训练样本。

## 运行

```bash
python projects/03-tokenizer-and-data/train_bpe_tokenizer.py
python projects/03-tokenizer-and-data/data_pipeline.py
```

脚本会完成：

1. 构造一份本地小语料。
2. 训练 BPE tokenizer。
3. 编码一段文本。
4. 构造 causal LM 的 `input_ids`、`attention_mask`、`labels`。
5. 打印 token 和 ID 的对应关系。

## 你要观察什么

- 同一句话会被切成哪些 token。
- `vocab_size` 改变后 token 粒度如何变化。
- padding 后 `attention_mask` 如何变化。
- `labels` 为什么通常复制自 `input_ids`。

`data_pipeline.py` 会演示指令数据清洗、去重、敏感信息过滤、chat template、token 长度统计和 sequence packing。
