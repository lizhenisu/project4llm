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

## 参考答案

1. 同一句话的 token 切分反映了 tokenizer 的词表和训练语料偏好。高频词或常见片段通常会合并成较长 token，低频词、生僻词、错别字、代码符号和多语言混合内容更容易被拆成多个 subword。面试中要能说明：tokenizer 不只是预处理工具，它会影响序列长度、训练成本、上下文利用率和模型对不同语言/领域文本的覆盖能力。

2. `vocab_size` 越大，tokenizer 能保存的合并片段越多，同一文本通常会被切得更粗，序列更短，但 embedding/lm head 参数量更大，低频 token 学得也可能不充分。`vocab_size` 越小，序列会变长，模型每次处理的 token 数增加，长文本更容易超过上下文窗口，但低频词可以通过更小片段组合出来。观察时重点比较 token 数量、罕见词拆分方式和中文/英文/代码符号的变化。

3. padding 后，真实 token 的 `attention_mask` 是 `1`，补齐位置是 `0`。训练和推理时模型应该忽略 padding 位置，否则 padding 会参与 attention 或 loss，污染表示和梯度。causal mask 解决“不能看未来”的问题，attention mask 解决“不能看 padding”的问题，两者作用不同，经常会在实现中组合使用。

4. Causal LM 的目标是预测下一个 token，因此 `labels` 常直接复制 `input_ids`，再由模型内部把 logits 和 labels 错位：第 `t` 个位置的 logits 用来预测第 `t+1` 个 token。这样做的好处是数据管道简单，整段序列可以一次并行前向计算所有位置的 next-token loss。需要注意，padding 位置或不希望训练的 prompt/user 部分通常要在 labels 中置为 `-100`，让交叉熵忽略。
