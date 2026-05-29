# 教材审计与优化建议

审计依据：[target.md](../target.md) 的 7 个总体产出和 12 周训练路线。

结论：当前教材已经从“路线规划”升级为“可运行的入门训练营”，但还没有完全达到 `target.md` 中要求的最终项目级产出。它适合作为第 1-7 周的学习起点；第 8-12 周仍需要接入真实开源模型、真实微调和开源项目精读。

## 覆盖情况

| target.md 要求 | 当前状态 | 证据 |
| --- | --- | --- |
| 迷你 Transformer/GPT 项目 | 部分完成 | `projects/02-transformer-from-scratch/model.py`、`train_tiny_gpt.py` |
| Transformers 指令微调项目 | 未完成真实训练 | 目前只有 `projects/04-sft-qwen-lora/build_sft_dataset.py` |
| QLoRA/LoRA 微调项目 | 未完成真实训练 | 有 LoRA 教材，没有 adapter 训练脚本 |
| DPO 偏好优化项目 | 未完成真实训练 | 有 `projects/05-dpo-preference/build_preference_dataset.py` |
| 数据处理流水线 | 入门完成 | `projects/03-tokenizer-and-data/data_pipeline.py` 覆盖清洗、去重、PII、chat text、packing |
| 主流模型结构对比笔记 | 入门完成 | `notes/05-model-architecture-comparison.md` |
| 开源项目阅读与复现 | 模板完成，复现未完成 | `projects/06-open-source-reading/README.md` |

## 这次已优化的点

1. 增加 tiny GPT 训练闭环，不只停留在 forward smoke test。
2. 增加 valid loss 和 perplexity，让训练结果有评估指标。
3. 增加数据清洗、去重、PII 过滤、chat template 和 sequence packing 练习。
4. 增加 GPT/Llama/Qwen/DeepSeek 结构对比笔记。
5. 增加实验报告模板，避免训练后无法面试复述。
6. 扩展开源阅读模板，加入调用链追踪要求。

## 仍然建议优化的地方

### 1. 补真实 LoRA SFT 脚本

当前只构造了 SFT 数据，还没有调用 `transformers`、`peft`、`trl` 训练 adapter。下一步应增加：

```text
projects/04-sft-qwen-lora/train_lora_sft.py
projects/04-sft-qwen-lora/infer_lora_adapter.py
```

要求：

- 支持 `model_name_or_path` 指向本地模型。
- 支持 LoRA `r`、`alpha`、`target_modules` 配置。
- 保存 adapter。
- 加载 adapter 做推理对比。

### 2. 补真实 DPO 脚本

当前只有 preference 数据构造，还没有 `DPOTrainer` 训练。下一步应增加：

```text
projects/05-dpo-preference/train_dpo.py
```

要求：

- 支持本地 SFT 模型或小型基座模型。
- 读取 `prompt/chosen/rejected` JSONL。
- 输出训练前后样例对比。

### 3. 给 TinyGPT 增加 KV cache 版本

`target.md` 明确要求加入 KV cache 并比较推理速度。当前 `TinyGPT.generate` 没有 KV cache。下一步可以增加：

```text
projects/02-transformer-from-scratch/model_kv_cache.py
projects/02-transformer-from-scratch/benchmark_kv_cache.py
```

要求：

- 对比无 cache 和有 cache 的生成耗时。
- 打印每步 K/V shape。
- 解释为什么训练阶段收益不明显。

### 4. 补 Hugging Face `datasets` 练习

当前 tokenizer 练习使用本地列表和 JSONL，尚未演示 `datasets.Dataset` 的 `map`、`train_test_split`、`save_to_disk`。

建议增加：

```text
projects/03-tokenizer-and-data/hf_datasets_pipeline.py
```

### 5. 补开源项目精读记录

当前只有模板，没有真实精读。建议至少补两份：

```text
projects/06-open-source-reading/nanogpt-reading.md
projects/06-open-source-reading/trl-reading.md
```

每份都要包含：

- 最小运行命令。
- 入口文件。
- 数据流。
- 模型构建。
- 训练循环。
- checkpoint。
- 自己改过的实验。

## 当前适合怎么学

现在不要急着直接跑 Qwen LoRA。更合理的顺序是：

1. 跑通 MLP，确认 PyTorch 训练循环。
2. 跑通 tokenizer 和 data pipeline，理解 `input_ids`、`labels`、packing。
3. 跑通 tiny GPT 训练，理解 logits、loss、perplexity、generate。
4. 阅读结构对比笔记，再去看 Hugging Face 模型 config。
5. 准备本地小模型后，再做 LoRA SFT 和 DPO。

## 当前教材等级判断

- 作为入门教材：合格。
- 作为第 1-7 周实践材料：基本合格。
- 作为完整 12 周 LLM 实习冲刺材料：还不够。
- 最大短板：缺真实 Hugging Face/PEFT/TRL 训练项目和开源项目复现记录。
