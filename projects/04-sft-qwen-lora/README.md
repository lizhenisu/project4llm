# 04-sft-qwen-lora

目标：理解 SFT 数据格式、chat template、LoRA 微调参数，并为后续真实模型微调做准备。

当前目录先提供离线可运行的数据构造练习；真实 Qwen LoRA 训练需要能下载模型或本地已有模型权重。

## 离线练习

```bash
python projects/04-sft-qwen-lora/build_sft_dataset.py
```

脚本会输出一份 JSONL，每行是一条 messages 格式的 SFT 样本。

## 真实训练时的关键参数

- `model_name_or_path`：基座模型路径，例如 Qwen 小模型。
- `learning_rate`：LoRA 常用 `1e-4` 到 `2e-4` 起步。
- `per_device_train_batch_size`：单卡 batch size。
- `gradient_accumulation_steps`：用多步累计模拟更大 batch。
- `lora_r`：低秩维度，常从 8、16、32 试。
- `lora_alpha`：LoRA 缩放系数。
- `target_modules`：通常包含 attention 和 MLP 的线性层。

## 面试必须能讲清楚

- 为什么 SFT 数据要区分 user 和 assistant。
- 为什么只对 assistant answer 计算 loss 更合理。
- LoRA 为什么减少显存。
- QLoRA 相比 LoRA 多了什么量化折中。
