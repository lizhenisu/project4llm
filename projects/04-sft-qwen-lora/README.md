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

## 参考答案

1. SFT 数据区分 `user` 和 `assistant`，是为了让模型学会“看到用户指令后生成助手回答”的条件分布，而不是把所有文本都当成同一种普通续写。chat template 会把不同角色转换成带 special token 或固定格式的训练文本，模型通过这些格式信号理解哪里是问题、哪里是回答、哪里应该停止生成。如果角色混乱，模型可能学会复述用户输入、生成多余角色标记，或在推理时不遵循对话格式。

2. 只对 assistant answer 计算 loss 更合理，因为 SFT 的目标是训练模型在给定 system/user 上下文后生成高质量回答。user prompt 是条件，不是希望模型模仿输出的目标；如果对 user 内容也算 loss，模型会把容量花在复现问题本身，还可能学会在回答中生成用户角色文本。工程上通常把 prompt、system、user 部分的 labels 置为 `-100`，只保留 assistant token 参与交叉熵。

3. LoRA 减少显存的核心原因是冻结基座模型权重，只训练少量低秩增量参数。对一个原始权重矩阵 `W`，LoRA 不直接更新完整 `W`，而是学习 `BA` 这样的低秩更新，其中 rank `r` 远小于原矩阵维度。这样可训练参数、优化器状态和梯度显存都会显著减少；前向仍使用基座权重加 adapter 增量，所以能以较低成本适配新任务。

4. QLoRA 在 LoRA 基础上把基座模型权重量化到 4-bit，进一步降低权重显存；LoRA adapter 通常仍以较高精度训练。折中是量化/反量化会带来额外计算开销，数值精度可能影响稳定性和效果，对硬件、bitsandbytes、计算 dtype、梯度检查点等配置更敏感。面试表达可以概括为：LoRA 省训练参数和优化器状态，QLoRA 进一步省基座权重显存，但换来量化误差和工程复杂度。
