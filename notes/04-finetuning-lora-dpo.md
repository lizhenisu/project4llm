# SFT、LoRA、QLoRA 与 DPO

## Pre-training

预训练使用海量无标注文本做 next-token prediction，让模型学习语言、知识和通用模式。

特点：

- 数据量极大。
- 成本极高。
- 目标通常是 causal LM loss。

## SFT

SFT 是 supervised fine-tuning，用高质量指令数据让模型学会按用户指令回答。

数据通常包含：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

或聊天格式：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## LoRA

LoRA 冻结原模型参数，只在部分线性层旁边训练低秩矩阵：

```text
W' = W + BA
```

其中 `A` 和 `B` 的参数量远小于原始 `W`。

常见 target modules：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

## QLoRA

QLoRA 把基座模型量化到 4-bit，再训练 LoRA adapter。它进一步降低显存占用，但可能带来训练速度和数值精度上的折中。

## RLHF

经典 RLHF 三阶段：

1. SFT：让模型学会基本回答格式。
2. Reward Model：用 chosen/rejected 数据训练奖励模型。
3. PPO：用奖励模型指导策略模型优化。

## DPO

DPO 直接使用 chosen/rejected 偏好对优化模型，不显式训练 reward model，也不需要 PPO 的在线采样流程。

工程优点：

- 流程更简单。
- 更稳定。
- 更容易在中小规模实验中跑通。

## 面试口述版

Pre-training 让模型具备通用能力，SFT 让模型学会遵循指令，RLHF/DPO 让模型更符合人类偏好。LoRA 通过低秩增量矩阵减少可训练参数，QLoRA 进一步用 4-bit 量化降低显存。DPO 相比 PPO 的工程链路更短，不需要单独训练 reward model。
