# 12 周执行检查表

## 第 1-2 周

- 跑通 `projects/01-ml-basics/train_mlp.py`。
- 能解释 `zero_grad`、`backward`、`step`。
- 对比 SGD 和 Adam。
- 写一份训练 loss 记录。

## 第 3 周

- 跑通 `projects/03-tokenizer-and-data/train_bpe_tokenizer.py`。
- 能解释 token、token id、attention mask、labels。
- 改 `vocab_size` 并观察切词变化。

## 第 4-5 周

- 跑通 `projects/02-transformer-from-scratch/smoke_test.py`。
- 能画出 attention 的 Q/K/V shape。
- 能解释 causal mask 和 next-token loss。

## 第 6 周

- 阅读 Hugging Face 中 Llama/Qwen 模型 config。
- 写一张 GPT、Llama、Qwen、DeepSeek 对比表。

## 第 7 周

- 构造 SFT JSONL。
- 统计 token 长度分布。
- 了解 packing 为什么提升训练效率。

## 第 8-9 周

- 跑通一个 LoRA 或 QLoRA SFT。
- 保存 adapter。
- 加载 adapter 做推理对比。

## 第 10 周

- 构造 preference JSONL。
- 跑通一个 DPO 小实验。
- 对比 SFT 和 DPO 输出。

## 第 11 周

- 精读 `nanoGPT` 或 `trl` 的核心路径。
- 画出入口命令到训练循环的调用链。

## 第 12 周

- 整理 3 个项目到简历。
- 每个项目准备 2 分钟口述版本。
- 准备常见追问和实验细节。
