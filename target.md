# 大模型算法工程师 LLM 方向实习特训计划

目标：从“知道概念但不会操作”提升到能读懂主流 LLM 代码、复现实验、完成小规模训练/微调/RLHF 项目，并能在面试中讲清楚原理、工程取舍和实验结果。

周期建议：12 周。每天 2-4 小时，周末做一次完整代码实验和复盘。

## 总体产出

完成后应至少拥有以下可展示材料：

1. 一个从零实现的迷你 Transformer / GPT 训练项目。
2. 一个基于 Hugging Face Transformers 的指令微调项目。
3. 一个 QLoRA / LoRA 微调项目，最好使用 Qwen 或 Llama 系列小模型。
4. 一个偏好优化项目，覆盖 DPO，了解 PPO/RLHF 的核心流程。
5. 一个数据处理流水线：清洗、去重、tokenize、packing、构造 SFT 数据。
6. 一份主流大模型结构对比笔记：GPT、Llama、Qwen、DeepSeek。
7. 至少阅读并复现 2 个优秀开源项目的核心流程。

## 环境准备

推荐环境：

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- Tokenizers
- Accelerate
- PEFT
- TRL
- bitsandbytes
- wandb 或 tensorboard

建议先建立统一工程目录：

```text
practice4llm/
  notes/
  projects/
    01-ml-basics/
    02-transformer-from-scratch/
    03-tokenizer-and-data/
    04-sft-qwen-lora/
    05-dpo-preference/
    06-open-source-reading/
  experiments/
  checkpoints/
```

## 第 1-2 周：机器学习与深度学习基础补齐

目标：能手写常见算法，理解训练循环、梯度、损失函数、过拟合、正则化和评估。

必须掌握：

- 线性回归、逻辑回归、Softmax 回归
- 决策树、随机森林、GBDT/XGBoost 的基本思想
- MLP、CNN、RNN/LSTM/GRU 的结构和适用场景
- 交叉熵、MSE、KL 散度
- 梯度下降、Adam、学习率调度、权重衰减
- train/valid/test 划分，指标评估，过拟合诊断

代码任务：

1. 用 PyTorch 写一个二分类 MLP。
2. 手写一个训练循环：forward、loss、backward、optimizer.step。
3. 对比 SGD、Adam、不同学习率对 loss 曲线的影响。
4. 用 sklearn 或 PyTorch 做一个文本分类 baseline。

验收标准：

- 能解释 `loss.backward()` 之后发生了什么。
- 能讲清楚训练集 loss 下降但验证集指标下降的原因和处理方法。
- 能独立写出一个最小 PyTorch 训练脚本。

## 第 3 周：NLP 基础与 Transformer 前置知识

目标：理解文本如何变成模型可训练的数据。

必须掌握：

- one-hot、word2vec、embedding
- BPE、WordPiece、SentencePiece
- token、vocab、special token、padding、attention mask
- 语言模型、masked LM、causal LM
- perplexity

代码任务：

1. 使用 `tokenizers` 训练一个小型 BPE tokenizer。
2. 用 `datasets` 加载文本数据，完成 tokenize、截断、padding。
3. 实现 causal language modeling 的 label 构造。
4. 计算一个小语言模型的 perplexity。

验收标准：

- 能说明 tokenizer 为什么会影响模型效果。
- 能解释 `input_ids`、`attention_mask`、`labels` 的含义。
- 能说清楚 causal LM 为什么训练时可以并行。

## 第 4-5 周：从零实现 Transformer / GPT

目标：真正理解 LLM 的核心结构，而不是只会调用 API。

必须掌握：

- Self-Attention
- Multi-Head Attention
- Causal Mask
- Position Embedding、RoPE
- LayerNorm / RMSNorm
- FFN / SwiGLU
- Residual Connection
- KV Cache
- Pre-Norm 与 Post-Norm

代码任务：

1. 从零实现 Multi-Head Causal Self-Attention。
2. 从零实现一个 GPT block。
3. 训练一个 character-level 或 tiny-token-level GPT。
4. 实现 autoregressive generate。
5. 加入 KV cache，比较推理速度。

推荐参考项目：

- `karpathy/nanoGPT`
- `karpathy/llm.c`
- `rasbt/LLMs-from-scratch`

验收标准：

- 能画出 Transformer decoder block 的数据流。
- 能解释 Q、K、V 的形状变化。
- 能说明 causal mask 的作用。
- 能解释 KV cache 为什么只在推理时有明显收益。

## 第 6 周：主流模型结构对比

目标：理解 Qwen、DeepSeek、Llama、GPT 的共同点和差异。

重点比较：

- GPT：Decoder-only causal LM 范式
- Llama：RoPE、RMSNorm、SwiGLU、GQA/MQA、SentencePiece/tokenizer 设计
- Qwen：多语言能力、tokenizer、上下文长度扩展、工程生态
- DeepSeek：MoE、MLA、推理模型训练、蒸馏与强化学习路线

代码任务：

1. 用 Transformers 加载一个小型 Qwen 或 Llama 模型。
2. 打印并阅读模型 config。
3. 对照源码定位 attention、MLP、norm、position embedding。
4. 写一份结构对比表。

建议阅读源码：

- Hugging Face Transformers 中的 `modeling_llama.py`
- Hugging Face Transformers 中的 Qwen/Qwen2 相关实现
- DeepSeek 官方技术报告和开源模型说明

验收标准：

- 能解释 decoder-only 模型如何做问答。
- 能说清楚 MHA、MQA、GQA 的区别。
- 能解释 MoE 的优点和训练/推理难点。

## 第 7 周：大模型数据处理

目标：掌握训练前最容易被低估但最关键的数据流程。

必须掌握：

- 数据清洗
- 去重
- 质量过滤
- PII/敏感信息过滤
- 指令数据格式
- chat template
- sequence packing
- train/eval split

代码任务：

1. 清洗一份 JSONL 指令数据。
2. 构造 Alpaca 或 ShareGPT 风格 SFT 数据。
3. 使用 tokenizer 的 chat template 生成训练文本。
4. 实现 packing，减少 padding 浪费。
5. 统计 token 长度分布。

验收标准：

- 能判断一条数据是否适合 SFT。
- 能解释为什么数据质量通常比训练技巧更重要。
- 能说清楚 packing 对训练效率的影响。

## 第 8-9 周：Fine-tuning / SFT / LoRA / QLoRA

目标：能独立完成一次小模型指令微调。

必须掌握：

- Full fine-tuning
- LoRA
- QLoRA
- rank、alpha、target modules
- gradient accumulation
- mixed precision
- checkpoint
- eval loss 与人工评估

推荐模型：

- Qwen2.5-0.5B-Instruct 或 Qwen2.5-1.5B-Instruct
- TinyLlama
- Llama 系列小尺寸开源模型

代码任务：

1. 用 PEFT 对 Qwen 小模型做 LoRA SFT。
2. 用 TRL 的 `SFTTrainer` 跑通训练。
3. 修改 LoRA rank、学习率、batch size，比较效果。
4. 保存 adapter，并加载 adapter 做推理。
5. 写一份实验报告：数据、参数、显存、loss、样例输出。

推荐参考项目：

- Hugging Face `trl`
- Hugging Face `peft`
- Qwen 官方 cookbook
- LLaMA-Factory

验收标准：

- 能解释 LoRA 为什么省显存。
- 能说明 LoRA 的 target modules 应该如何选。
- 能处理 OOM、loss 不下降、输出乱码等常见问题。

## 第 10 周：RLHF、DPO 与偏好优化

目标：理解大模型对齐的核心流程，并完成一个小规模 DPO 实验。

必须掌握：

- SFT
- Reward Model
- PPO
- RLHF
- DPO
- preference dataset
- chosen/rejected
- KL penalty

代码任务：

1. 构造或下载一份 chosen/rejected 偏好数据。
2. 使用 TRL 的 `DPOTrainer` 跑通小规模 DPO。
3. 对比 SFT 模型和 DPO 后模型的输出。
4. 写一页笔记解释 DPO 相比 PPO 的工程优势。

验收标准：

- 能讲清楚 RLHF 三阶段：SFT、RM、PPO。
- 能解释 DPO 为什么不需要显式训练 reward model。
- 能说明偏好数据质量对结果的影响。

## 第 11 周：开源项目精读与复现

目标：从优秀项目中学习真实工程组织方式。

建议精读项目：

- `karpathy/nanoGPT`：最小 GPT 训练闭环
- `huggingface/transformers`：模型结构与通用接口
- `huggingface/trl`：SFT、DPO、PPO 训练器
- `huggingface/peft`：LoRA/QLoRA 实现
- `hiyouga/LLaMA-Factory`：大模型微调工程实践
- `vllm-project/vllm`：高性能推理、PagedAttention

阅读方法：

1. 先跑通 README 的最小示例。
2. 找到入口脚本。
3. 画出调用链。
4. 只读核心路径，不陷入所有边角逻辑。
5. 修改一个参数或模块，观察行为变化。

验收标准：

- 能讲清楚一个训练项目从命令行参数到模型保存的完整调用链。
- 能在开源项目中定位模型、数据、训练循环、评估逻辑。
- 能基于项目改出一个自己的小实验。

## 第 12 周：面试项目整理

目标：把学习成果转化为能在简历和面试中表达的项目。

建议包装 3 个项目：

### 项目 1：MiniGPT 从零训练

可写亮点：

- 使用 PyTorch 从零实现 Decoder-only Transformer。
- 支持 causal mask、RoPE、KV cache 和自回归生成。
- 在小规模语料上训练并分析 loss/perplexity。

### 项目 2：基于 Qwen 的 LoRA 指令微调

可写亮点：

- 构建指令数据清洗、chat template、tokenize、packing 流水线。
- 使用 PEFT/TRL 完成 LoRA 或 QLoRA 微调。
- 对比不同 rank、学习率、batch size 对效果和显存的影响。

### 项目 3：DPO 偏好优化实验

可写亮点：

- 构造 chosen/rejected 偏好数据。
- 基于 TRL 实现 DPO 训练。
- 对比 SFT 与 DPO 模型在安全性、遵循指令、回答偏好上的差异。

## 每日训练模板

每天按这个顺序执行：

1. 30 分钟读原理。
2. 60-120 分钟写代码或跑实验。
3. 20 分钟记录 bug、参数、结果。
4. 10 分钟写面试口述版总结。

每次实验必须记录：

- 实验目标
- 数据来源
- 模型和参数
- 训练命令
- 显存和耗时
- loss/metric 曲线
- 样例输出
- 遇到的问题
- 下一步改进

## 面试必会问题清单

基础类：

- 交叉熵和 KL 散度有什么关系？
- Adam 和 SGD 的区别是什么？
- 过拟合如何判断和处理？
- BatchNorm、LayerNorm、RMSNorm 的区别是什么？

Transformer 类：

- Self-Attention 的计算复杂度是多少？
- Multi-Head Attention 为什么有效？
- causal mask 是什么？
- RoPE 相比绝对位置编码有什么优势？
- KV cache 的原理是什么？
- MHA、MQA、GQA 有什么区别？

LLM 结构类：

- Decoder-only 模型为什么能做聊天？
- Llama 和原始 Transformer 有哪些结构差异？
- Qwen/Llama/DeepSeek 的核心特点是什么？
- MoE 的优势和问题是什么？

训练类：

- Pre-training、SFT、RLHF、DPO 的区别是什么？
- LoRA 为什么能减少训练参数？
- QLoRA 的 4-bit 量化会带来什么影响？
- gradient accumulation 解决什么问题？
- loss 不下降可能是什么原因？
- 大模型数据清洗有哪些步骤？

工程类：

- 如何处理 OOM？
- 如何设计一次微调实验？
- 如何评估一个微调模型是否变好？
- 如何读一个大模型训练开源项目？

## 第一阶段立即执行任务

从今天开始，先完成以下 5 个任务：

1. 创建 `projects/01-ml-basics/`，写一个 PyTorch 二分类 MLP。
2. 创建 `notes/01-pytorch-training-loop.md`，解释训练循环每一步。
3. 创建 `projects/02-transformer-from-scratch/`，准备实现 attention。
4. 安装并测试 `torch`、`transformers`、`datasets`、`accelerate`、`peft`、`trl`。
5. 选定第一批复现项目：`nanoGPT`、`trl`、`LLaMA-Factory`。

## 学习原则

- 每个概念必须对应一段代码。
- 每个训练脚本必须保存实验记录。
- 不追求一开始训练大模型，先把小模型闭环跑通。
- 不只看教程，必须读源码和改源码。
- 面试表达要围绕“问题、方法、实验、结果、反思”组织。
