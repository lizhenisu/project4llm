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

## 面试必会问题参考答案

### 基础类

1. 交叉熵和 KL 散度有什么关系？

交叉熵可以写成 `H(P, Q) = H(P) + KL(P || Q)`。监督学习里真实分布 `P` 固定时，`H(P)` 是常数，最小化交叉熵等价于最小化真实分布和模型分布之间的 KL 散度。分类任务中使用 cross entropy，本质是在提高正确类别的预测概率；语言模型中对每个位置做 vocabulary 上的交叉熵，就是让模型分布靠近真实下一个 token 的 one-hot 分布。

2. Adam 和 SGD 的区别是什么？

SGD 直接使用当前 batch 的梯度更新参数，形式简单，对学习率较敏感，但泛化表现常不错。Adam 会维护梯度的一阶矩和二阶矩估计，相当于给不同参数使用自适应学习率，通常收敛更快、调参更容易，是 LLM 训练和微调中常见选择。工程上还会使用 AdamW，把 weight decay 从梯度更新中解耦，避免普通 Adam 中 L2 正则和自适应学习率耦合带来的问题。

3. 过拟合如何判断和处理？

典型现象是训练 loss 持续下降，但验证 loss 上升或验证指标变差，模型在训练集上表现很好，在未见数据上泛化差。处理方法包括增加或清洗数据、早停、减小模型容量、增大 weight decay、加入 dropout、数据增强、降低训练 epoch、检查 train/valid 分布是否一致。对 LLM 微调还要警惕训练数据太少、学习率太大、只记住模板、验证集泄漏等问题。

4. BatchNorm、LayerNorm、RMSNorm 的区别是什么？

BatchNorm 通常沿 batch 维统计均值和方差，常用于 CNN，但在变长序列、小 batch 或自回归推理中不方便。LayerNorm 对单个样本的 hidden 维做归一化，不依赖 batch size，因此更适合 Transformer。RMSNorm 进一步简化 LayerNorm，不减均值，只用均方根做缩放，计算更轻，Llama 等现代 LLM 常用 RMSNorm。

### Transformer 类

1. Self-Attention 的计算复杂度是多少？

标准 self-attention 需要计算所有 query 和 key 的两两相似度，时间复杂度约为 `O(seq_len^2 * hidden)`，attention matrix 的显存复杂度约为 `O(seq_len^2)`。这也是长上下文训练和推理昂贵的主要原因之一。FFN 通常也很耗计算，但 attention 的二次复杂度会随着序列长度增长迅速放大。

2. Multi-Head Attention 为什么有效？

多头注意力把 hidden 分成多个 head，每个 head 在不同子空间里学习注意力模式。有的 head 可能关注局部邻近 token，有的关注长距离依赖，有的关注格式、括号、实体或语法关系。多个 head 的输出拼接后再线性投影，使模型能同时表达多种关系，比单个大 head 更灵活。

3. causal mask 是什么？

causal mask 是下三角 mask，用于禁止当前位置关注未来 token。对第 `t` 个位置，attention 只能看 `0..t` 的 token，不能看 `t+1` 之后的信息。这样训练时虽然整段序列并行输入模型，目标仍然符合自回归生成：每个位置只能根据历史预测下一个 token。

4. RoPE 相比绝对位置编码有什么优势？

绝对位置编码通常把 position embedding 加到 token embedding 上，模型直接记住每个绝对位置。RoPE 在 Q/K 上做旋转变换，使 attention score 自然包含相对位置信息，更适合 decoder-only 语言模型，也更利于长度外推和长上下文扩展。现代 Llama/Qwen 等模型普遍采用 RoPE 或它的变体。

5. KV cache 的原理是什么？

自回归推理每次只生成一个新 token。没有 KV cache 时，每一步都会把完整上下文重新送入模型，重复计算历史 token 的 key/value。KV cache 会在每层保存历史 K/V，下一个 step 只计算新 token 的 Q/K/V，再让新 token 的 Q attend 到缓存的历史 K/V 和当前 K/V。它减少重复计算和显存带宽开销，主要提升推理，不改变训练目标。

6. MHA、MQA、GQA 有什么区别？

MHA 中每个 query head 都有自己的 key/value head，表达能力强但 KV cache 大。MQA 中多个 query head 共享同一组 K/V，KV cache 很小但共享更强。GQA 把 query head 分组，每组共享一组 K/V，是 MHA 和 MQA 的折中。面试中可以用 `num_attention_heads` 和 `num_key_value_heads` 的关系解释：二者相等是 MHA，KV head 为 1 接近 MQA，介于中间是 GQA。

### LLM 结构类

1. Decoder-only 模型为什么能做聊天？

Decoder-only 模型本质是 causal LM，只会根据已有 token 预测下一个 token。聊天能力来自数据格式和对齐训练：把 system/user/assistant 多轮对话序列化成 token，通过 SFT 学会在 user 后生成 assistant，通过 RLHF/DPO 等偏好优化让回答更符合人类偏好。也就是说，模型不是结构上专门有“聊天模块”，而是把聊天建模成条件续写。

2. Llama 和原始 Transformer 有哪些结构差异？

Llama 仍是 decoder-only Transformer，但采用了现代 LLM 常见改造：RoPE 替代绝对位置编码，RMSNorm 替代 LayerNorm，SwiGLU/门控 FFN 替代普通 FFN，部分版本使用 GQA/MQA 降低 KV cache 成本，并通常采用 Pre-Norm 结构。这些变化主要服务于大规模训练稳定性、推理效率和长上下文能力。

3. Qwen/Llama/DeepSeek 的核心特点是什么？

Llama 代表主流开源 decoder-only 架构路线，重点是 RoPE、RMSNorm、SwiGLU、GQA/MQA 和生态复现。Qwen 强调中文/多语言、代码能力、chat template、长上下文和 Transformers 生态适配。DeepSeek 的重点包括 MoE、MLA/KV 优化、推理模型训练、蒸馏和强化学习路线。面试时最好结合具体 config 和源码字段说明，不要只背品牌名。

4. MoE 的优势和问题是什么？

MoE 通过 router 为每个 token 选择少数 expert，做到总参数量很大但单 token 激活参数较少。优势是提升模型容量和专业化能力，同时控制每 token 计算量。问题包括 expert load balancing、跨设备通信、router 稳定性、训练不均衡、部署复杂、小 batch 推理利用率低，以及某些 expert 被过度或不足使用。

### 训练类

1. Pre-training、SFT、RLHF、DPO 的区别是什么？

Pre-training 用海量无标注文本做 next-token prediction，学习通用语言和知识。SFT 用高质量指令数据训练模型按用户需求回答。RLHF 通常包含 SFT、Reward Model、PPO，用人类偏好奖励进一步优化策略。DPO 直接用 chosen/rejected 偏好对优化模型，不单独训练 reward model，也不需要 PPO 在线采样流程，工程链路更短。

2. LoRA 为什么能减少训练参数？

LoRA 冻结原模型权重，只在部分线性层上训练低秩增量 `BA`。因为 rank `r` 远小于原矩阵维度，所以新增可训练参数很少，梯度和优化器状态显存也大幅降低。它适合微调大模型，因为基座能力保留在冻结权重中，adapter 学任务增量。

3. QLoRA 的 4-bit 量化会带来什么影响？

QLoRA 把基座权重量化到 4-bit，显著降低权重显存，使单卡训练更大模型成为可能。代价是量化误差、反量化计算开销、对 dtype/硬件/bitsandbytes 配置敏感，训练速度和稳定性可能受影响。通常 LoRA adapter 仍以较高精度训练，目标是在显存和效果之间折中。

4. gradient accumulation 解决什么问题？

gradient accumulation 用多个小 micro-batch 累加梯度，再执行一次 optimizer step，模拟更大的 effective batch size。它解决单卡显存放不下大 batch 的问题。需要注意 loss 通常要按累计步数缩放，学习率、日志 step、梯度裁剪和 scheduler step 都要按实际 optimizer step 理解。

5. loss 不下降可能是什么原因？

常见原因包括学习率过大或过小、数据 labels 构造错误、attention mask/causal mask 错误、tokenizer 和模型词表不匹配、没有正确训练参数、梯度被截断或为 NaN、batch 太小噪声大、数据质量差、loss mask 把有效 token 全忽略、模型处于 eval/no_grad、优化器没有 step。排查时先在极小数据上 overfit，确认模型能记住小 batch，再扩大数据。

6. 大模型数据清洗有哪些步骤？

常见流程包括格式解析、字段校验、去空样本、去重、语言/长度过滤、质量打分、PII/敏感信息过滤、脏词和乱码过滤、指令和回答格式统一、chat template 渲染、tokenize、截断、packing、train/eval split。SFT 还要确认 user/assistant 角色正确、回答有帮助且不胡编；DPO 要确认 chosen/rejected 在同一 prompt 下有明确偏好差异。

### 工程类

1. 如何处理 OOM？

先确认 OOM 发生在加载、前向、反向还是 optimizer step。常用手段包括减小 batch size/seq_len、增加 gradient accumulation、开启 gradient checkpointing、使用 fp16/bf16、使用 LoRA/QLoRA、减少 LoRA target modules、使用更小模型、开启 FlashAttention 或更省显存 attention、清理无用张量、避免保存过多 logits。LLM 中 seq_len 对显存影响很大，优先检查最大长度和 packing。

2. 如何设计一次微调实验？

先定义目标和评价标准，再固定基座模型、数据版本、训练/验证划分和 prompt 格式。选择少量关键变量，例如 learning rate、LoRA rank、batch size、epoch、max length，每次只改一两个变量。记录命令、环境、随机种子、显存、耗时、loss 曲线、验证指标和样例输出。实验结束要能回答：解决了什么问题、效果如何、代价是什么、下一步怎么改。

3. 如何评估一个微调模型是否变好？

不能只看训练 loss。应同时看验证 loss、固定评测集、人工样例对比、指令遵循、事实性、安全性、格式稳定性和拒答边界。对任务型模型可以设计小型 gold set；对聊天模型可以用固定 prompts 做 before/after 对比；对 DPO 要看 chosen 风格是否泛化到未见问题。还要检查是否退化：输出变短、模板化、胡编增多、过度拒答或忘记基座能力。

4. 如何读一个大模型训练开源项目？

先跑最小示例，再从入口命令追踪参数解析、配置加载、数据集读取、tokenize/collator、模型构建、forward、loss、backward、optimizer/scheduler、checkpoint 和 eval。不要一开始读所有文件，先画主调用链，再深入关键模块。读完要改一个小参数或模块并复现实验现象，这样才能确认自己理解了真实执行路径。

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
