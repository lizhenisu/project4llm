# practice4llm

面向“大模型算法工程师 - LLM 方向 - 实习生”的代码实践教材。

核心原则：每个概念都必须落到代码，每个脚本都必须能跑，每个实验都要能讲清楚“为什么这样做”。

## 使用方式

先激活当前目录已有的虚拟环境：

```bash
source .venv/bin/activate
```

运行第一阶段练习：

```bash
python projects/01-ml-basics/train_mlp.py --epochs 3
python projects/03-tokenizer-and-data/train_bpe_tokenizer.py
python projects/03-tokenizer-and-data/data_pipeline.py
python projects/02-transformer-from-scratch/smoke_test.py
python projects/02-transformer-from-scratch/train_tiny_gpt.py --steps 20
python projects/04-sft-qwen-lora/build_sft_dataset.py
python projects/05-dpo-preference/build_preference_dataset.py
```

## 学习顺序

1. [target.md](target.md)：12 周总路线。
2. [notes/01-pytorch-training-loop.md](notes/01-pytorch-training-loop.md)：PyTorch 训练循环。
3. [projects/01-ml-basics](projects/01-ml-basics)：二分类 MLP。
4. [notes/02-tokenizer-and-causal-lm.md](notes/02-tokenizer-and-causal-lm.md)：Tokenizer 与 causal LM 数据。
5. [projects/03-tokenizer-and-data](projects/03-tokenizer-and-data)：BPE tokenizer 与 labels 构造。
6. [notes/03-transformer-decoder.md](notes/03-transformer-decoder.md)：Decoder-only Transformer。
7. [projects/02-transformer-from-scratch](projects/02-transformer-from-scratch)：从零实现 attention 和 tiny GPT。
8. [notes/04-finetuning-lora-dpo.md](notes/04-finetuning-lora-dpo.md)：SFT、LoRA、DPO。
9. [projects/04-sft-qwen-lora](projects/04-sft-qwen-lora)：SFT 数据与 LoRA 训练准备。
10. [projects/05-dpo-preference](projects/05-dpo-preference)：DPO 偏好数据准备。
11. [notes/05-model-architecture-comparison.md](notes/05-model-architecture-comparison.md)：主流大模型结构对比。
12. [notes/06-experiment-report-template.md](notes/06-experiment-report-template.md)：实验报告模板。
13. [notes/08-milvus-vector-retrieval-for-rag.md](notes/08-milvus-vector-retrieval-for-rag.md)：Milvus 向量检索与 RAG 面试教程。
14. [projects/07-milvus-rag](projects/07-milvus-rag)：Milvus Lite 检索链路实践。
15. [notes/00-weekly-checklist.md](notes/00-weekly-checklist.md)：12 周执行检查表。

## 每个项目的验收方式

- 能运行脚本。
- 能解释输入、输出和张量 shape。
- 能修改 1-2 个关键参数并观察变化。
- 能写一段面试口述总结。
