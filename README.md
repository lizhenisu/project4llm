# practice4llm

面向「大模型算法工程师—LLM 方向—实习生」的代码实践教材。项目通过 Vibe Coding 完成。核心原则：每个概念落到代码，每个脚本都能跑，每个实验都能讲清楚"为什么这样做"。

## 项目总览

```
practice4llm/
├── target.md                     # 12 周学习路线总纲（含面试必会问题清单）
├── notes/                        # 理论笔记（每篇对应一个学习阶段）
│   ├── 00-weekly-checklist.md          # 12 周执行检查表
│   ├── 01-pytorch-training-loop.md     # PyTorch 训练循环
│   ├── 02-tokenizer-and-causal-lm.md   # Tokenizer 与 Causal LM 数据
│   ├── 03-transformer-decoder.md       # Decoder-only Transformer
│   ├── 04-finetuning-lora-dpo.md       # SFT、LoRA、QLoRA 与 DPO
│   ├── 05-model-architecture-comparison.md  # GPT/Llama/Qwen/DeepSeek 结构对比
│   ├── 06-experiment-report-template.md     # 实验报告模板
│   ├── 07-curriculum-audit.md               # 教材审计与优化建议
│   └── 08-milvus-vector-retrieval-for-rag.md # Milvus 向量检索与 RAG 面试教程
│
├── projects/                     # 9 个渐进式代码项目
│   ├── 01-ml-basics/             # PyTorch 二分类 MLP
│   ├── 02-transformer-from-scratch/  # Attention 与迷你 GPT 从零实现
│   ├── 03-tokenizer-and-data/    # BPE Tokenizer 训练 + 数据管线
│   ├── 04-sft-qwen-lora/        # SFT 数据集构建与 LoRA 训练准备
│   ├── 05-dpo-preference/       # DPO 偏好数据集构建
│   ├── 06-open-source-reading/  # 开源项目精读与复现
│   ├── 07-milvus-rag/           # Milvus Lite 检索链路实践
│   ├── 08-industrial-rag/       # 工业级 RAG 向量检索设计方案
│   └── 09-production-rag/       # 🚀 生产级多模态 RAG 知识库（完整可上线）
│
└── .github/                      # CI/CD 配置
```

## 快速开始

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行各阶段练习
python projects/01-ml-basics/train_mlp.py --epochs 3
python projects/03-tokenizer-and-data/train_bpe_tokenizer.py
python projects/03-tokenizer-and-data/data_pipeline.py
python projects/02-transformer-from-scratch/smoke_test.py
python projects/02-transformer-from-scratch/train_tiny_gpt.py --steps 20
python projects/04-sft-qwen-lora/build_sft_dataset.py
python projects/05-dpo-preference/build_preference_dataset.py
```

## 学习路线

按下面顺序逐步进阶，每一阶段先读笔记→写代码→跑实验→做面试口述总结。

| 阶段 | 笔记 | 项目 | 核心技能 |
|------|------|------|---------|
| 1. ML 基础 | `01-pytorch-training-loop` | `01-ml-basics` | 训练循环、梯度、损失函数、过拟合 |
| 2. Tokenizer | `02-tokenizer-and-causal-lm` | `03-tokenizer-and-data` | BPE、tokenize、causal LM labels |
| 3. Transformer | `03-transformer-decoder` | `02-transformer-from-scratch` | Attention、GPT block、KV cache |
| 4. 微调 | `04-finetuning-lora-dpo` | `04-sft-qwen-lora` | SFT、LoRA、QLoRA |
| 5. 对齐 | `04-finetuning-lora-dpo` | `05-dpo-preference` | DPO、偏好数据、chosen/rejected |
| 6. 模型对比 | `05-model-architecture-comparison` | `06-open-source-reading` | GPT/Llama/Qwen/DeepSeek 结构差异 |
| 7. RAG 入门 | `08-milvus-vector-retrieval-for-rag` | `07-milvus-rag` | Milvus Lite、向量检索 |
| 8. RAG 进阶 | — | `08-industrial-rag` | 工业级 RAG 设计方案 |
| 9. RAG 实战 | — | `09-production-rag` | **完整可上线系统** |

详细 12 周计划、每日训练模板、面试必会问题与参考答案见 **[target.md](target.md)**。

## 09 Production RAG（综合实战项目）

这是本教材的**旗舰项目**——一个基于 Milvus 的企业级多模态 RAG 知识库系统，可直接上线使用。

**核心能力：**

| 功能 | 说明 |
|------|------|
| 📄 多格式摄入 | PDF（PyMuPDF + 图片 OCR）、Markdown、HTML、TXT、CSV |
| 🔍 混合检索 | Dense（语义向量）+ Sparse（BM25）→ RRF 融合 → BGE-Reranker 精排 |
| 🖼️ 多模态问答 | PDF 图片提取 → Vision LLM 描述 → 图文联合检索 |
| 🤖 LLM 智能增强 | 查询改写、答案生成、思维导图、数据表格、文档摘要 |
| 🔐 多租户 ACL | PBKDF2 密码认证、Tenant 隔离、用户角色管理 |
| 🐳 Docker 一键部署 | ETCD + MinIO + Milvus + FastAPI + Nginx/React，2 核 2G 即可运行 |

**技术栈：** React 19 + FastAPI + Milvus 2.6 + SiliconFlow API + Docker Compose

详见 [projects/09-production-rag/README.md](projects/09-production-rag/README.md)。

## 验收标准

每个练习项目需满足：

- 能运行脚本
- 能解释输入、输出和张量 shape
- 能修改 1-2 个关键参数并观察变化
- 能写一段面试口述总结

## 学习原则

- 每个概念对应一段代码
- 每个训练脚本保存实验记录
- 不追求一开始训练大模型，先把小模型闭环跑通
- 不只看教程，必须读源码和改源码
- 面试表达围绕「问题→方法→实验→结果→反思」组织
