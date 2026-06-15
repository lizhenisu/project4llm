# 09 Production RAG

> 基于 Milvus 的企业级多模态 RAG 知识库系统 —— 支持文本+图片混合检索、思维导图生成、多租户 ACL 和生产级 Docker 部署。

## 功能概览

- **📄 多格式文档摄入**：PDF（PyMuPDF 解析 + 嵌入图片 OCR）、Markdown、HTML、TXT、CSV/TSV
- **🔍 混合检索**：Dense（语义向量）+ Sparse（BM25 关键词）→ RRF 融合 + BGE-Reranker 精排
- **🖼️ 多模态问答**：PDF 图片提取 → Vision LLM 描述 → 图文联合检索
- **🤖 LLM 智能增强**：查询改写、答案生成、思维导图、数据表格、文档摘要
- **🔐 多租户 ACL**：PBKDF2 密码认证、Tenant 隔离、用户角色管理
- **📊 Studio**：基于已选来源一键生成思维导图和数据表格
- **🧪 完整评估框架**：Recall@K、MRR@K、nDCG@K、答案忠实度、发布门禁
- **🐳 Docker 一键部署**：ETCD + MinIO + Milvus + FastAPI + Nginx/React

## 快速开始

### 开发环境（热重载）

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env，填入 LLM API Key 等配置

# 2. 启动基础设施（Milvus 向量数据库等）
docker compose up -d milvus

# 3. 初始化 Schema
python schema.py --reset

# 4. 启动后端（--reload 热重载）
source ../../.venv/bin/activate
MILVUS_URI="http://127.0.0.1:19530" \
RAG_OBJECT_STORE_DIR="$(pwd)/object_store" \
RAG_RUNTIME_DIR="$(pwd)/runtime" \
uvicorn serve:app --reload --host 0.0.0.0 --port 8008

# 5. 启动前端（Vite 热重载，默认 :5173）
cd frontend && npm install && npm run dev -- --host 0.0.0.0
```

前端通过 Vite Proxy 将 `/api/*` 代理到 `http://127.0.0.1:8008`。访问 `http://localhost:5173` 即可使用。

### 生产环境（Docker Compose）

```bash
# 1. 完整部署
docker compose up -d

# 2. 一键配置 HTTPS（自动获取 Let's Encrypt 证书）
sudo bash scripts/setup_caddy.sh your-domain.com

# 访问 https://your-domain.com
# 前端: http://localhost:8080
# API:  http://localhost:8008
# Milvus: localhost:19530

# 可选：批量摄入
RAG_TEXT_INPUT="/data/docs" \
RAG_IMAGE_INPUT="/data/images" \
docker compose --profile ingest up rag-ingest
```

## 项目结构

```
09-production-rag/
├── serve.py                  # FastAPI 应用入口
├── schema.py                 # Milvus Collection 初始化
├── answer.py                 # 文本 RAG 问答入口
├── answer_multimodal.py      # 多模态 RAG 问答入口
├── search_*.py               # 各类检索脚本（dense/sparse/hybrid/multimodal）
├── rerank.py                 # 重排序独立脚本
├── ingest_*.py               # 文档摄入脚本（files/pdf/markdown/text/tables/images）
├── eval_retrieval.py         # 检索评估
├── eval_answer.py            # 答案评估
├── release_gate.py           # 发布门禁
├── benchmark_latency.py      # 延时基准测试
├── Makefile                  # 常用命令快捷入口
├── Dockerfile                # API 镜像构建
├── docker-compose.yml        # 完整部署编排
├── .env.example              # 环境变量参考
│
├── rag_core/                 # RAG 核心模块
│   ├── config.py             # 配置管理（38 项环境变量）
│   ├── pipeline.py           # 检索管道编排
│   ├── rewrite.py            # LLM 查询改写
│   ├── embeddings.py         # 嵌入模型（SiliconFlow/BGE/CLIP）
│   ├── milvus_store.py       # Milvus Schema、索引、混合检索
│   ├── rerankers.py          # 重排序（SiliconFlow/BGE Cross-Encoder）
│   ├── context.py            # 上下文打包（三重约束）
│   ├── answering.py          # LLM 答案生成
│   ├── prompts.py            # System/User Prompt 模板
│   ├── io.py                 # PDF/HTML/MD 解析 + 图片提取
│   ├── text_utils.py         # 结构化分块（代码块/表格保持完整）
│   ├── sources.py            # 来源管理（上传、解读、删除、版本）
│   ├── artifacts.py          # 思维导图/数据表格生成
│   ├── conversations.py      # 对话 CRUD
│   ├── object_store.py       # JSONL 文档归档
│   ├── versioning.py         # 版本发布与解析
│   ├── auth.py               # ACL 鉴权上下文
│   ├── user_auth.py          # 用户注册/登录/Session
│   ├── database.py           # SQLite 元数据库（WAL 模式）
│   ├── pii.py                # PII 检测与脱敏
│   ├── guards.py             # 跨租户查询防护
│   ├── events.py             # 事件日志
│   ├── source_guides.py      # LLM 文档摘要
│   ├── citations.py          # 引用评估工具
│   ├── types.py              # 核心数据模型
│   └── readiness.py          # 健康检查报告
│
├── frontend/                 # React + TypeScript 前端
│   └── src/
│       ├── App.tsx           # 路由（/、/login、/register、/architecture）
│       ├── app/
│       │   ├── WorkspacePage.tsx  # 工作台主页
│       │   ├── AuthPage.tsx       # 登录/注册
│       │   ├── ArchitecturePage.tsx # 系统架构文档
│       │   └── SettingsDialog.tsx
│       ├── components/
│       │   ├── chat/ChatPanel.tsx
│       │   ├── sources/SourcePanel.tsx
│       │   ├── studio/StudioPanel.tsx
│       │   └── ui/
│       └── lib/
│           ├── api.ts        # API 客户端
│           ├── AuthContext.tsx
│           ├── storage.ts    # 本地持久化
│           └── types.ts
│
├── docs/                     # 项目文档
│   ├── ARCHITECTURE.md       # 系统架构详解（18 章）
│   ├── RELEASE_CHECKLIST.md  # 发布检查清单
│   ├── EXECUTION_PLAN.md     # 开发执行计划
│   ├── PREPARE_ENV.md        # 环境准备指南
│   ├── frontend-design/      # 前端 UI 设计参考
│   └── archive/              # 历史版本文档
│
├── tests/                    # 测试用例
├── scripts/                  # 部署辅助脚本
├── object_store/             # 文档归档与版本数据
├── runtime/                  # 运行时数据（DB、对话、日志）
└── volumes/                  # Docker 持久化卷
```

## LLM 配置

本项目默认使用 SiliconFlow API 网关，所有 Embedding、Rerank、Chat 调用均通过其 OpenAI 兼容端点：

```bash
# .env 关键配置
RAG_LLM_BASE_URL=https://api.siliconflow.cn
RAG_LLM_API_KEY=sk-your-key
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash

# 嵌入模型
RAG_EMBEDDING_BACKEND=siliconflow
EMBEDDING_MODEL=BAAI/bge-m3

# 重排序模型
RAG_RERANK_BACKEND=siliconflow
RERANK_MODEL=BAAI/bge-reranker-v2-m3

# 图片嵌入（默认启用）
RAG_IMAGE_EMBEDDING_BACKEND=siliconflow    # 改为 none 关闭多模态
IMAGE_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
```

也可切换为本地模型（需 GPU）：
- `RAG_EMBEDDING_BACKEND=bge` → 本地加载 BGE-M3
- `RAG_RERANK_BACKEND=bge` → 本地加载 BGE-Reranker-v2-m3

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/query` | RAG 查询（核心） |
| POST | `/search` | 仅检索 |
| GET/POST | `/sources` | 来源文档管理 |
| POST | `/sources/upload` | 文件上传 |
| GET | `/sources/content/{doc_id}` | 文档内容展示 |
| GET/POST | `/conversations` | 对话管理 |
| GET/POST | `/artifacts` | Studio 产物 |
| POST | `/artifacts/mindmap` | 生成思维导图 |
| POST | `/feedback` | 答案反馈 |
| GET | `/announcements` | 公告 |
| POST | `/auth/register` | 注册 |
| POST | `/auth/login` | 登录 |
| GET | `/admin/*` | 管理员接口 |

## 文档

| 文档 | 说明 |
|------|------|
| [系统架构](docs/ARCHITECTURE.md) | 完整 RAG 系统架构详解（LLM、检索管道、多模态、Milvus Schema 等） |
| [发布检查清单](docs/RELEASE_CHECKLIST.md) | 生产发布验收步骤 |
| [执行计划](docs/EXECUTION_PLAN.md) | 开发执行与重构计划 |
| [环境准备](docs/PREPARE_ENV.md) | 开发环境搭建指南 |
| [前端设计](docs/frontend-design/readme.md) | UI 设计规格与组件规划 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 19 + TypeScript + Vite |
| 后端 | FastAPI + Uvicorn (Python 3.14) |
| 向量数据库 | Milvus 2.6（HNSW + BM25 + 图片索引） |
| 元数据库 | SQLite（WAL 模式） |
| 对象存储 | 本地文件系统（JSONL 归档 + 版本管理） |
| LLM 网关 | SiliconFlow API（OpenAI 兼容） |
| 容器化 | Docker Compose（ETCD + MinIO + Milvus + API + Web） |

## 许可证

MIT
