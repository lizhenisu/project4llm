# 09 Production RAG 环境准备与部署指南

本文档记录 `09-production-rag` 的开发环境搭建和轻量服务器（2 核 2G）部署要点。

## 1. 开发环境要求

| 项目 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.14+ | |
| uv | 0.11+ | Python 包管理器 |
| Node.js | 20+ | 前端开发和构建 |
| npm | 11+ | |
| Docker | 24+ | 运行 Milvus 等基础设施容器 |
| Docker Compose | v2 | |
| git | 任意 | |

### 1.1 检查脚本

```bash
bash projects/09-production-rag/scripts/prepare_user_env.sh
```

该脚本仅输出诊断结果，不会安装任何包或调用 sudo。

## 2. 本地开发环境搭建

### 2.1 Python 后端

```bash
# 从仓库根目录
source .venv/bin/activate
uv pip check
cd projects/09-production-rag

# 初始化 Milvus Schema
python schema.py --reset

# 启动后端（热重载）
uvicorn serve:app --reload --host 127.0.0.1 --port 8008
```

### 2.2 前端

```bash
cd projects/09-production-rag/frontend
npm install
npm run dev -- --host 0.0.0.0
```

前端通过 Vite Proxy 将 `/api/*` 代理到 `http://127.0.0.1:8008`。访问 `http://localhost:5173`。

### 2.3 基础设施容器

```bash
cd projects/09-production-rag
docker compose up -d milvus
```

仅在本地需要 Milvus 向量数据库时启动。Embedding、Rerank、LLM 调用走 SiliconFlow API，无需本地 GPU。

## 3. Playwright E2E 测试

```bash
cd projects/09-production-rag/frontend

# 安装浏览器
npx playwright install chromium

# Ubuntu 26.04 需要覆盖平台标识
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 npx playwright install chromium
sudo env "PATH=$PATH" PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 npx playwright install-deps chromium

# 运行测试
npm run test:e2e           # 标准环境
npm run test:e2e:ubuntu26  # Ubuntu 26.04
```

如果 Node/npm 通过 nvm 安装，`sudo` 时必须用 `sudo env "PATH=$PATH" ...` 写法，否则 root 的 PATH 中找不到 npx。

## 4. 轻量服务器部署（2 核 2G）

### 4.1 资源评估

在 2 核 2G 云服务器上部署时，资源占用如下：

| 容器 | 内存占用（大约） | 说明 |
|------|-----------------|------|
| `rag-milvus` | ~600 MB | Milvus Standalone 模式 |
| `rag-etcd` | ~100 MB | Milvus 元数据 |
| `rag-minio` | ~150 MB | Milvus 对象存储 |
| `rag-api` | ~200 MB | FastAPI + Uvicorn（纯 API 调用，无本地模型） |
| `rag-web` | ~30 MB | Nginx 静态文件服务 |
| **合计** | **~1.1 GB** | 剩余 ~900 MB 给系统和突发负载 |

**完全可以部署在 2 核 2G 服务器上。**

### 4.2 Docker 镜像不包含 torch/transformers

这是关键点。Docker 镜像使用 `requirements-api.txt`，仅包含 8 个轻量依赖：

```
fastapi  openai  pydantic  pymilvus  pymupdf  pypdf  python-multipart  uvicorn
```

**不会下载 torch（~2.5 GB）、transformers（~2 GB）、sentence-transformers 等大型 ML 框架**。所有 Embedding、Rerank、LLM 调用均通过 SiliconFlow API 网关完成，模型推理发生在 SiliconFlow 云端，不在你的服务器上。

`pyproject.toml` 中的 torch、transformers 等依赖仅用于本地开发环境的 `bge`/`clip` 本地模型后端。Docker Compose 部署时，环境变量默认使用 `siliconflow` 后端，不会触发这些库的 lazy import。

### 4.3 部署步骤

```bash
# 1. 确保 Docker 和 Docker Compose 可用
docker --version
docker compose version

# 2. 进入项目目录
cd projects/09-production-rag

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入:
#   SILICONFLOW_API_KEY=sk-...
#   RAG_LLM_API_KEY=sk-...
#   LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash

# 4. 启动所有服务
docker compose up -d

# 5. （可选）一键配置 HTTPS 域名
sudo bash scripts/setup_caddy.sh your-domain.com

# 6. 检查服务状态
docker compose ps
curl http://localhost:8080/api/health

# 7. 访问前端
# http://<服务器IP>:8080
# 或 https://your-domain.com（如果已配置 Caddy）
```

### 4.4 轻量服务器优化建议

1. **图片嵌入走 API，无需本地资源**：默认 `RAG_IMAGE_EMBEDDING_BACKEND=siliconflow`，通过 `Qwen/Qwen3-VL-Embedding-8B` 模型 API 生成图片向量，不下载本地模型，不占额外内存和磁盘。如需关闭可设为 `none`
2. **控制 Milvus 内存**：Docker Compose 中 Milvus 使用 Standalone 模式，如需限制内存可在 `docker-compose.yml` 中为 `milvus` 服务添加 `mem_limit: 800m`
3. **关闭不必要的 profile**：不要使用 `--profile ingest` 在轻量服务器上跑批量摄入，摄入任务在另一台机器上执行后同步 `object_store/` 即可
4. **磁盘空间**：`volumes/`（Milvus 数据）和 `object_store/`（文档归档）会随使用增长，建议挂载数据盘或定期清理旧版本

## 5. 生产部署方式对比

| 方式 | 适用场景 | 说明 |
|------|---------|------|
| `docker compose up -d` | 单机部署 | 全部服务在一台机器上 |
| GHCR 镜像 | CI/CD 流水线 | 避免在生产服务器上构建镜像 |
| `docker compose -f docker-compose.yml -f docker-compose.ghcr.yml pull && docker compose -f docker-compose.yml -f docker-compose.ghcr.yml up -d --no-build` | 从 GitHub Container Registry 拉取预构建镜像 | 适合不想在生产服务器上安装 Node/npm 或在服务器本地构建后端辅助镜像的场景 |

## 6. 环境诊断常见问题

### Docker Daemon 权限

如果 `docker ps` 报权限错误：

```bash
sudo usermod -aG docker $USER
# 重新登录后生效
```

### WSL 内 Docker

如果使用 Windows Docker Desktop + WSL 集成，确保在 Docker Desktop → Settings → Resources → WSL Integration 中启用了当前发行版。

不建议同时混用 Docker Desktop WSL 集成和 WSL 内独立 Docker Engine。

### uv pip check 报缓存错误

脚本会使用 `/tmp/practice4llm-uv-cache` 避免写用户 home cache，这在受限环境下是正常的。
