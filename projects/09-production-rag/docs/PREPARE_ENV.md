# 09 Production RAG 开发前环境准备

本文档记录正式编写 09 前端和部署能力前，需要提前准备的环境、依赖和工具。

## 1. 当前环境检查结果

在当前机器上已经确认：

| 项目 | 当前状态 | 结论 |
| --- | --- | --- |
| Python | `Python 3.14.4` | 可用 |
| 虚拟环境 | `.venv` 可激活 | 可用 |
| uv | `uv 0.11.17` | 可用 |
| Python 依赖 | `uv pip check` 通过 | 后端依赖健康 |
| Node.js | `node v24.16.0` | 可用 |
| npm | `npm 11.13.0` | 可用 |
| Docker | CLI 和 daemon 可访问 | 可用 |
| Docker Compose | 可显示版本 | 可用 |

后端 Python、前端 Node 工具链和 Docker 当前都可用。本文档仍保留安装建议，方便读者在自己的机器上复现环境准备过程。

## 2. 需要的工具

### 2.1 必需

- Python 3.14
- uv
- Node.js 20+，推荐 22+
- npm
- Docker
- Docker Compose v2
- git
- curl
- ca-certificates

### 2.2 建议安装

- jq：调试 JSON API。
- unzip：处理下载包或前端构建产物。
- build-essential / pkg-config：部分 Python 或 Node 原生依赖编译时需要。

## 3. 环境诊断脚本

本项目只提供普通用户检查脚本，不提供自动安装脚本：

```bash
bash projects/09-production-rag/scripts/prepare_user_env.sh
```

它会检查：

- Python 版本。
- uv 版本。
- Python 依赖兼容性。
- Node/npm 是否可用。
- Docker/Compose 是否可用。
- 09 后端关键入口的 Python 语法。

脚本只输出诊断结果和建议，不会安装包，不会修改系统配置，也不会调用 sudo。

注意：`uv pip check` 默认可能尝试写 `~/.cache/uv`。当前脚本会使用：

```text
/tmp/practice4llm-uv-cache
```

避免在受限环境下写用户 home cache。

## 4. Node/npm 准备建议

如果环境里 `node` 不存在，或 `npm` 指向 Windows 路径但不可用，建议选择一种方式处理：

### 4.1 WSL 内安装 Node/npm

适合希望前端开发完全运行在 Linux / WSL 内的情况。

建议安装 Node.js 20+ 或 22+。可以使用系统包管理器、NodeSource、nvm、fnm 等任一方式，但要保证：

```bash
node --version
npm --version
```

在 WSL shell 中直接可用，且不要依赖 `/mnt/c/Program Files/nodejs/npm`。

### 4.2 修复 PATH，使用 Windows Node

不推荐作为主方案，因为 WSL 与 Windows Node/npm 在路径、文件监听和 node_modules 上容易出问题。如果坚持使用，应确认：

```bash
node --version
npm --version
```

在 WSL shell 中都能正常执行。

## 5. Docker 准备建议

如果你更想使用 Windows Docker Desktop，而不是 WSL 内 Docker：

1. 打开 Docker Desktop。
2. 进入 Settings。
3. 打开 Resources。
4. 打开 WSL Integration。
5. 启用当前 Ubuntu 发行版。
6. 重新打开 WSL shell。
7. 验证：

```bash
docker --version
docker compose version
docker ps
```

如果你希望在 WSL 内直接运行 Docker，也可以自行安装 Docker Engine，并确保当前用户能访问 Docker daemon：

```bash
docker --version
docker compose version
docker ps
```

不建议同时混用 Docker Desktop WSL 集成和 WSL 内 Docker Engine，避免 socket、权限和上下文混乱。

## 6. 前端项目初始化前置条件

当前前端已经初始化完成。重新安装依赖时执行：

```bash
cd projects/09-production-rag/frontend
npm install
npm run build
```

## 7. 后端开发前置条件

从仓库根目录：

```bash
source .venv/bin/activate
uv pip check
python -m py_compile projects/09-production-rag/serve.py
```

如果本地要跑 Milvus Lite 或后端 API：

```bash
cd projects/09-production-rag
python schema.py --reset
uvicorn serve:app --host 127.0.0.1 --port 8008
```

## 8. 部署联调前置条件

部署联调前应满足：

```bash
docker --version
docker compose version
docker ps
```

然后在 `projects/09-production-rag` 下运行：

```bash
docker compose up -d milvus rag-api rag-web
```

## 9. 当前建议

下一步先完成工具链：

1. 运行 `scripts/prepare_user_env.sh`，查看缺少哪些工具。
2. 重新打开 shell。
3. 自行选择 Node/npm 安装方式。
4. 自行选择 Docker Desktop WSL 集成或 WSL 内 Docker Engine。
5. 再次运行 `scripts/prepare_user_env.sh`。
6. 确认 Node/npm/Docker 可用后，再开始初始化 `frontend/`。
