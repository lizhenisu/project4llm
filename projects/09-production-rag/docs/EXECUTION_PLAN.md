# 09 Production RAG 执行计划

本文档定义 `projects/09-production-rag` 从当前后端项目演进为“可部署、带 TypeScript 前端的完整 RAG 系统”的执行方案。

设计参考见：

- `frontend-design/readme.md`
- `frontend-design/1.png` ~ `frontend-design/8.png`
- `PREPARE_ENV.md`
- `../README.md`
- `../ARCHITECTURE.md`

## 1. 最终目标

09 的目标不是继续做教学脚本，而是形成一个可上线的 Web RAG 系统：

1. 用户打开网页后直接进入 RAG 工作台。
2. 用户可以上传资料，系统解析、切分、入库。
3. 用户可以选择资料来源并进行带引用的 RAG 问答。
4. 用户可以对回答点赞/点踩，反馈写入后端事件。
5. 用户可以基于已选来源生成思维导图。
6. 系统可以通过 Docker Compose 部署到服务器。
7. 前端、后端、向量库、对象存储、运行日志有清晰边界。

## 2. 架构原则

09 应该按“前后端分离、模块化、易扩展”推进。这不是口号，而是后续目录、接口、部署和提交拆分的约束。

### 2.1 前后端分离

前端只负责：

- 页面布局和交互状态。
- 来源选择、上传队列、对话消息、Studio Artifact 的前端状态管理。
- 调用后端 API。
- 展示后端返回的 answer、citations、source、artifact。

前端不负责：

- 文档解析。
- chunk 切分。
- embedding。
- Milvus schema 和检索逻辑。
- rerank。
- prompt 构造。
- LLM 调用。
- 权限过滤。

后端只负责：

- 对外提供稳定 API。
- 文档入库和版本管理。
- RAG 检索、rerank、context packing、回答生成。
- 引用、反馈、事件、评估和发布门禁。
- 权限上下文校验。

后端不负责：

- 浏览器 UI 状态。
- 前端路由。
- 组件级交互。
- 静态资源构建。

部署也保持分离：

```text
rag-web  独立前端容器，Nginx 托管静态文件并代理 /api
rag-api  独立后端容器，FastAPI 提供 API
```

这样以后可以单独升级前端，不重建后端模型环境；也可以单独扩容 API，不影响静态资源服务。

### 2.2 模块化

模块边界按业务能力划分，而不是按“文件随手放”划分。

前端模块：

```text
sources   来源上传、来源列表、来源选择、来源详情
chat      问答输入、消息列表、引用展示、反馈
studio    Studio 工具入口、Artifact 列表、思维导图
settings  API 地址、Token、Tenant、ACL 配置
shared    基础 UI、类型、请求层、工具函数
```

后端模块：

```text
api        FastAPI route 层，请求/响应模型
rag_core   RAG 领域逻辑，检索、rerank、context、answering
ingest     文件解析、chunk、入库、版本发布
artifacts  思维导图等 Studio 产物
ops        readiness、monitoring、eval、release gate
```

当前 09 后端还没有 `api/`、`ingest/`、`artifacts/` 目录。第一版可以先保持现状，避免大迁移；但新增能力时应按这些边界写，不继续把所有 route 都塞进 `serve.py`。

### 2.3 易扩展

扩展性重点体现在三处：

1. UI 扩展：Studio 现在只做“思维导图”，以后增加报告、闪卡、测验时，应新增 tool 配置和 artifact renderer，不改 Chat 主流程。
2. 来源扩展：现在先做文件上传，以后增加网页、云盘、数据库来源时，应扩展 source adapter，不改前端来源列表模型。
3. RAG 能力扩展：现在是 text / multimodal query，以后增加新检索策略或新模型时，应由后端配置和 trace 暴露，不要求前端理解 Milvus 细节。

### 2.4 接口优先

前后端通过明确 API contract 协作。前端不要直接依赖后端内部文件结构，后端也不要假设前端组件状态。

每个新增能力都先定义：

- Request
- Response
- Loading 状态
- Error 状态
- 权限行为
- 是否写 runtime event

### 2.5 可替换部署

默认部署是 Docker Compose，但结构应允许以后替换为：

- 前端静态文件部署到 CDN / 对象存储。
- 后端部署到单独 API 服务器。
- Milvus 换成外部托管服务。
- LLM 网关换成其他兼容 OpenAI API 的服务。

因此配置必须通过环境变量注入，不能把服务器地址、Token、Tenant 写死在前端代码里。

## 3. 技术选型

### 3.1 前端

推荐：

- 语言：TypeScript
- 框架：React
- 构建工具：Vite
- 样式：CSS Modules 或普通 CSS，先不引入重型 UI 框架
- 图标：lucide-react
- 状态管理：React state + context；等复杂度上来再考虑 Zustand
- 请求层：fetch 封装，不先引入 Axios
- Markdown 渲染：react-markdown
- 思维导图画布：优先用 React Flow；如果只读树图实现更简单，也可以先用 SVG + 自定义布局

不推荐第一版使用 Next.js。原因：

- 当前后端已经是 FastAPI，前端只需要静态部署和 API 调用。
- Vite 静态产物更容易用 Nginx 部署。
- 09 第一阶段重点是产品闭环，不需要 SSR。

### 3.2 后端

保留当前 Python / FastAPI 后端：

- `serve.py`：API 入口
- `rag_core/`：RAG 核心逻辑
- `ingest_*.py`：入库能力
- `eval_*.py` / `release_gate.py`：上线门禁和评估工具

后续新增的上传、来源列表、Artifact 接口也放在 FastAPI 中，不单独新建 Node 后端。

### 3.3 部署

推荐 Docker Compose 服务拆分：

```text
rag-web      Nginx 静态服务，托管前端 dist，并反代 /api 到 rag-api
rag-api      FastAPI RAG 服务
rag-ingest   可选 profile，用于批量入库任务
milvus       向量库
etcd         Milvus 依赖
minio        Milvus 依赖
```

生产访问路径：

```text
Browser
  -> http(s)://server/
    -> rag-web
      -> /assets/* 静态前端资源
      -> /api/* 反向代理到 rag-api:8008/*
```

这样前端永远调用同源 `/api`，避免浏览器 CORS 问题。

## 4. 推荐项目目录结构

目标结构：

```text
projects/09-production-rag/
├── frontend/
│   ├── package.json
│   ├── package-lock.json              # 如果使用 npm
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── public/
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── app/
│       │   ├── AppShell.tsx
│       │   ├── WorkspacePage.tsx
│       │   └── SettingsDialog.tsx
│       ├── components/
│       │   ├── sources/
│       │   │   ├── SourcePanel.tsx
│       │   │   ├── SourceUploadDialog.tsx
│       │   │   ├── SourceList.tsx
│       │   │   └── SourceDetailDrawer.tsx
│       │   ├── chat/
│       │   │   ├── ChatPanel.tsx
│       │   │   ├── ChatInput.tsx
│       │   │   ├── MessageList.tsx
│       │   │   ├── AssistantMessage.tsx
│       │   │   └── CitationPopover.tsx
│       │   ├── studio/
│       │   │   ├── StudioPanel.tsx
│       │   │   ├── StudioToolGrid.tsx
│       │   │   ├── ArtifactList.tsx
│       │   │   └── MindMapView.tsx
│       │   └── ui/
│       │       ├── IconButton.tsx
│       │       ├── Panel.tsx
│       │       ├── Tooltip.tsx
│       │       └── EmptyState.tsx
│       ├── lib/
│       │   ├── api.ts
│       │   ├── auth.ts
│       │   ├── storage.ts
│       │   └── types.ts
│       ├── mocks/
│       │   ├── sources.ts
│       │   ├── chat.ts
│       │   └── mindmap.ts
│       └── styles/
│           ├── globals.css
│           ├── layout.css
│           └── tokens.css
├── rag_core/
├── scripts/
├── tests/
├── Dockerfile                         # 后端镜像
├── docker-compose.yml
├── .env.example
├── serve.py
├── ingest_*.py
├── search_*.py
├── answer*.py
└── release_gate.py
```

说明：

- `frontend/` 放 TypeScript 前端，不放到仓库根目录，避免 09 的产品代码分散。
- `docs/frontend-design/` 保留为 09 的设计准备资料，不作为运行时代码目录；未来前端可以安全使用 `frontend/src/components/ui/` 放基础 UI 组件。
- 后端 Python 结构暂时不大改，避免一次性迁移破坏已有脚本。
- 等 09 稳定后，可以再把后端入口整理进 `backend/`，但不是第一阶段必须项。

## 5. 前端目录职责

### 5.1 `app/`

放页面级组合组件：

- `AppShell.tsx`：顶部栏、三栏布局、全局设置入口。
- `WorkspacePage.tsx`：工作台主页面，管理来源、消息、Studio 状态。
- `SettingsDialog.tsx`：API 地址、Token、Tenant、ACL 配置。

### 5.2 `components/sources/`

来源相关组件：

- `SourcePanel.tsx`：左栏整体。
- `SourceUploadDialog.tsx`：上传弹窗。
- `SourceList.tsx`：来源列表、全选、单选。
- `SourceDetailDrawer.tsx`：来源详情。

### 5.3 `components/chat/`

对话相关组件：

- `ChatPanel.tsx`：中间栏整体。
- `ChatInput.tsx`：底部输入框。
- `MessageList.tsx`：消息列表。
- `AssistantMessage.tsx`：AI 回答、引用、反馈按钮。
- `CitationPopover.tsx`：引用详情浮层。

### 5.4 `components/studio/`

Studio 相关组件：

- `StudioPanel.tsx`：右栏整体。
- `StudioToolGrid.tsx`：Studio 功能卡片，只有思维导图可用。
- `ArtifactList.tsx`：生成中和已生成 Artifact 列表。
- `MindMapView.tsx`：思维导图详情画布。

### 5.5 `lib/`

前端基础库：

- `api.ts`：封装 `/api/health`、`/api/query`、`/api/feedback` 等请求。
- `auth.ts`：读取和生成请求头。
- `storage.ts`：localStorage 配置持久化。
- `types.ts`：后端响应和前端状态类型。

### 5.6 `mocks/`

第一阶段静态 UI 使用 mock 数据，后续接 API 时仍可用于离线开发和 UI 回归。

## 6. 后端需要补齐的接口

当前后端已经有：

```text
GET  /health
GET  /ready
POST /search
POST /query
POST /feedback
```

前端完整闭环还缺：

```text
POST   /sources/upload
GET    /sources
GET    /sources/{doc_id}
DELETE /sources/{doc_id}
POST   /artifacts/mindmap
GET    /artifacts
GET    /artifacts/{artifact_id}
DELETE /artifacts/{artifact_id}
```

### 6.1 来源上传接口

`POST /sources/upload`

请求：

```text
multipart/form-data
file: UploadFile
tenant_id: string
acl_groups: string
```

响应：

```json
{
  "doc_id": "internship-guide",
  "title": "创维 AI 研究院实习介绍资料(1).pdf",
  "source_type": "pdf",
  "doc_version": 1,
  "status": "processing"
}
```

第一版可以同步解析并返回 `ready`；如果解析较慢，再引入后台任务。

### 6.2 来源列表接口

`GET /sources`

响应：

```json
{
  "sources": [
    {
      "doc_id": "internship-guide",
      "title": "创维 AI 研究院实习介绍资料(1).pdf",
      "source_type": "pdf",
      "source_uri": "uploads/internship-guide.pdf",
      "doc_version": 1,
      "chunk_count": 32,
      "status": "ready",
      "created_at": "2026-06-12T10:30:00Z"
    }
  ]
}
```

可以先复用 `list_documents.py` 和 object store registry 的能力实现。

### 6.3 思维导图接口

`POST /artifacts/mindmap`

请求：

```json
{
  "source_doc_ids": ["internship-guide"],
  "title": "实习招聘思维导图"
}
```

响应：

```json
{
  "id": "mindmap-001",
  "title": "实习招聘思维导图",
  "status": "ready",
  "source_doc_ids": ["internship-guide"],
  "root": {
    "id": "root",
    "label": "创维集团AI研究院实习介绍",
    "children": []
  }
}
```

MVP 可以先不新增接口，前端用 `/query` 生成 JSON；但最终上线建议落成正式 Artifact 接口。

## 7. 前端 API 客户端约定

前端统一通过 `/api` 访问后端：

```ts
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
```

生产：

```text
/api/query -> rag-web Nginx -> rag-api:8008/query
```

本地开发：

```text
Vite dev server /api proxy -> http://127.0.0.1:8008
```

认证头：

```ts
{
  Authorization: `Bearer ${token}`,
  "X-RAG-Tenant-ID": tenantId,
  "X-RAG-ACL-Groups": aclGroups.join(",")
}
```

设置默认值：

```text
tenant_id: team_a
acl_groups: engineering
candidate_limit: 20
context_limit: 5
query_mode: text
```

## 8. Docker 部署结构

### 8.1 后端 Dockerfile

保留当前：

```text
projects/09-production-rag/Dockerfile
```

职责：

- 安装 Python 依赖。
- 复制 09 后端代码。
- 启动 `scripts/start_api.sh`。

### 8.2 前端 Dockerfile

新增：

```text
projects/09-production-rag/frontend/Dockerfile
```

推荐多阶段构建：

```dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY projects/09-production-rag/frontend/package*.json ./
RUN npm ci
COPY projects/09-production-rag/frontend ./
RUN npm run build

FROM nginx:1.27-alpine
COPY projects/09-production-rag/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

### 8.3 Nginx 配置

新增：

```text
projects/09-production-rag/frontend/nginx.conf
```

核心规则：

```nginx
server {
  listen 80;
  server_name _;

  root /usr/share/nginx/html;
  index index.html;

  location /api/ {
    proxy_pass http://rag-api:8008/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  location / {
    try_files $uri $uri/ /index.html;
  }
}
```

### 8.4 Compose 服务

`docker-compose.yml` 目标新增：

```yaml
rag-web:
  build:
    context: ../..
    dockerfile: projects/09-production-rag/frontend/Dockerfile
  container_name: rag-web
  ports:
    - "8080:80"
  depends_on:
    - rag-api
```

上线后访问：

```text
http://server:8080
```

如果服务器已有 Nginx / Caddy，也可以把 `rag-web` 改为只暴露内网端口，由外层网关负责 HTTPS。

## 9. 本地开发流程

### 9.1 后端

从仓库根目录：

```bash
source .venv/bin/activate
cd projects/09-production-rag
python schema.py --reset
uvicorn serve:app --host 127.0.0.1 --port 8008
```

### 9.2 前端

```bash
cd projects/09-production-rag/frontend
npm install
npm run dev
```

Vite 配置代理：

```ts
server: {
  proxy: {
    "/api": {
      target: "http://127.0.0.1:8008",
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ""),
    },
  },
}
```

本地访问：

```text
http://127.0.0.1:5173
```

## 10. 执行阶段

### Phase 0：前端工程初始化

目标：创建 TypeScript 前端项目骨架。

任务：

1. 在 `projects/09-production-rag/frontend` 初始化 Vite React TS。
2. 安装 `lucide-react`、`react-markdown`。
3. 建立 `src/` 目录结构。
4. 添加 `tokens.css`、`globals.css`、`layout.css`。
5. 添加 Vite `/api` proxy。
6. 添加前端 Dockerfile 和 nginx.conf。
7. 更新 09 的 `docker-compose.yml`，加入 `rag-web`。

验收：

```bash
cd projects/09-production-rag/frontend
npm run build
```

### Phase 1：静态 UI 复现

目标：不接后端，先把界面做出来。

任务：

1. 实现顶部栏。
2. 实现三栏布局。
3. 实现来源空状态和来源列表 mock。
4. 实现上传弹窗静态交互。
5. 实现对话空状态、有来源摘要状态、问答结果 mock。
6. 实现 Studio 功能卡片、列表、思维导图详情。
7. 做桌面和移动端响应式。

验收：

- 页面视觉接近 `frontend-design/1.png`、`frontend-design/3.png`、`frontend-design/4.png`、`frontend-design/5.png`、`frontend-design/7.png`、`frontend-design/8.png`。
- 1440px 和 390px 宽度下无重叠、无横向溢出。

### Phase 2：接入现有问答 API

目标：让对话可以真实调用后端 RAG。

任务：

1. 实现 `lib/api.ts`。
2. 设置页支持 API Base、Token、Tenant、ACL。
3. 调用 `/api/health` 检查连接。
4. 对话输入调用 `/api/query`。
5. 渲染 answer 和 citations。
6. 点赞/点踩调用 `/api/feedback`。
7. 错误状态显示后端错误信息。

验收：

- 可以向已存在知识库提问。
- 回答显示引用。
- feedback 事件写入后端 runtime。

### Phase 3：来源管理闭环

目标：用户可以在网页上传文件并看到来源列表。

后端任务：

1. 增加 `POST /sources/upload`。
2. 增加 `GET /sources`。
3. 增加 `DELETE /sources/{doc_id}`。
4. 上传后调用现有 ingest pipeline。
5. 来源列表从 object store/current version registry 或 Milvus stats 汇总。

前端任务：

1. 上传弹窗接入真实 upload。
2. 来源列表接入 `/api/sources`。
3. 支持上传状态轮询或同步状态刷新。
4. 支持删除来源。
5. 选中来源影响当前 query 范围。

注意：

当前 `/query` 没有 `doc_ids` 字段，只能通过 tenant、ACL、source_types、doc_version 过滤。要做到“只问选中的来源”，后端需要给 QueryRequest 增加：

```py
doc_ids: list[str] = Field(default_factory=list)
```

并在 Milvus filter 中加入 doc_id 过滤。

验收：

- 上传 PDF/TXT/Markdown 后，来源列表出现新文件。
- 选中文件后提问，只引用该文件内容。
- 删除文件后，来源列表和检索结果不再出现该文档。

### Phase 4：思维导图

目标：Studio 中“思维导图”可用。

第一版实现：

1. 前端点击“思维导图”。
2. 调用 `/api/query`，要求返回 JSON mindmap。
3. 前端解析 JSON，生成本地 artifact。
4. 在 Studio 列表显示生成结果。
5. 点击进入 MindMapView。

第二版实现：

1. 后端新增 `/artifacts/mindmap`。
2. 后端保存 artifact JSON 到 runtime 或 object store。
3. 前端通过 `/artifacts` 拉取历史结果。

验收：

- 有来源时能生成思维导图。
- 生成中、成功、失败状态明确。
- 详情页支持缩放、适配视图、下载 JSON 或 PNG。

### Phase 5：部署联调

目标：服务器上用 Docker Compose 一键启动。

任务：

1. `docker compose build rag-api rag-web`
2. `docker compose up -d milvus rag-api rag-web`
3. 检查 `rag-web` 页面可访问。
4. 检查 `/api/health` 通过 Nginx 正常代理。
5. 检查前端可以真实问答。
6. 检查 runtime、object_store、volumes 都在宿主机挂载目录。

验收命令：

```bash
curl http://127.0.0.1:8080/api/health
docker compose ps
docker compose logs --tail=100 rag-api
docker compose logs --tail=100 rag-web
```

## 11. 推荐提交顺序

建议拆成小提交：

1. `Add frontend execution plan`
2. `Scaffold Vite React frontend`
3. `Build static RAG workspace UI`
4. `Wire chat UI to RAG query API`
5. `Add source upload and listing API`
6. `Connect source management UI`
7. `Add mind map studio artifact`
8. `Add production web container`

不要把前端脚手架、后端接口、UI 大改、部署改动塞进同一个提交。

## 12. 验收总清单

完成 09 第一版上线形态时，应满足：

- `projects/09-production-rag/frontend` 存在完整 TypeScript 前端工程。
- `npm run build` 通过。
- FastAPI 后端 `python -m py_compile` 通过。
- `docker compose up -d milvus rag-api rag-web` 可启动。
- 浏览器访问 `rag-web` 看到三栏工作台。
- 上传资料后来源列表出现文件。
- 选择来源后可以问答。
- 回答显示 citations。
- 点赞/点踩写入 `/feedback`。
- Studio 只有“思维导图”可用。
- 思维导图能生成、展示、下载。
- 运行产物不进入 git：`runtime/`、`object_store/`、`volumes/`、本地 DB。
- 移动端宽度下无明显布局错乱。

## 13. 当前下一步

建议下一步直接执行 Phase 0：

1. 先按 `PREPARE_ENV.md` 补齐 Node/npm/Docker 工具链。
2. 创建 `projects/09-production-rag/frontend`。
3. 初始化 Vite React TypeScript。
4. 加入 `lucide-react` 和 `react-markdown`。
5. 建立组件目录。
6. 先用 mock 数据做出三栏静态工作台。

只有当静态 UI 站稳之后，再开始补后端上传和 Artifact 接口。
