# RAG 项目用户认证、管理员系统与高并发持久化重构 执行计划

> **⚠️ AI 辅助开发执行约束 (全局多轮迭代与防死循环原则)**
> 为了在“避免陷入局部死循环”与“保证极高生产级代码质量”之间取得完美平衡，执行本计划必须严格遵循**“全局多轮迭代 (Multi-round Iteration)”**策略：
> 1. **小步稳跑，按“阶段”交付**：首轮开发（Round 1）以构建全栈贯通的 MVP（基本可行性产品）为目标。每个阶段分2到4个批次提交，严禁在某一阶段死磕底层细节。
> 2. **先跑通，再跑好 (Make it work, then make it right)**：每轮推进时，只要核心逻辑无报错跑通即可。对于深度的异常捕获、极致的性能抽象、或 UI 的“锦上添花”，绝不允许其阻碍当前轮次的全局推进。
> 3. **阶段性 Checkpoint**：每个阶段完成后，必须保证系统全栈可用，并执行 Git Commit 作为回滚基线。
> 4. **全局多轮循环升华 (Iterative Refinement)**：当跑完整个计划的全部阶段（完成当前轮次）后，**必须反过头来，从阶段一重新开始执行第二轮（Round 2）、第三轮（Round 3）……**。在每个轮次中，由 AI 自主审视粗糙代码，集中精力重构优化、打磨 UI 细节、强化高并发健壮性和安全性。如此循环多轮，**直到 AI 评估当前代码架构、容错率和体验已经达到生产级完整度，方可宣告整个史诗级更新结束**。

## 1. 项目当前情况俯瞰与高并发架构诊断
- **当前架构**：FastAPI (后端) + React (前端) + Milvus (向量) + 本地 JSON 文件 (元数据)。
- **120人并发场景下的致命隐患**：
  1. **长耗时请求阻塞**：上传大文件和生成思维导图强依赖 LLM 的同步返回，大量 HTTP 连接保持打开状态，瞬间耗尽 FastAPI 工作线程，导致全局卡死。
  2. **无用户体系**：无法实现个人数据的鉴权隔离。
  3. **持久化性能隐患**：JSON 文件缺乏高并发写入锁和高效的索引遍历。
- **重构目标**：
  1. 引入完整的用户、角色、鉴权机制。
  2. **引入 SQLite 彻底重构元数据存储**。
  3. **利用 BackgroundTasks 解耦长耗时操作，彻底解决 Web 容器的并发阻塞问题**。

## 2. 核心技术选型与企业级优化
- **元数据数据库**：`sqlite3`。**【关键优化】**：强制开启 `PRAGMA journal_mode=WAL;` (预写日志)，并设置超时重试机制，确保 120 人规模下不会出现写死锁。
- **密码加密**：使用 `hashlib.pbkdf2_hmac` 进行加盐哈希。
- **异步任务队列**：使用 FastAPI 原生的 `BackgroundTasks`，将“文档切分入库”和“LLM 生成思维导图”放入后台执行，前台接口仅做 DB 状态变更并极速返回。

## 3. 分步执行计划

### 阶段一：高并发 SQLite 数据库搭建与长任务异步化 (DB & Async Foundation)
1. 创建 `rag_core/database.py` 模块，初始化启用 WAL 模式的 SQLite 数据库 (`metadata.db`)：
   - `users`: `id`, `username`, `password_hash`, `salt`, `role`, `created_at`
   - `sessions`: `token`, `user_id`, `expires_at`
   - `announcements`: `id`, `title`, `content`, `author_id`, `created_at`
   - `conversations`: `id`, `tenant_id`, `title`, `source_doc_ids`, `created_at`, `updated_at`
   - `messages`: `id`, `conversation_id`, `role`, `content`, `status`, `citations`, `created_at`
   - `artifacts`: `id`, `tenant_id`, `title`, `status` (generating/ready/failed), `source_doc_ids`, `root`, `error`, `created_at`, `updated_at`
2. **异步化改造**：
   - 修改 `POST /artifacts/mindmap`：接收请求后，插入一条 `status='generating'` 的 Artifact 记录到 SQLite，**立即返回给前端**。同时向 `BackgroundTasks` 投递 `build_llm_mindmap` 任务，该任务完成后再更新 DB 状态。
   - 前端改为轮询 (Polling) 该 Artifact 的最新状态，直到变为 `ready`。

### 阶段二：后端认证服务与 API 接口 (Auth Services & Endpoints)
1. 创建 `rag_core/user_auth.py`：
   - 注册逻辑：首个注册用户自动成为 `admin`，其余为 `user`。
   - 登录校验与 Token 派发。
   - 依赖注入函数 `get_current_user`。
2. 注册并实现 FastAPI 路由 (`serve.py`)：
   - `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
   - 管理员接口：`GET /admin/users`, `POST /admin/announcements`
   - 公共接口：`GET /announcements`

### 阶段三：前端状态管理与安全拦截 (Frontend Context & API)
1. 扩展 `frontend/src/lib/api.ts`，封装新增接口。
2. 改造请求拦截器：在 Headers 中自动携带 `Authorization: Bearer <token>`。如果接口返回 401，触发全局登出。
3. 创建 `frontend/src/lib/AuthContext.tsx`，维护全局用户状态。

### 阶段四：前端 UI 面板实现 (Frontend UI Components)
1. **Avatar 下拉与鉴权态**：
   - 改造右上角 `class="avatar"` 的交互，点击弹出面板。
   - 提供注册、登录、登出、个人信息入口。
2. **管理员控制台 (Admin View)**：
   - 实现用户列表表格。
   - 提供发布系统公告的富文本/多行输入表单。
3. **公告跑马灯/展示**：在 Workspace 顶部展示最新公告。

### 阶段五：压测、联调与清理 (Stress Testing & Cleanup)
1. 确保 Docker 挂载 `volumes/db` 目录持久化数据库。
2. 模拟多用户并发调用“生成脑图”接口，监控 FastAPI 的响应时间和后台任务的队列执行情况。
3. 测试 Admin 权限越权访问。
4. 清理旧代码中无用的 JSON 持久化逻辑与目录常量。
