# 09 Production RAG 前端 UI 设计说明

本目录记录 `projects/09-production-rag` 的前端界面设计。参考图来自 Google NotebookLM 风格，但本项目不做完整复刻；目标是做一个可部署、可演示、可继续扩展的 RAG 工作台。

这部分资料也作为 09 教学中的“vibecoding 准备过程”留档：读者可以看到作者在真正写前端代码前，如何先整理参考图、确定 MVP 范围、拆分页面状态、明确 API 缺口，再和 AI 一起推进实现。

## 1. 产品定位

09 的前端不是教学展示页面，而是一个面向真实用户的知识库问答系统：

- 用户可以上传资料，等待系统解析并入库。
- 用户可以勾选一个或多个来源文件，围绕这些来源进行 RAG 问答。
- 回答必须显示引用来源，用户可以查看证据、复制答案、反馈答案质量。
- 右侧 Studio 只实现一个功能：基于已选来源生成“思维导图”。
- 其他 NotebookLM 式 Studio 能力只保留视觉入口，不实现业务逻辑。

前端首屏应该直接是工作台，不做营销落地页。

## 2. 参考图对应关系

| 图片 | 用途 | 需要实现的重点 |
| --- | --- | --- |
| `1.png` | 空工作台三栏布局 | 左来源、中对话、右 Studio，顶部全局栏 |
| `2.png` | 添加来源弹窗 | 文件上传、拖拽区域、上传方式入口 |
| `3.png` | 来源列表 | 文件条目、全选、选中状态、文件类型图标 |
| `4.png` | 有来源后的对话首页 | 文档摘要、建议问题、底部输入框 |
| `5.png` | 问答结果 | 用户问题、AI 回答、引用角标、反馈按钮 |
| `6.png` | Studio 空状态 | 功能卡片网格、空状态、添加笔记按钮 |
| `7.png` | Studio 列表状态 | 思维导图生成中、已生成条目列表 |
| `8.png` | 思维导图详情 | 画布、缩放、下载、全屏、反馈按钮 |

## 3. MVP 范围

### 必须实现

1. 三栏工作台布局。
2. 来源上传弹窗。
3. 来源列表、全选、单选、多选。
4. 对话区空状态、有来源状态、问答状态。
5. 调用后端 `/query` 完成 RAG 问答。
6. 回答引用展示。
7. 调用后端 `/feedback` 提交点赞、点踩。
8. Studio 中只实现“思维导图”。
9. 思维导图生成中、列表、详情页。
10. 基础错误、加载、空状态。

### 暂不实现

以下能力在接通后再进入产品界面，当前不展示未实现入口：

- Web 搜索来源。
- Fast Research。
- Google Drive / 云端硬盘导入。
- 网站 URL 解析。
- YouTube / 视频概览。
- 音频概览。
- 演示文稿。
- 报告。
- 闪卡。
- 测验。
- 信息图。
- 数据表格。
- 多工作区管理。
- 用户头像菜单和真实账号体系。

## 4. 页面结构

### 4.1 顶部全局栏

高度约 64px。左侧显示产品标识和当前笔记本标题：

```text
[Logo] 未命名的知识库
```

右侧保留操作区：

- `设置`：打开本地设置弹窗，可配置 API 地址、Token、Tenant、ACL。
- 用户头像：静态展示。

顶部栏不承载复杂导航，核心体验保持在三栏工作台内。

### 4.2 主工作台布局

桌面端使用三栏布局：

```text
┌──────────────┬────────────────────────┬──────────────┐
│ 来源          │ 对话                    │ Studio        │
│ 320px-360px   │ flex: 1                 │ 320px-360px   │
└──────────────┴────────────────────────┴──────────────┘
```

布局要求：

- 背景使用极浅冷灰色，三栏面板使用白色。
- 面板之间留 12px-16px 间距。
- 面板圆角 8px，不要做过大的卡片圆角。
- 三栏高度占满视口剩余空间。
- 左右栏支持折叠，折叠后只保留窄图标按钮。
- 中间对话区始终是主要区域。

移动端暂不做完整三栏同屏，采用标签页切换：

```text
[来源] [对话] [Studio]
```

默认进入 `对话` 标签，上传和 Studio 通过标签或抽屉打开。

## 5. 左侧：来源面板

### 5.1 空状态

面板标题为 `来源`。空状态包含：

- 顶部按钮：`+ 添加来源`
- 空状态图标
- 文案：`已保存的来源将显示在此处`
- 辅助文案：`点击上方的“添加来源”上传 PDF、Markdown、TXT、CSV、图片等文件。`

### 5.2 添加来源弹窗

触发方式：

- 点击左侧 `添加来源`
- Studio 空状态底部 `添加笔记` 可以复用此弹窗，但文案应改成添加来源或添加资料

弹窗结构：

```text
标题：添加来源
副标题：上传文件后，系统会解析、切分并写入知识库。

[拖拽上传区域]
  或拖放文件
  支持 PDF、Markdown、TXT、HTML、CSV/TSV、图片元数据 JSONL

[上传文件]
```

MVP 行为：

- `上传文件` 可用。
- 未接通的粘贴、网站、云盘导入不进入产品界面。
- 拖拽文件进入区域时高亮边框。
- 上传后显示文件队列和解析状态。

上传状态：

| 状态 | UI |
| --- | --- |
| 待上传 | 文件名、大小、删除按钮 |
| 上传中 | 进度条、百分比 |
| 解析中 | `正在解析并写入索引...` |
| 成功 | 绿色状态点、`已入库` |
| 失败 | 红色状态点、错误原因、重试按钮 |

### 5.3 来源列表

上传成功后显示来源列表：

```text
[ ] 全选
[x] PDF 图标  创维 AI 研究院实习介绍资料(1).pdf
[ ] MD 图标   设备维护手册.md
[ ] CSV 图标  compressor_energy.tsv
```

文件条目字段：

- `doc_id`
- `title`
- `source_type`
- `source_uri`
- `doc_version`
- `chunk_count`
- `created_at`
- `status`

交互：

- 点击 checkbox 切换是否参与当前问答。
- 点击文件名打开来源详情抽屉。
- 全选只选择 `status=ready` 的来源。
- 解析中和失败文件不可勾选。

来源详情抽屉可展示：

- 文件名
- 类型
- 版本
- chunk 数
- ACL / tenant 信息
- 最近更新时间
- 删除按钮

## 6. 中间：对话面板

### 6.1 无来源空状态

参考 `1.png`。

内容：

- 欢迎图标
- 标题：`让我们开始构建知识库...`
- 说明：`添加来源后，你可以基于资料提问、生成摘要和创建思维导图。`
- 三个建议按钮：
  - `了解新主题`
  - `创建新内容`
  - `推进项目`

无来源时，底部输入框仍显示，但发送按钮禁用，并提示 `请先添加并选择来源`。

### 6.2 有来源首页

参考 `4.png`。

当用户选择来源后，对话区生成一个资料概览：

```text
标题：{知识库或文件主题}
副标题：{N 个来源 · YYYY年M月D日}
正文：系统基于来源生成的 1-2 段摘要
建议问题：3 个
```

MVP 实现方式：

- 如果后端暂时没有摘要接口，前端可以在首次选择来源后调用 `/query`：

```json
{
  "query": "请基于当前选中的来源，生成一段简洁的资料概览，并给出3个后续可追问问题。",
  "query_mode": "text",
  "source_types": [],
  "candidate_limit": 20,
  "context_limit": 5
}
```

- 摘要结果作为普通 AI 消息展示。
- 建议问题可以由前端静态生成，也可以从回答中解析。

### 6.3 对话输入框

固定在对话面板底部。

字段：

- placeholder：`提问或创作内容`
- 左侧可预留附件/工具入口，但 MVP 不实现附件。
- 右侧显示 `{N} 个来源`
- 发送按钮使用箭头图标。

行为：

- Enter 发送。
- Shift + Enter 换行。
- 输入为空时发送按钮禁用。
- 没有选中来源时发送按钮禁用。
- 请求中显示 loading，不允许重复提交同一条问题。

### 6.4 消息样式

用户消息：

- 右对齐。
- 浅蓝灰背景。
- 圆角 16px。
- 最大宽度约 520px。

AI 消息：

- 左侧主内容区域。
- 不使用卡片包裹整段回答。
- 正文宽度控制在 720px-860px。
- 支持 Markdown：标题、列表、加粗、代码块。
- 引用角标显示为小圆点数字，例如 `1`、`2`、`3`。

AI 消息操作：

- `保存到笔记`
- `复制`
- `点赞`
- `点踩`

点赞/点踩调用 `/feedback`：

```json
{
  "request_id": "xxx",
  "rating": 1,
  "comment": "",
  "selected_doc_ids": ["doc_a", "doc_b"]
}
```

点踩后可以弹出一个小输入框：

```text
这条回答哪里有问题？（可选）
```

### 6.5 引用展示

回答返回的 `citations` 来自后端 `/query`：

```json
{
  "doc_id": "string",
  "title": "string",
  "source_uri": "string",
  "source_type": "pdf",
  "chunk_index": 0,
  "score": 0.82,
  "rerank_score": 0.91,
  "metadata": {}
}
```

引用交互：

- 鼠标悬停角标时显示来源浮层。
- 点击角标打开右侧或中间抽屉，展示引用片段详情。
- 详情中显示标题、chunk index、score、rerank_score、source_uri。

如果后端当前没有返回 chunk 原文，引用详情先展示 metadata 和来源标题；后续应补充后端字段 `text_preview`。

## 7. 右侧：Studio 面板

### 7.1 功能卡片

参考 `6.png`。视觉上保留 2-3 列彩色功能入口：

- 思维导图：可用

未接通的 Studio 能力不展示卡片。

思维导图卡片行为：

- 无来源：提示 `请先选择来源`
- 有来源：创建思维导图生成任务

### 7.2 Studio 空状态

内容：

```text
Studio 输出将保存在此处。
添加来源后，点击即可生成思维导图。
```

底部按钮：

```text
[添加来源]
```

### 7.3 Studio 列表状态

参考 `7.png`。

列表包含：

- 生成中条目
- 已生成条目
- 每条右侧更多菜单

条目结构：

```text
[思维导图图标] 实习招聘思维导图
1 个来源 · 9 分钟前
```

生成中状态：

```text
正在生成思维导图...
基于 1 个来源
```

### 7.4 思维导图详情页

参考 `8.png`。

进入方式：

- 点击 Studio 列表中的思维导图条目。

顶部：

```text
Studio > 应用
标题：实习招聘思维导图
[查看 1 个来源]
```

右上角：

- 全屏
- 更多菜单

画布工具：

- 适配视图
- 放大
- 缩小
- 下载

底部反馈：

- `优质内容`
- `劣质内容`

画布内容：

- 中心节点为文档主题。
- 一级节点为主要主题。
- 二级节点为具体事实或结论。
- 节点不需要支持复杂编辑，MVP 只读即可。

## 8. 思维导图数据结构

前端建议统一使用树结构：

```ts
type MindMapNode = {
  id: string;
  label: string;
  children?: MindMapNode[];
  citationIds?: string[];
};

type MindMapArtifact = {
  id: string;
  title: string;
  sourceDocIds: string[];
  status: "generating" | "ready" | "failed";
  createdAt: string;
  updatedAt: string;
  root?: MindMapNode;
};
```

MVP 可以先让前端调用 `/query` 生成 JSON：

```json
{
  "query": "请基于当前选中的来源生成一份思维导图。只返回 JSON，结构为 {\"title\":\"...\",\"root\":{\"id\":\"root\",\"label\":\"...\",\"children\":[...]}}。",
  "query_mode": "text",
  "candidate_limit": 20,
  "context_limit": 8
}
```

长期建议新增后端接口：

```text
POST /artifacts/mindmap
GET  /artifacts
GET  /artifacts/{artifact_id}
DELETE /artifacts/{artifact_id}
```

## 9. 与后端 API 的关系

### 9.1 当前可直接使用的接口

健康检查：

```text
GET /health
GET /ready
```

检索：

```text
POST /search
```

问答：

```text
POST /query
```

反馈：

```text
POST /feedback
```

`/query` 请求示例：

```json
{
  "query": "有哪些职位提供？",
  "query_mode": "text",
  "history": [],
  "tenant_id": "team_a",
  "acl_groups": ["engineering"],
  "source_types": [],
  "candidate_limit": 20,
  "context_limit": 5
}
```

如果生产开启 `RAG_REQUIRE_AUTH_CONTEXT=1`，前端需要通过请求头传：

```text
Authorization: Bearer <token>
X-RAG-Tenant-ID: team_a
X-RAG-ACL-Groups: engineering,ops
```

### 9.2 前端上线前建议补的后端接口

当前后端还没有文件上传和文档列表 API。为了完成 NotebookLM 风格体验，建议补充：

```text
POST   /sources/upload
GET    /sources
GET    /sources/{doc_id}
DELETE /sources/{doc_id}
POST   /sources/{doc_id}/publish
```

来源列表响应建议：

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

上传响应建议：

```json
{
  "doc_id": "internship-guide",
  "status": "processing",
  "title": "创维 AI 研究院实习介绍资料(1).pdf"
}
```

## 10. 前端状态模型

建议前端按以下状态拆分：

```ts
type SourceStatus = "uploading" | "processing" | "ready" | "failed";

type SourceItem = {
  docId: string;
  title: string;
  sourceType: string;
  sourceUri: string;
  docVersion?: number;
  status: SourceStatus;
  selected: boolean;
  chunkCount?: number;
  error?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  requestId?: string;
  citations?: Citation[];
  status?: "sending" | "streaming" | "done" | "failed";
};

type Citation = {
  docId: string;
  title: string;
  sourceUri: string;
  sourceType: string;
  chunkIndex: number;
  score: number;
  rerankScore?: number;
};
```

## 11. 视觉规范

整体气质：安静、工具型、适合长时间阅读和问答。不要做营销页风格。

颜色：

- 页面背景：`#EEF2FA` 或接近的浅冷灰。
- 面板背景：`#FFFFFF`。
- 主文字：`#111827`。
- 次级文字：`#6B7280`。
- 边框：`#DDE3EE`。
- 主按钮：黑底白字。
- 用户消息：浅蓝灰 `#E9EDF9`。
- 思维导图节点：中心淡紫、一级淡蓝、二级淡绿。

控件：

- 面板圆角：8px。
- 按钮圆角：999px 仅用于胶囊按钮和输入框发送区。
- 普通卡片圆角：8px。
- 图标优先使用 `lucide-react`。
- 交互按钮优先使用图标按钮，并配 tooltip。

字体：

- 中文优先使用系统字体。
- 不使用负 letter-spacing。
- 不用 viewport width 动态缩放字体。
- 对话正文建议 16px-18px。
- 面板标题 16px-18px。

## 12. 组件拆分建议

```text
frontend/src/
├── app/
│   ├── AppShell.tsx
│   ├── WorkspacePage.tsx
│   └── SettingsDialog.tsx
├── components/
│   ├── sources/
│   │   ├── SourcePanel.tsx
│   │   ├── SourceUploadDialog.tsx
│   │   ├── SourceList.tsx
│   │   └── SourceDetailDrawer.tsx
│   ├── chat/
│   │   ├── ChatPanel.tsx
│   │   ├── ChatInput.tsx
│   │   ├── MessageList.tsx
│   │   ├── AssistantMessage.tsx
│   │   └── CitationPopover.tsx
│   ├── studio/
│   │   ├── StudioPanel.tsx
│   │   ├── StudioToolGrid.tsx
│   │   ├── ArtifactList.tsx
│   │   └── MindMapView.tsx
│   └── ui/
├── lib/
│   ├── api.ts
│   ├── auth.ts
│   └── types.ts
└── styles/
```

## 13. 实现阶段建议

### Phase 1：静态 UI

- 搭建三栏布局。
- 实现来源面板、对话面板、Studio 面板静态状态。
- 用 Playwright 网络拦截数据复现 `1.png`、`3.png`、`4.png`、`5.png`、`7.png`、`8.png`，运行时代码不保留静态样例数据。

### Phase 2：接入现有后端问答

- 配置 API Base URL。
- 接入 `/query`。
- 接入 `/feedback`。
- 显示 citations。
- 处理 loading、error、empty 状态。

### Phase 3：补齐来源管理

- 新增或接入 `/sources/upload`。
- 接入 `/sources`。
- 上传后刷新来源列表。
- 来源选择影响问答上下文。

### Phase 4：思维导图

- 先用 `/query` 生成 JSON mindmap。
- 前端渲染只读树图。
- 支持缩放、适配视图、下载 PNG 或 JSON。
- 后续替换为正式 `/artifacts/mindmap` 接口。

## 14. 验收标准

前端第一版完成时，应满足：

- 进入系统后直接看到三栏工作台。
- 未上传来源时，页面状态与 `1.png` 接近。
- 点击添加来源能打开类似 `2.png` 的上传弹窗。
- 上传成功后，左侧来源列表与 `3.png` 接近。
- 选择来源后，中间对话首页与 `4.png` 接近。
- 提问后，用户消息和 AI 回答布局与 `5.png` 接近。
- AI 回答能显示 citations，并能提交点赞/点踩。
- Studio 只展示已接通的“思维导图”入口。
- 思维导图生成中和列表状态与 `7.png` 接近。
- 思维导图详情与 `8.png` 接近，支持缩放和下载。
- 桌面端 1440px、移动端 390px 宽度下没有文字重叠和布局溢出。
