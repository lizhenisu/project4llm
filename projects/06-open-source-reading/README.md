# 06-open-source-reading

目标：通过优秀开源项目学习真实 LLM 工程。

## 第一批项目

- `karpathy/nanoGPT`
- `huggingface/trl`
- `hiyouga/LLaMA-Factory`

## 阅读模板

- 项目名称：
- 入口命令：
- 入口文件：
- 数据处理路径：
- 模型构建路径：
- 训练循环路径：
- 保存 checkpoint 路径：
- 评估路径：
- 配置系统：
- 依赖管理：
- 我修改过的地方：
- 修改后的现象：

## 精读顺序

1. 跑 README 的最小示例。
2. 找入口脚本和参数解析。
3. 追踪数据从原始文件到 batch 的路径。
4. 追踪模型从 config 到 forward 的路径。
5. 追踪 loss、backward、optimizer、scheduler、checkpoint。
6. 修改一个小参数，记录行为变化。

## 项目记录文件

每读一个项目，新建一份 Markdown：

```text
projects/06-open-source-reading/nanogpt-reading.md
projects/06-open-source-reading/trl-reading.md
projects/06-open-source-reading/llama-factory-reading.md
```

每份记录必须包含一张调用链：

```text
命令行 -> 参数解析 -> 数据加载 -> batch 构造 -> model forward -> loss -> backward -> checkpoint
```
