# 实验报告模板

每次训练或微调都按这个模板记录。面试时项目讲不清，通常是因为实验过程没有记录。

## 基本信息

- 日期：
- 项目：
- 目标：
- 代码入口：
- git commit：
- 运行命令：

## 数据

- 数据来源：
- 样本数量：
- train/valid/test 划分：
- 清洗规则：
- token 长度分布：
- 是否 packing：

## 模型

- 模型名称：
- 参数量：
- hidden size：
- layer 数：
- attention heads：
- max sequence length：
- tokenizer：

## 训练参数

- optimizer：
- learning rate：
- batch size：
- gradient accumulation：
- epoch/steps：
- precision：
- LoRA rank/alpha：
- target modules：
- checkpoint 策略：

## 结果

- train loss：
- eval loss：
- perplexity：
- 显存占用：
- 训练耗时：
- 样例输入：
- 样例输出：

## 失败与修复

- 遇到的问题：
- 定位方法：
- 修改内容：
- 修改后结果：

## 面试口述版

用 5 句话回答：

1. 我解决了什么问题。
2. 我用了什么数据和模型。
3. 我做了哪些关键工程处理。
4. 实验结果如何。
5. 如果继续优化，下一步做什么。
