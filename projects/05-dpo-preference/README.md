# 05-dpo-preference

目标：理解偏好数据和 DPO 的输入格式。

当前目录先提供离线可运行的 chosen/rejected 数据构造练习；真实 DPO 训练需要本地模型或联网下载模型。

## 离线练习

```bash
python projects/05-dpo-preference/build_preference_dataset.py
```

脚本会输出一份 JSONL，每行包含：

- `prompt`：用户问题。
- `chosen`：更好的回答。
- `rejected`：更差的回答。

## DPO 的直觉

DPO 不训练单独的 reward model，而是直接让模型提高 chosen 的相对概率，降低 rejected 的相对概率。它仍然需要一个 reference model 来约束模型不要偏离原始分布太远。

## 面试必须能讲清楚

- RLHF 的 SFT、Reward Model、PPO 三阶段。
- DPO 为什么比 PPO 工程流程更短。
- chosen/rejected 数据质量为什么关键。
- DPO 不是让模型“只记住 chosen”，而是学习偏好排序。
