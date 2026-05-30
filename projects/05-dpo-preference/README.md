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

## 参考答案

1. 经典 RLHF 通常先做 SFT，让模型具备基本指令跟随能力；再用同一 prompt 下的 chosen/rejected 或人工排序数据训练 Reward Model，让它给回答质量打分；最后用 PPO 等强化学习算法优化策略模型，使模型生成能获得更高 reward 的回答，同时用 KL 约束避免偏离原模型太远。三阶段分别解决“会回答”“知道什么回答更好”“按偏好进一步优化生成策略”。

2. DPO 比 PPO 短，是因为它不显式训练 Reward Model，也不需要 PPO 的在线采样、优势估计、策略更新和复杂的 RL 稳定性调参。DPO 直接用 preference pair 构造一个分类式目标：让 policy 相对 reference 更偏向 chosen、远离 rejected。工程上通常只需要 policy model、reference model 和 chosen/rejected 数据，因此更容易离线训练和复现。

3. chosen/rejected 数据质量关键，因为 DPO 学到的是偏好差异，而不是绝对真理。如果 chosen 只是略好、rejected 明显有格式问题但内容并不差，模型可能学到表面模式；如果偏好标注互相矛盾，训练信号会互相抵消；如果数据覆盖不足，模型只会在少数风格上变好。高质量偏好对应该在同一 prompt 下有明确、可解释的优劣差异，例如事实性、完整性、安全性、遵循指令和表达质量。

4. DPO 不是简单背诵 chosen。它优化的是 chosen 相对 rejected 的 log probability margin，并通过 reference model 保留原模型分布约束。也就是说，模型学习“在类似上下文中哪些回答特征更受偏好”，例如更直接回答问题、更少编造、更符合格式，而不是把训练集里的 chosen 文本原样记住。判断是否学到偏好，需要在未见 prompt 上比较输出，而不是只看训练样本复现。
