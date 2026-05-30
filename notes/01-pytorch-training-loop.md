# PyTorch 训练循环笔记

## 最小训练循环

```python
for batch in dataloader:
    x, y = batch
    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 每一步在做什么

- `model(x)`：前向传播，得到预测值。
- `criterion(logits, y)`：计算预测和真实标签之间的差距。
- `optimizer.zero_grad()`：清空上一轮累计的梯度。
- `loss.backward()`：反向传播，计算每个参数的梯度。
- `optimizer.step()`：根据梯度更新模型参数。

## 必须理解的问题

1. 为什么每轮都要 `zero_grad()`？
2. `loss.backward()` 后，梯度保存在哪里？
3. batch size、learning rate、epoch 分别影响什么？
4. 训练 loss 下降但验证集效果变差说明什么？

## 参考答案

1. PyTorch 默认会把梯度累加到参数的 `.grad` 上，而不是每次反向传播自动覆盖。这样设计是为了支持 gradient accumulation、多 loss 累加等场景。如果普通训练循环里不调用 `zero_grad()`，上一轮 batch 的梯度会混到当前 batch 里，实际更新方向就不再对应当前 loss，训练会变得不可控。

2. `loss.backward()` 会沿计算图反向传播，把每个 `requires_grad=True` 的叶子参数的梯度写到参数对象的 `.grad` 字段里。例如 `model.linear.weight.grad` 保存的是当前 loss 对这个权重矩阵的偏导。optimizer 本身不负责求梯度，它只在 `optimizer.step()` 时读取这些 `.grad` 并更新参数。

3. batch size 决定每次梯度估计用多少样本：batch 越大，梯度更稳定但显存占用更高，单步更新次数更少；batch 越小，噪声更大但可能有正则化效果。learning rate 决定每次沿梯度方向走多远：太大容易震荡或发散，太小收敛慢甚至看起来不下降。epoch 表示完整遍历训练集的次数：epoch 太少可能欠拟合，太多可能过拟合。

4. 训练 loss 下降但验证集效果变差通常说明模型在记忆训练集噪声或分布细节，泛化能力下降，也就是过拟合。处理方式包括增加数据、数据增强、减小模型容量、加 weight decay/dropout、早停、降低训练轮数、检查 train/valid 分布是否一致，以及确认验证集构造没有泄漏或标注问题。

## 实验记录模板

- 日期：
- 数据集：
- 模型：
- loss：
- optimizer：
- learning rate：
- batch size：
- epoch：
- 训练结果：
- 遇到的问题：
- 下一步：
