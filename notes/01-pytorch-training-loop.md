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
