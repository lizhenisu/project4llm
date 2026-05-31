# model.py 默写路线图

这个文件不是答案代码，而是写作顺序。你可以把 `model.py` 关掉，只看这份路线图，从空文件一步一步把模型默写出来。

每一步只关注一个小目标：先写能跑的骨架，再逐层补功能，最后补推理生成和 KV cache。

## 0. 先写文件头

目标：准备后面会用到的标准库和 PyTorch 模块。

要写的东西：

- 导入 `math`
- 从 `dataclasses` 导入 `dataclass`
- 导入 `torch`
- 从 `torch` 导入 `nn`
- 导入 `torch.nn.functional as F`

自检：

- 文件能被 Python import。
- 不要急着写模型，先把依赖放齐。

## 1. 写配置类 TinyGPTConfig

目标：把模型超参数集中放到一个配置对象里。

要写的东西：

- 用 `@dataclass` 定义 `TinyGPTConfig`
- 字段包括 `vocab_size`
- 字段包括 `block_size`
- 字段包括 `n_layer`
- 字段包括 `n_head`
- 字段包括 `n_embd`
- 字段包括 `dropout`
- 字段包括可选的 `eos_token_id`

脑内结构：

```text
config 决定模型长什么样：
词表多大、上下文多长、几层、几个 head、hidden 多宽、dropout 多少。
```

自检：

- `n_embd` 是总 hidden size。
- 每个 head 的维度后面会用 `n_embd // n_head` 得到。

## 2. 写 KV cache 类型别名

目标：给 KV cache 起名字，让后面函数签名更容易读。

要写的东西：

- `LayerKVCache` 表示单层的 `(k, v)`
- `KVCache` 表示多层 cache 列表

脑内结构：

```text
一层 cache: (past_k, past_v)
整个模型 cache: [第 0 层 cache, 第 1 层 cache, ...]
```

自检：

- 单层的 `k/v` 形状以后会是 `[batch, n_head, cached_seq_len, head_dim]`。

## 3. 写 RMSNorm

目标：写一个只按 RMS 缩放、不减均值的归一化层。

要写的东西：

- 类继承 `nn.Module`
- `__init__` 保存 `eps`
- `__init__` 创建可学习参数 `weight`
- `forward` 中沿最后一维计算平方均值
- 用 `rsqrt` 得到 `1 / sqrt(...)`
- 返回归一化后的 `x` 再乘 `weight`

脑内公式：

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

输入输出形状：

```text
输入: [batch, seq_len, n_embd]
输出: [batch, seq_len, n_embd]
```

自检：

- `mean` 要沿 `dim=-1`。
- `keepdim=True`，否则后面不好 broadcast。

## 4. 写 apply_rope 函数

目标：把 RoPE 位置旋转应用到 q 或 k 上。

要写的东西：

- 函数参数是 `x`、`inv_freq`、`start_pos`
- 从 `x` 的倒数第二维拿到 `seq_len`
- 根据 `start_pos` 和 `seq_len` 生成 positions
- 用 positions 和 inv_freq 做外积，得到每个位置、每组维度的旋转角度
- 对角度求 `cos` 和 `sin`
- 把最后一维拆成偶数维和奇数维
- 用二维旋转公式得到旋转后的偶数维和奇数维
- 把两两维度重新拼回最后一维

脑内结构：

```text
原始最后一维:
[x0, x1, x2, x3, ...]

分组:
(x0, x1), (x2, x3), ...

每一组做二维旋转:
x_even' = x_even * cos - x_odd * sin
x_odd'  = x_even * sin + x_odd * cos
```

输入输出形状：

```text
输入 x: [batch, n_head, seq_len, head_dim]
输出:   [batch, n_head, seq_len, head_dim]
```

自检：

- `head_dim` 必须是偶数。
- `cos/sin` 要能 broadcast 到 `[batch, n_head, seq_len, head_dim / 2]`。

## 5. 写 CausalSelfAttention 的 __init__

目标：准备 attention 层需要的参数、mask 和 RoPE 频率。

要写的东西：

- 检查 `n_embd` 能被 `n_head` 整除
- 计算 `head_dim`
- 检查 `head_dim` 是偶数
- 定义一个线性层一次性投影出 q/k/v
- 定义输出投影层
- 定义 attention dropout 和 residual dropout
- 注册 causal mask buffer
- 计算 RoPE 的 `inv_freq`
- 注册 RoPE 频率 buffer

脑内结构：

```text
x -> qkv linear -> q, k, v
attention 输出 -> proj -> dropout
```

自检：

- qkv linear 的输出维度是 `3 * n_embd`。
- causal mask 形状最后要能 broadcast 到 attention score。
- RoPE 的 `inv_freq` 长度是 `head_dim / 2`。

## 6. 写 CausalSelfAttention.forward 的前半段

目标：把输入 hidden states 变成多头 q/k/v。

要写的东西：

- 从 `x.shape` 取出 `batch, seq_len, channels`
- 用 qkv linear 得到大 tensor
- 按最后一维 split 成 q/k/v
- 把 q/k/v reshape 成多头格式
- transpose 让 head 维提前

脑内形状变化：

```text
x: [B, T, C]
q/k/v: [B, T, C]
reshape: [B, T, H, D]
transpose: [B, H, T, D]
```

自检：

- `C = n_embd`
- `H = n_head`
- `D = head_dim`
- `H * D = C`

## 7. 补 CausalSelfAttention.forward 的 RoPE 和 KV cache

目标：给 q/k 加位置旋转，并在推理时拼接历史 k/v。

要写的东西：

- 默认 `past_len = 0`
- 如果传入 `past_kv`，取出 `past_k, past_v`
- 从 `past_k` 的序列维得到 `past_len`
- 对 q 和新 k 应用 RoPE，`start_pos` 用 `past_len`
- 如果有 `past_kv`，把历史 k 和新 k 沿序列维拼接
- 如果有 `past_kv`，把历史 v 和新 v 沿序列维拼接

脑内结构：

```text
不用 cache:
q 看当前序列，k/v 也是当前序列

用 cache:
q 是当前新 token 的 q
k/v 是历史 k/v + 当前新 k/v
```

自检：

- 拼接维度是 `dim=-2`，也就是序列长度维。
- RoPE 只作用在 q/k，不作用在 v。

## 8. 补 CausalSelfAttention.forward 的注意力计算

目标：完成 masked self-attention。

要写的东西：

- 计算 `total_len = past_len + seq_len`
- 检查是否超过 `block_size`
- 用 `q @ k.transpose(-2, -1)` 得到 attention score
- 除以 `sqrt(head_dim)`
- 根据 `past_len` 和 `total_len` 切 causal mask
- 对 mask 为 0 的位置填 `-inf`
- softmax 得到注意力权重
- attention dropout
- 用注意力权重乘 v
- transpose 回 `[B, T, H, D]`
- contiguous 后 view 回 `[B, T, C]`
- 输出投影和 dropout
- 如果 `use_cache=True`，返回输出和当前完整 `(k, v)`
- 否则只返回输出

脑内形状：

```text
q:      [B, H, T_new, D]
k:      [B, H, T_all, D]
scores: [B, H, T_new, T_all]
v:      [B, H, T_all, D]
out:    [B, H, T_new, D]
final:  [B, T_new, C]
```

自检：

- 普通训练时 `T_new == T_all`。
- cache 推理时 `T_new` 通常是 1，`T_all` 是历史长度加 1。

## 9. 写 FeedForward

目标：写 Transformer block 里的 MLP。

要写的东西：

- hidden dim 通常设为 `4 * n_embd`
- 定义 `gate` 线性层
- 定义 `up` 线性层
- 定义 `down` 线性层
- 定义 dropout
- forward 中使用 SwiGLU 结构：激活 gate 后乘 up，再 down 投回 n_embd

脑内结构：

```text
x -> gate -> silu
x -> up
两路相乘 -> down -> dropout
```

输入输出形状：

```text
输入: [B, T, C]
输出: [B, T, C]
```

自检：

- `gate` 和 `up` 输出都是 `hidden_dim`。
- `down` 把 `hidden_dim` 投回 `n_embd`。

## 10. 写 Block

目标：把 RMSNorm、Attention、MLP 串成一个 Decoder block。

要写的东西：

- `__init__` 中创建第一层 norm
- 创建 self-attention
- 创建第二层 norm
- 创建 feed-forward
- forward 中先 norm 再 attention
- attention 输出走残差连接
- 再 norm 后过 MLP
- MLP 输出再走残差连接
- 如果使用 cache，要把 attention 返回的 present_kv 传出去

脑内结构：

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

自检：

- 这是 pre-norm 结构。
- `Block` 的输入输出形状都保持 `[B, T, C]`。

## 11. 写 TinyGPT.__init__

目标：搭出整个 decoder-only language model。

要写的东西：

- 保存 config
- 创建 token embedding
- 创建 embedding 后的 dropout
- 用 `ModuleList` 创建多层 Block
- 创建最终 RMSNorm
- 创建 lm head，把 hidden 投到 vocab
- 绑定 token embedding 和 lm head 的权重
- 调用初始化函数

脑内结构：

```text
input_ids
-> token embedding
-> dropout
-> N 个 Block
-> final norm
-> lm_head
-> logits
```

自检：

- `lm_head` 输出维度是 `vocab_size`。
- 权重绑定是让 `token_embedding.weight` 和 `lm_head.weight` 指向同一份参数。

## 12. 写 _init_weights

目标：给 Linear 和 Embedding 初始化。

要写的东西：

- 如果模块是 `nn.Linear`，初始化 weight 为小标准差正态分布
- 如果 Linear 有 bias，把 bias 置零
- 如果模块是 `nn.Embedding`，初始化 weight 为小标准差正态分布

自检：

- 这个函数会被 `self.apply(...)` 自动递归调用到所有子模块。

## 13. 写 TinyGPT.forward 的输入检查

目标：处理训练和 cache 推理共享的入口逻辑。

要写的东西：

- 从 `input_ids.shape` 得到当前 `seq_len`
- 如果有 `past_kvs`，从第一层 cache 得到 `past_len`
- 检查 `past_len + seq_len` 不超过 `block_size`
- 如果传了 labels，同时又 `use_cache=True`，报错

脑内结构：

```text
训练: labels 有值，use_cache=False
推理 cache: labels=None，use_cache=True
```

自检：

- cache 模式不能计算训练 loss，因为它只处理新 token。

## 14. 写 TinyGPT.forward 的主体

目标：完成从 token id 到 logits 的前向传播。

要写的东西：

- input ids 进入 token embedding
- 过 dropout
- 准备 `present_kvs` 列表
- 如果没有传 `past_kvs`，创建和 block 数量相同的空 cache 列表
- 逐层遍历 block 和对应 past_kv
- 如果 `use_cache=True`，收集每层新的 present_kv
- 否则正常前向
- 过最终 norm
- 过 lm head 得到 logits

输入输出形状：

```text
input_ids: [B, T]
embedding 后: [B, T, C]
logits: [B, T, vocab_size]
```

自检：

- block 数量要和 past_kvs 数量一致。
- cache 模式返回值比普通模式多一个 `present_kvs`。

## 15. 补 TinyGPT.forward 的 loss

目标：实现 causal language modeling loss。

要写的东西：

- 默认 `loss = None`
- 如果传入 labels，计算 next-token prediction loss
- logits 去掉最后一个时间步
- labels 去掉第一个时间步
- reshape 成二维 logits 和一维 labels
- 用 `F.cross_entropy`
- 根据是否 use_cache 返回不同元组

脑内对齐：

```text
logits at position 0 -> 预测 label at position 1
logits at position 1 -> 预测 label at position 2
...
```

自检：

- `logits[:, :-1, :]`
- `labels[:, 1:]`
- reshape 前最好 contiguous。

## 16. 写普通 generate

目标：不用 KV cache，逐 token 自回归生成。

要写的东西：

- 加 `@torch.no_grad()`
- 设置 eval 模式
- 检查 temperature 必须大于 0
- 检查 `min_new_tokens` 合法
- 解析最终使用的 `stop_id`
- 如果有 EOS，创建 batch 级 `finished` 布尔向量，初始全 False
- 循环 `max_new_tokens` 次
- 每步截取最后 `block_size` 个 token 当 context
- 调用 `self(context)` 得到 logits
- 取最后一个位置的 logits
- 除以 temperature
- 如果采样，softmax 后 multinomial
- 如果不采样，argmax
- 对已经 finished 的样本，把 next_id 强制改成 EOS
- 达到 `min_new_tokens` 后，更新 finished
- 把 next_id 拼到 input_ids 后面
- 如果所有样本 finished，提前 break
- 返回完整 input_ids

脑内结构：

```text
每一步:
已有 token -> 模型 -> 下一个 token -> 拼回去
```

自检：

- 普通 generate 会反复重算整个 context。
- `max_new_tokens` 是硬上限。
- EOS 是提前停止条件。

## 17. 写 generate_with_cache

目标：用 KV cache 避免每一步重复计算历史 token。

要写的东西：

- 函数参数基本和普通 generate 一样
- 加 `@torch.no_grad()`
- 设置 eval 模式
- 检查 temperature 和 `min_new_tokens`
- 检查 `prompt_len + max_new_tokens <= block_size`
- 初始化 stop_id 和 finished
- 先用完整 prompt 做一次 prefill，拿到 logits 和 past_kvs
- 循环生成新 token
- 每步根据当前 logits 选 next_id
- 处理 finished 样本和 EOS
- 拼接 next_id 到 input_ids
- 如果全 finished，提前 break
- 如果还要继续生成，就只把 `next_id` 喂给模型，并传入 past_kvs
- 更新 logits 和 past_kvs
- 返回完整 input_ids

脑内对比：

```text
普通 generate:
每步喂完整上下文

KV cache generate:
第一步喂完整 prompt
后面每步只喂刚生成的 1 个 token
```

自检：

- cache 模式里 q 的长度通常是 1。
- k/v 的长度会随着生成一步步增长。
- cache 输出应该和普通 generate 在 greedy 模式下一致。

## 18. 最后跑 smoke test

目标：确认你默写出来的模型至少能完成基础流程。

建议运行：

```bash
.venv/bin/python projects/02-transformer-from-scratch/smoke_test.py
```

应该重点看：

- logits shape 是否正确
- loss 是否能算出来
- 普通 generate 是否能生成
- KV cache generate 是否能生成
- greedy 情况下 cache 和 no-cache 输出是否一致
- EOS 停止逻辑是否能触发

## 19. 默写时的推荐节奏

第一次默写不要追求一次写完。按这个顺序来：

1. 只写到 `TinyGPT.forward`，先让 logits 和 loss 跑通。
2. 再写普通 `generate`。
3. 再写 attention 里的 KV cache。
4. 最后写 `generate_with_cache`。

如果卡住，优先检查形状：

```text
input_ids: [B, T]
x: [B, T, C]
q/k/v: [B, H, T, D]
scores: [B, H, T_query, T_key]
logits: [B, T, vocab_size]
```

形状想通了，这个文件就已经写出来一半了。
