# 第 9 章 大模型算子：Attention、Norm 与 KV Cache

[返回总目录](../README.md) · [上一章：后端、调试与调优](08-backend-debug.md) · [下一章：现代扩展](10-modern-extensions.md)

前面章节的算子都是“一个输入块 → 一个输出块”。这一章进入 Transformer 前向里的三类真实工作：

- **Attention**：Q/K/V、score、在线 softmax、因果 mask，以及不保存完整概率矩阵的状态更新；
- **Norm**：一行 hidden 的均值/方差，LayerNorm 与 RMSNorm 的合同差异；
- **KV Cache**：prefill/decode 的输入量差异、连续索引、分页教学推导和请求级边界。

它们不是三个孤立 kernel，而是同一条模型数据流上的不同位置：

```text
hidden [B, S, D]
  → Q/K/V 投影（GEMM）
  → Attention（沿 key 维归约）
  → 输出投影（GEMM）
  → 残差 + Norm（沿 hidden 维归约）
  → MLP（GEMM + 激活）
  → 下一层
```

KV Cache 则跨越整个生成过程：它保存历史 token 的 K/V，让 decode 每步不必重新计算全部前缀。

## 本章地图

| 部分 | 归约/访问轴 | 核心状态 | 关心的系统边界 |
|---|---|---|---|
| 9.2 Attention | 每个 query 沿所有 key | `m_i`、`l_i`、`acc` 三状态 | 不保存完整 score/概率矩阵 |
| 9.3 Norm | 每一行沿 hidden 维 | 均值、方差/逆标准差 | 参数共享、残差与 RMSNorm 合同 |
| 9.4 KV Cache | 历史 token 维 | 页表、有效长度、KV head | prefill/decode、分页、请求隔离 |
| 9.5 实际运行 | 小矩阵行归约 | 均值/方差与均方根 | 单块教学 kernel 的数值正确性 |

**读法：** 始终先问“沿哪一维归约、哪些值是历史状态、哪些值是本次输入”。三者都出现 sum/max，但归约轴和状态含义不同，不能因为“都是融合归约”就互相替换。

<a id="sec-9-1"></a>
## 9.1 从单算子走到 LLM 数据流

本章拆成三个连续闭环，固定现代 SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。Attention 与 KV 以静态带读和教学推导为主；9.5 节给出小矩阵 LayerNorm/RMSNorm 的实际运行。

1. [Attention：在线softmax怎样合并两块K/V](#sec-9-2)。读官方06教程，从Q/K/V索引、score、概率、状态更新到输出，手算与全量softmax相同的结果。
2. [Norm：一行hidden怎样变成归一化输出](#sec-9-3)。读官方05教程的三段循环，再明确推导RMSNorm的不同合同。
3. [KV：prefill、decode、分页与请求边界](#sec-9-4)。先从真实Attention的连续索引出发，再手算教学page table，区分kernel与推理服务职责。

阅读时保留一个外层形状：hidden张量常可写成 `[batch, sequence, hidden]`；Q/K/V投影后按head组织。Norm通常沿hidden维归约，Attention沿sequence/key维归约。两者都出现sum/max，但归约轴和中间状态不同，不能因“都是融合归约”就替换彼此。

这三个算子页不构成完整模型：没有权重加载、tokenizer、采样器、调度器、张量并行通信或完整请求计时。它们能解释局部数据流和可能减少的中间读写；是否改善真实LLM延迟须在同模型、同请求和同正确性合同下另行测量。

> 回到论文：2019基础论文§3–§5有tile编程与编译机制，**没有本章在线 Attention、KV 分页或现代 LLM 服务实现**。后续特性按本仓库教程与明确教学推导解释，不套用旧论文实验结果。见[论文定位表](../appendices/reference.md#sec-a-1)。

**从入口到第一个算子。** 先看 Attention：它把 GEMM 的二维块和 Softmax 的归约组合在同一个 kernel 里，而且必须在遍历 K/V 块的过程中增量更新状态。9.2 会用一个能手算的两块 K/V 例子，把“在线 softmax”从公式落到 `m_i`、`l_i`、`acc` 三个变量上。

<a id="sec-9-2"></a>
## 9.2 两块 K/V，怎样得到同一个 softmax 输出

Attention 的 score 是每个 query（Q）与各 key（K）的相似度分数；softmax 将同一行的分数变成权重，再用这些权重对 value（V）求加权和。下文「在线」指读到新一块 K/V 时就更新已有的最大值、分母和加权和。

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。带读[06-fused-attention.py](../../triton/python/tutorials/06-fused-attention.py)的forward路径；小数例为纸笔推演，不是合法硬件tile配置或实际运行结果。

### 1. 先把矩阵乘的两边说清

给一个head，Q形状 `[S_q,D]`，K/V形状 `[S_k,D]`。score `Q @ K.T * sm_scale` 是 `[S_q,S_k]`；每个query沿所有key做softmax，再乘V，输出 `[S_q,D]`。因果Attention限制query位置i只能看key位置j≤i。

普通分步实现可能把整个score和概率矩阵放到显存。官方融合kernel保留一块Q，依次读取K/V块，用少量行状态累积输出；这解释了减少大中间矩阵显存读写的方向，未给本机速度倍数。

本部分先选非因果、非FP8、普通acc修正分支，避免把不同硬件分支混到同一个数例。

### 2. 一个program负责哪块Q

真实源码177–221行 `_attn_fwd`，非连续摘录：

```python
dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
start_m = tl.program_id(0)
off_hz = tl.program_id(1)
off_z = off_hz // H
off_h = off_hz % H
offset_y = off_z * (N_CTX * H) + off_h * N_CTX
qo_offset_y = offset_y + start_m * BLOCK_M
offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
qk_scale = sm_scale
qk_scale *= 1.44269504
q = desc_q.load([qo_offset_y, 0])
```

`H` 是head数，`N_CTX`是每个序列的上下文长度，`Z`是batch数；host把它们传进来。grid第一维选query块，第二维把batch/head合并为 `off_hz`。整数除法/余数恢复batch与head。

例如Z=2、H=3、N_CTX=8、BLOCK_M=4，`off_hz=4,start_m=1`：`off_z=1,off_h=1`，`offset_y=1*24+1*8=32`，Q起始行36。该program处理本batch/head的query位置4..7，而不是batch0的行36。

`desc_q` 在195–206行由 `_maybe_make_tensor_desc` 接收指针或host descriptor，设置扁平shape `[Z*H*N_CTX,HEAD_DIM]`、stride `[HEAD_DIM,1]` 和块shape `[BLOCK_M,HEAD_DIM]`。因此load二维坐标能对应到原存储。descriptor是内存访问描述，不是Q值本身；设备执行load后才取得Q块。

`m_i/l_i` 每query行一个fp32状态；`acc` 每query行、每输出维一个fp32累加值。初始 `l_i=1` 不会多贡献一个概率，因为第一次 `alpha=exp2(-inf-new_max)=0`，旧分母被乘为0。

`1.44269504≈1/ln2` 将自然指数softmax转换成 `exp2`：`exp(s)=2^(s/ln2)`。不是把softmax温度随意改成1.4427。

### 3. caller怎样把状态交给inner

同文件225–230行，非因果时外层 `STAGE=1`，摘录：

```python
if STAGE & 1:
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                    desc_k, desc_v,
                                    off_hz, dtype, start_m, qk_scale,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n, N_CTX,
                                    warp_specialize, IS_HOPPER)
```

`&` 是位运算，用stage标志决定调用哪些区域。`acc/l_i/m_i/q` 是刚创建/加载的值；K/V descriptor来自host/外层构造；其余索引、shape和目标开关都在外层可知。inner接收 `4-STAGE=3`，选择全部key范围；返回的新三元组重新绑定给caller的三个状态名。

因此inner不是独立启动一个新GPU kernel。它是同一JIT设备程序中的辅助函数，保留同一query块、推进key循环。

### 4. 每轮先算score，再换共同的最大值

inner第55–83行，非连续摘录，保留非因果分支：

```python
else:
    lo, hi = 0, N_CTX
offset_y = off_hz * N_CTX
offsetk_y = offset_y + lo
offsetv_y = offset_y + lo
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = desc_k.load([offsetk_y, 0]).T
    qk = tl.dot(q, k)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)
```

`lo/hi` 规定这次stage处理的key区间。`offsetk_y` 是K descriptor扁平行索引；K先load `[BLOCK_N,D]` 再转置成 `[D,BLOCK_N]`。Q为 `[BLOCK_M,D]`，所以 `qk` 为 `[BLOCK_M,BLOCK_N]`。

`tl.multiple_of` 给编译器整除信息，不会把start_n向上取整。上游循环本来按BLOCK_N步进，才可作此承诺。

`tl.max(qk,1)` 沿key轴归约，每query行一个最大值，乘缩放后与历史 `m_i` 取更大值，得到 `m_ij`。`[:,None]` 把行向量扩成列，广播到当前所有key。`p` 是以新共同最大值平移的非归一化权重。

`alpha` 把历史状态从旧最大值基准重标到新基准。若新块更大，alpha小于1；旧概率不是丢掉，而是缩小到同一个坐标系。

### 5. 不存完整概率矩阵，怎样更新输出

同文件93–108行，非连续摘录，选择非FP8与普通acc分支：

```python
acc = acc * alpha[:, None]
v = desc_v.load([offsetv_y, 0])
p = p.to(dtype)
acc = tl.dot(p, v, acc)
l_i = l_i * alpha + l_ij
m_i = m_ij
offsetk_y += BLOCK_N
offsetv_y += BLOCK_N
return acc, l_i, m_i
```

`acc` 先做历史分子修正，`v` 是 `[BLOCK_N,D]`；`p @ v` 得 `[BLOCK_M,D]`，第三个实参将它加到原acc。`l_i` 同样缩放旧分母，再加当前块权重和。分子分母用同一个alpha，最终比值不变。

`p.to(dtype)` 是性能/精度折中，真实有限精度结果不必与全fp32数学完全相同。上游还有warp specialization下reshape/split再修正acc的分支，以及FP8 V的不同存储布局；本部分没有把这些分支抄进普通fp16路径。

更新K/V偏移后处理下一块。循环之外返回当前三状态，caller在241–245行收尾：

```python
m_i += tl.math.log2(l_i)
acc = acc / l_i[:, None]
m_ptrs = M + off_hz * N_CTX + offs_m
tl.store(m_ptrs, m_i)
desc_o.store([qo_offset_y, 0], acc.to(dtype))
```

`acc/l_i` 才是归一化输出。`M`保存行log-sum-exp的base2形式，后向计算会用；它不是输出矩阵O。`desc_o`复用Q对应的query/head坐标，将结果写到host预分配输出。

### 6. 手算两块合并

为避免指数小数，直接在转换后的base2 score坐标中选一行四个score `[0,1,2,1]`，V单输出维为 `[1,2,4,8]`。两key一块，忽略量化/舍入，仅算数学合同。

初始 `m=-inf,l=1,a=0`。

第一块 `[0,1]`：新m=1，p=`[1/2,1]`，alpha=0；l=`1.5`，a=`0.5*1+1*2=2.5`。

第二块 `[2,1]`：新m=2，p=`[1,1/2]`，alpha=`2^(1-2)=1/2`；l=`1.5/2+1.5=2.25`；a=`2.5/2+1*4+0.5*8=9.25`。输出 `9.25/2.25=37/9`。

一次算全行，减去全局最大2，权重 `[1/4,1/2,1,1/2]`，总和2.25，分子 `.25+1+4+4=9.25`，同样得到37/9。在线状态不需要保存四个权重；真实大矩阵按块重复相同归并。

### 7. 因果mask为什么分两段

inner55–61行：stage1取 `[0,start_m*BLOCK_M)`，stage2只取当前query块对应的key范围。当前块之前的key对本块所有query都合法，不必逐元素三角mask；当前对角块才使用73–76行：

```python
mask = offs_m[:, None] >= (start_n + offs_n[None, :])
qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
m_ij = tl.maximum(m_i, tl.max(qk, 1))
qk -= m_ij[:, None]
```

例如query位置4，当前key块4..7，则仅key4合法；query7可看4..7。上游这里用有限大负数 `-1e6`，不是字面 `-inf`；不能把它写成支持任意极端输入、全mask行的普遍证明。教程host/test的shape、dtype与适用配置也属于合同，纸笔小矩阵不是直接可启动配置。

### 8. 自测与边界

- 新块最大值不变时alpha是多少？1，历史状态无需重标。
- 为什么不能只修正l不修正acc？分子分母不在同一基准，比值错误。
- `M`的行状态能代替KV cache吗？不能；它是本次Attention归约统计量，K/V是后续token要访问的投影数据。
- 小算例数学相等是否证明真实kernel完全一致？没有，实际dtype转换、目标指令、舍入及mask合同须按8.2验证。

在线Attention属于2019之后的算法/实现路线。本部分直接以固定版官方教程为实现依据；基础论文§3的tile语义有助于读shape，不能当作本部分算法来源或性能依据。

**换一个归约轴。** Attention 沿 key 维把多个 K/V 块合并成一个输出；Norm 则对每一行的 hidden 维做归约，而且通常要读同一行多次（均值、方差、仿射）。下一部分用官方 LayerNorm kernel 解释三段循环，再明确推导 RMSNorm 的不同合同。

<a id="sec-9-3"></a>
## 9.3 一行 hidden 的 LayerNorm 与 RMSNorm

hidden 维是每个 token 表示中各个分量所在的一轴。以一行 `[1,3]` 为例，LayerNorm 会先围绕均值 2 做中心化；RMSNorm 的核心尺度来自平方的均值，仍直接对原来的 1 和 3 应用尺度。

固定 SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`，真实带读 [05-layer-norm.py](../../triton/python/tutorials/05-layer-norm.py) forward。RMSNorm 另给明确教学改写，不冒充本教程已有实现。9.5 节给出小矩阵 LayerNorm/RMSNorm 的实际运行。

### 1. 哪个维度被归一化

hidden张量 `[batch,sequence,D]` 可在存储合同允许时看成 `[rows,D]`，其中 `rows=batch*sequence`。LayerNorm对每一行的D个特征计算均值/方差，不混合不同token。参数weight/bias通常各有D个元素，由不同行共享。

公式为 `mean=sum(x)/D`、`var=sum((x-mean)^2)/D`、`y=(x-mean)/sqrt(var+eps)*w+b`。分母是D，不是样本统计中的D−1；eps是数值稳定参数，不是任意忽略的装饰。

例：一行x=`[1,3]`，w=`[2,1]`，b=`[0,1]`，为纸笔简洁暂设eps=0。mean=2，var=1，归一化值 `[-1,1]`，输出 `[-2,2]`。真实实现一般要求适当正eps，应另测常量输入。

### 2. 指针从host参数变成行起点

官方源码47–64行 ` _layer_norm_fwd_fused`，摘录参数和前四条主体，省略注释：

```python
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd,
                          stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
```

`X/Y` 来自host的输入/输出张量指针；`W/B` 是同一行所有特征的仿射参数；`Mean/Rstd` 是host分配的每行统计量缓冲，后向需要。`N`是行列数，`stride`是相邻行起点跨越的元素数；`BLOCK_SIZE`是一次处理的特征块长。

设shape `[2,3]`、连续行stride=3，row1则X移至元素3，后面的局部 `cols=0,1,2` 对应全局3、4、5。`W/B`不加row偏移，因为每行用同一套权重。

该kernel对X/Y共用stride，且列指针用 `X+cols`，因此不能称支持任意二维转置视图；只有合同匹配的行/列布局才能直接调用。类型化指针的加法单位为元素，不是字节。

### 3. 第一段循环：按槽累加后再归约

65–72行，真实摘录：

```python
mean = 0
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _mean += a
mean = tl.sum(_mean, axis=0) / N
```

`_mean`是BLOCK_SIZE长的fp32逻辑向量，`mean`最终为标量IR值；它们不是同一种shape。每轮 `cols` 从off出发，load有效列，无效补0，再加到相同槽位。循环结束才对槽求和，因此允许N大于block。

教学例N=5、BLOCK_SIZE=4、行 `[1,2,3,4,5]`：第一轮 `_mean=[1,2,3,4]`；第二轮读 `[5,0,0,0]`，得到 `[6,2,3,4]`；sum=15，除N=5，mean=3。不能除BLOCK_SIZE=4，也不能除补齐长度8。

这些fp32累加减少低精度累加误差，但不是无限精度保证。前端编译时这里只构造循环、load、add、reduce操作；具体数值在设备执行时产生。

### 4. 第二段循环：mask必须在减均值后再处理

74–84行：

```python
_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    x = tl.where(cols < N, x - mean, 0.)
    _var += x * x
var = tl.sum(_var, axis=0) / N
rstd = 1 / tl.sqrt(var + eps)
tl.store(Mean + row, mean)
tl.store(Rstd + row, rstd)
```

只在load时补0还不够：若后面所有槽都减mean，尾块无效槽会变成 `-mean`，平方后污染方差。`tl.where` 明确让无效位置的**中心化结果**为0。

沿N=5例：第一轮中心化 `[-2,-1,0,1]`，平方 `[4,1,0,1]`；第二轮 `[2,0,0,0]`，平方 `[4,0,0,0]`；累计和10，var=2。若忘where，第二轮三个无效槽会各贡献9，方差错误为37/5。

`rstd`是逆标准差，每行一个值；两个store将统计量写到本行位置。它们不是把每个特征分别归一化后的数组。

### 5. 第三段循环：仿射输出

86–95行：

```python
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    tl.store(Y + cols, y, mask=mask)
```

`w/b/x/y`均是BLOCK_SIZE长向量，标量mean/rstd广播。W/B无效位置没有other，因此不能给它们编造尾部值；store只保留有效位置，正确性论证只依赖这些有效值。

三个源码循环意味着同一行数据在语义上被多次读取。称“fused”是因为归约、归一化、仿射写在一个kernel中，不能翻译为“每个输入元素必定只从显存读一次”；实际缓存与编译优化另论。

本教程还包含后向及梯度部分，本部分闭环只到forward输出，不把保存Mean/Rstd误认为完整训练过程已解释。

### 6. RMSNorm不能由改名字得到

RMSNorm的常用合同为 `r=sqrt(sum(x*x)/N+eps)`，`y=x/r*w`。它没有减均值，通常没有LayerNorm的bias项；具体模型是否加偏置、是否融合残差要按模型代码确认。

以下为**教学 Triton 改写**，不是上述官方文件的摘录，适用单块能容纳一行、连续列的简化合同；9.5 节给出实际运行：

```python
@triton.jit
def rms_teaching(X, Y, W, N: tl.constexpr, B: tl.constexpr,
                 EPS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    w = tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, x * inv * w, mask)
```

host负责分配Y，检查X连续、W长度N、B为覆盖N的受支持块长，并以grid `(rows,)` 传入指针；import同4.1。这里row步长固定N，未声称支持教程LayerNorm的可配置stride，也未支持任意超大hidden维。

`x*x`尾部补零天然不污染平方和，因此无需LayerNorm那个减均值后的where。N=2、x=`[1,3]`、w=`[2,1]`、eps=0时，均方5、inv=`1/sqrt5`，输出 `[2/sqrt5,3/sqrt5]`，明显不是LayerNorm的 `[-2,1]`（无bias版本）。

### 7. 残差融合仍须明确旧值和输出

若想先 `z=x+residual` 再RMSNorm，归约应计算z的平方，不是x平方加residual平方；两者少交叉项。模型可能还需要保留z作为后续残差分支的输出，不能只写归一化Y而丢掉z。

若原地覆盖X，须证明后续程序不再读取旧X，且每个program写入范围不交叠；如果模型有额外消费者，节省一次临时分配不能替代alias分析。是否融合值得做要包含所有新增输出、dtype转换和资源占用的测量。

### 8. 自测

1. LayerNorm输入 `[7,7]`，eps>0，w全1/b全0，输出？全0；方差0但分母非0。
2. RMSNorm同输入输出是否全0？不是，约为每项1，受eps影响。
3. N=5/B=4为什么均值除5？只五个真实特征，padding不是数据。
4. 一次kernel是否意味着只有一次load？不是，本段源码有三轮读取逻辑。

2019论文§3.2.2可帮助理解标量/向量广播；本部分LLM Norm合同与当前Python实现单独核对，不据旧论文声称模型收益。

**从单层算子走到请求。** 前面两部分都假设当前序列的 K/V 已经在手边。真实生成时，历史 token 的 K/V 要跨步骤保存和索引；prefill 与 decode 的输入量也不同。下一部分从官方 Attention 的连续索引出发，再用手算 page table 解释分页 KV，最后明确 kernel 测试与请求级性能的边界。

<a id="sec-9-4"></a>
## 9.4 从连续 K/V 索引到请求中的分页

KV cache 存的是先前 token 产生的 K/V。分页先把逻辑 token 位置分成「第几页、页内第几个」，再查页表得到物理存储位置。例如每页 4 个 token 时，位置 6 对应逻辑页 1、页内位置 2。

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。本部分真实源码基线仍是[06-fused-attention.py](../../triton/python/tutorials/06-fused-attention.py)的连续descriptor路径；分页和 decode 专用接口是**教学推导**，不是该教程已经实现的 paged serving kernel。本部分未运行 GPU/模型。

### 1. Prefill和decode先用输入量区分

设一个请求已有prompt四个token，head维D=2。prefill可同时产生四行Q、四行K、四行V，因果Attention每行只看自己及以前位置。随后生成第五个token时，decode只为新增token计算一行Q/K/V，再用该行Q看历史加当前的五行K/V。

KV cache缓存的是各层各KV head的K/V投影，不是token ID、softmax概率或最终hidden。一个普通请求每生成一个token，逻辑长度通常增加1；具体缓存分配、量化、滑动窗口、prefix共享由引擎合同决定。

| 阶段 | 本次query数 | 可看的key数 | 本部分教学例 |
|---|---:|---:|---|
| prefill | prompt长度 | 至少覆盖prompt有效因果前缀 | 4行Q，4行K/V |
| 一步decode | 通常每活跃请求1行 | 历史长度+新token | 1行Q，5行K/V |

数学公式相同，不意味着相同tile/grid是最佳选择。decode的Q行很少，长K/V读取、并行归约和调度开销更突出；是否受带宽限制必须结合真实资源与测量，不靠阶段名字下结论。

### 2. 官方教程实际上怎样访问连续K/V

官方inner63–71、103–107行，非连续摘录：

```python
offset_y = off_hz * N_CTX
offsetk_y = offset_y + lo
offsetv_y = offset_y + lo
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = desc_k.load([offsetk_y, 0]).T
    qk = tl.dot(q, k)
    offsetk_y += BLOCK_N
    offsetv_y += BLOCK_N
```

注释为本部分增加，其他语句来自上游。`off_hz` 是batch/head扁平编号，`N_CTX`为每个序列相同长度。`lo/hi`来自stage范围；每轮连续前进BLOCK_N行，V在同文件96–101行按同样token块读取。

例如H=2、batch1/head0，则 `off_hz=2`；N_CTX=5，head起始扁平行10。key逻辑位置3在行13、feature1的元素偏移为 `13*D+1=27`（D=2）。这是descriptor shape/stride合同推得的连续偏移，尚未经过page table。

这个kernel接口没有请求block table、每请求长度列表、append slot mapping或页分配器参数。不能仅把教程Q行数改成1，就宣布已实现可用于引擎的分页decode；教程的shape、stage与descriptor配置也要相应检查。

### 3. 先推导连续cache的append

教学cache布局为 `[request,kv_head,max_tokens,D]`，连续内层D。某请求r、head h、位置t、维d的元素偏移：

```text
(((r * Hkv + h) * max_tokens + t) * D + d)
```

例r=1、Hkv=2、h=0、max_tokens=8、t=4、D=2、d=1：`((2*8+4)*2+1)=41`。把新K/V写进位置4后，本次decode有效key长度为5，而非预分配容量8。

读取容量尾部没有语义依据，mask应以请求有效长度判断。指针还必须落在已分配容量之内，不能靠 `other=0` 掩盖错误地址构造。追加K/V和读取Attention若在同stream有正确先后关系，可利用stream顺序；跨stream要用明确事件/依赖，不能依赖CPU调用看似相邻。

### 4. 分页多出哪一次查表

以下全是教学布局，不是对上游06实现的描述。每页P=4个token，物理cache布局 `[physical_page,Hkv,P,D]`。请求r有逻辑页表 `table[r,logical_page]`。

访问逻辑token t：

```text
logical_page = t // P
slot = t % P
physical_page = table[r, logical_page]
element_offset = (((physical_page * Hkv + h) * P + slot) * D + d)
```

r=0的页表 `[3,1]`，Hkv=2、h=1、D=2，访问t=5、d=0：逻辑页1、槽1、物理页1，偏移 `((1*2+1)*4+1)*2=26`。若直接按token5线性读，会跳过页表映射；即使偶然落在合法分配内，读到的也可能属于别的请求。

注意分页单位、head顺序、K/V是否分开存、feature打包顺序均可不同。上述公式只对这里声明的物理shape成立，不是所有引擎通用地址公式。

### 5. 用短代码走完一次paged K读取

下面是**教学 Triton 片段**，展示把逻辑 token 块映射为 K 值，本部分未执行。它不是完整 Attention kernel，没有在线 softmax 和输出写回，也不声称支持全部目标/解释器：

```python
@triton.jit
def gather_k_teaching(K, Table, Out, length,
                     request: tl.constexpr, head: tl.constexpr,
                     MAX_PAGES: tl.constexpr, H_KV: tl.constexpr,
                     PAGE: tl.constexpr, D: tl.constexpr,
                     BT: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    valid_t = t < length
    page = tl.load(Table + request * MAX_PAGES + t // PAGE,
                   valid_t, other=0)
    offset = (((page[:, None] * H_KV + head) * PAGE
               + (t % PAGE)[:, None]) * D + d[None, :])
    k = tl.load(K + offset,
                valid_t[:, None] & (d[None, :] < D), other=0)
    tl.store(Out + t[:, None] * D + d[None, :], k,
             valid_t[:, None] & (d[None, :] < D))
```

host要保证0≤length≤MAX_PAGES*PAGE、head有效、每个已使用页表项有效、Out容量足够，并用grid `cdiv(length,BT)`。`Table`存整数物理页号，不是K指针列表；Out是连续教学输出，方便核对gather结果。BT/BD是受支持的逻辑块长，BD≥D。

`t/d`分别是token和feature向量；页号每token一个，`[:,None]`广播到所有feature。地址由页、head、页内槽和feature四项组成。无效token的页表load也受mask，否则可能在第二次K load之前已经越界。

这个示例有“先读页表值，再据此算K地址”的依赖。解释器对间接模式的覆盖应按8.2的已知限制判断，不能把解释器不支持归因于算法数学错误。

真实高性能paged Attention通常不先gather完整K到显存Out，而是将查表后的K/V块直接交给QK和在线softmax/PV。此处写Out只用于把索引合同闭合，不能据此宣称减少了一次显存中间写。

### 6. 新token落在哪一页

同一页表 `[3,1]`，当前长度L=5，下一新token位置t=5，写物理页1的槽1；若L=8，位置8属于逻辑页2，host必须先准备合法第三个页表项。不能让kernel看到未初始化页号再希望mask修复，因为新token本身是有效位置。

若多个请求共享prefix页，追加时要维护只读共享/写时复制规则；本部分没有实现allocator、引用计数或回收。同步正确也不能防止两个请求因错误页表写进同一有效槽。

在一个请求取消后复用页，旧GPU访问必须已经结束，才能给新请求覆盖；这与7.3共享scratch复用的“旧消费者是否完成”问题相似，但这里是跨kernel/请求的全局存储生命周期，不能只靠CTA barrier解决。

### 7. GQA：Q head与KV head不总相等

教学常见分组合同：Hq=4、Hkv=2，每2个Q head共享一个KV head，则 `kv_h=q_h//2`，q0/q1用kv0，q2/q3用kv1。必须由模型/引擎约定这个分组；不能把原教程的相同H直接当作GQA已实现。

Q块索引使用q_h，K/V地址使用kv_h，输出又按q_h写回。误把head索引完全复用会读错cache，即使每个内存访问都没有越界。

### 8. 从kernel回到请求的验收边界

一次请求还包含tokenization、QKV投影、Norm、RoPE、Attention、输出投影、MLP、采样、调度/通信等。prefill可能按chunk处理，decode可能batch大小不断变化，内存页和活跃请求也会变化。

局部kernel测试至少检查：页内、跨页、部分最后页、不同请求页表、不同有效长度、GQA映射、追加后读取一致性。服务层还须检查请求隔离、页回收和输出token是否保持合同。测kernel时间不能替代首token延迟、逐token延迟或吞吐测量。

自测：P=4、table=`[7,2]`，t=6在哪？逻辑页1、物理页2、槽2。长度6时t=6可读吗？不行，有效位置为0..5。哪个同步能修复错误页号？没有，同步只建立访问顺序，不能修复地址语义。

2019论文没有本部分分页KV请求系统；以上分层把真实连续教程与教学推导明确分开，不捏造上游paged kernel、性能数字或整模型收益。

## 9.5 实际运行：小矩阵 LayerNorm 与 RMSNorm

`actual_layernorm.py` 用单块教学 kernel 跑 9.3 的数学合同：一行作为一个 program，`BLOCK=next_power_of_2(D)`，LayerNorm 保留“求和→中心化平方和→仿射”三遍结构；RMSNorm 使用 `rsqrt(mean(x²)+eps)`。环境同 4.3。

**命令。**

```bash
python examples/actual_layernorm.py
```

**输出。**

```text
layernorm_output [[-2.4494714736938477, 1.0, -0.3876321315765381],
                  [-2.4494853019714355, 1.0, -0.3876286745071411]]
layernorm_reference [[-2.4494712352752686, 1.0, -0.38763219118118286],
                     [-2.4494853019714355, 1.0, -0.3876286745071411]]
layernorm_max_abs_diff 2.384185791015625e-07
rmsnorm_output [[0.9258190393447876, 0.9258190393447876, 0.6943643093109131],
                [0.9258198142051697, 0.9258198142051697, 0.6943648457527161]]
rmsnorm_reference [[0.9258190393447876, 0.9258190393447876, 0.6943643093109131],
                   [0.9258198142051697, 0.9258198142051697, 0.6943648457527161]]
rmsnorm_max_abs_diff 0.0
```

**怎样读。**

- 输入是两行 `[1,2,3]` 和 `[2,4,6]`，`w=[2,1,0.5]`、`b=[0,1,-1]`、`eps=1e-5`。LayerNorm 与 PyTorch 的 `mean/var` 参考最大差约 `2.4e-7`。
- RMSNorm 不减均值，所以同一输入的输出与 LayerNorm 不同；这个例子里 RMSNorm 与 PyTorch 参考逐位一致。两行 RMSNorm 输出之间仍有极小差异，是因为第二行是第一行的两倍，均方根求和的舍入路径不同。
- 这个 LayerNorm 是单块教学实现，不是官方教程完整的 backward/多轮循环版本；它验证的是 9.3 的数学合同，不证明官方 kernel 的所有分支。

**边界。** 本节没有运行 Attention、KV cache 或完整模型；这些部分仍是静态带读与教学推导。

## 第 9 章小结

| 算子 | 输入/输出 | 归约轴 | 关键中间状态 | 不能混淆的边界 |
|---|---|---|---|---|
| Attention | Q `[S_q,D]`、K/V `[S_k,D]` → O `[S_q,D]` | 每个 query 沿 key | `m_i`、`l_i`、`acc` | 在线 softmax 数学不等于真实 dtype/指令逐位一致 |
| LayerNorm | 一行 `[D]` → 一行 `[D]` | hidden 维 | 均值、方差、`rstd` | 三个循环不等于只读一次显存 |
| RMSNorm（教学） | 一行 `[D]` → 一行 `[D]` | hidden 维的平方和 | `rsqrt(mean(x²)+eps)` | 不是 LayerNorm 改名字；没有减均值 |
| KV Cache | 历史 K/V + 新 token | 历史 token 维 | 页表、有效长度、KV head 映射 | kernel 时间不等于请求延迟 |

### 综合检查

1. 在线 Attention 为什么必须同时修正 `acc` 和 `l_i`？因为新块的最大值可能改变，分子和分母都要重标到同一基准。
2. Softmax 尾块补 `-inf`、GEMM 尾块补 0、Norm 尾块补 0 后又 `where`，为什么不能统一？补齐值要由后续运算的中性值决定。
3. LayerNorm 的方差为什么要对中心化后的无效位置再置 0？只在 load 补 0 不够，减均值后无效槽会变成 `-mean`，平方后污染方差。
4. RMSNorm 和 LayerNorm 的核心合同差异是什么？RMSNorm 不减均值、通常无 bias，按均方根归一化。
5. 分页 KV 的 page table 存的是 K/V 指针列表吗？通常存物理页号；地址还要由页号、head、页内槽和 feature 计算。
6. 一个 paged gather 教学片段能代表完整 serving 吗？不能；它只闭合索引合同，没有 allocator、请求隔离、调度和端到端计时。
