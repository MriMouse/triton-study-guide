# 第 6 章 两个经典算子：Softmax 与 GEMM

[返回总目录](../README.md) · [上一章：前端与 IR](05-frontend-ir.md) · [下一章：布局、Pass 与内存同步](07-layout-passes-memory.md)

第 4 章的向量加法每个输出只依赖一个输入位置。这一章换成两种“有归约”的计算：

- **Softmax**：一行内所有元素共同决定每个输出，需要 max、sum 和广播；
- **GEMM**：一个输出元素沿 K 维累加，需要二维地址、K 循环、数据复用和分组调度。

它们放在同一章，不是因为算法相似，而是因为它们正好展示归约的两个方向：Softmax 沿“行内列”归约，GEMM 沿“内积 K”归约。把两者对照，读者更容易看出“归约轴”“补齐值”和“复用来源”是三件不同的事。

## 本章地图

| 维度 | Softmax（6.1） | GEMM（6.2） |
|---|---|---|
| 输出依赖 | 同一行的所有列 | 沿 K 维的所有 A/B 元素 |
| 关键运算 | max、exp、sum、除法 | 逐块乘加、`tl.dot` 累加 |
| 无效位置填充 | `-inf`（max 的中性值，exp 后为 0） | `0`（乘加的中性值） |
| 数据复用 | 一行通常只读一次 | A/B 块跨多个输出块复用 |
| 主要风险 | 数值溢出、mask 污染分母 | 尾块地址、精度、配置与写回 mask |
| host 侧重点 | 行数→grid、warp/共享存储→persistent 行数 | 块大小、分组顺序、autotune 配置 |

6.3 节给出这两个教学 kernel 的实际运行与数值检查。

**读法：** 先手算 6.1 的 2×3 小矩阵，再手算 6.2 的 2×2 输出。算完后再回到源码，看每个变量对应手算表里的哪一列。

<a id="sec-6-1"></a>
## 6.1 从一行三个数到 persistent softmax

这次不再是每项互不相关的加法：softmax 的一个输出依赖同一行所有输入。我们先算清楚一行，再把整行放入一个逻辑块，最后读官方教程为什么让一个 program 循环处理多行。

源码固定提交 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`，主文件为 [02-fused-softmax.py](../../triton/python/tutorials/02-fused-softmax.py)。代码许可见 [CODE-NOTICE](../CODE-NOTICE.md)。以下先做静态源码带读；6.3 节给出教学 kernel 的实际运行，但没有性能结论。

### 1. 先定义数学目标和输入合同

输入是一张2行3列的连续 FP32 矩阵，按行计算 softmax：

```text
X = [[1, 2, 3],
     [2, 2, 2]]
```

第 r 行、第 c 列输出为 `exp(X[r,c]) / sum_j exp(X[r,j])`。每一行各有自己的分母，不能把两行六个数一起求和。

为了避免大正数指数溢出，使用稳定形式，令 `m = max_j X[r,j]`：

```text
Y[r,c] = exp(X[r,c] - m) / sum_j exp(X[r,j] - m)
```

为什么减最大值不改变数学答案？分子分母同时乘了 `exp(-m)`，公共因子约掉。每个指数的输入都小于等于0，至少一个为0，因此有限输入情况下分母至少包含 `exp(0)=1`。

本章限定行数与列数都为正、元素有限、同设备、列连续，输入与输出不重叠。用 `input_row_stride` 允许行间距大于列数，但相邻列仍按一步一个元素读取。NaN、整行负无穷、正无穷和空行没有被这个短实现定义为通用合同；“稳定”并不自动覆盖所有特殊值。

### 2. 手算每个中间值

第一行 max=3，所以：

| 中间量 | 近似值或精确值 |
|---|---|
| row | `[1,2,3]` |
| row - max | `[-2,-1,0]` |
| numerator | `[0.135335,0.367879,1]` |
| denominator | `1.503214...` |
| output | `[0.090031,0.244728,0.665241]` |

第二行 max=2，减后 `[0,0,0]`，指数 `[1,1,1]`，分母3，输出每项约0.333333。上表来自实数数学的纸笔近似，不是 Triton 的 exp 实测；设备近似指数和浮点加法会造成细小差异。

把第一行全部加1000后，稳定形式仍减去1003，得到同样的 `[-2,-1,0]`。直接计算 exp(1003) 则会遇到有限浮点范围问题。稳定形式解决了这个溢出来源，没有让浮点运算变成精确实数。

### 3. 用一个 program 处理一行，先看完整教学 kernel

下列是根据官方第84–109行**教学改写**的简化版本：去掉跨行 persistent 循环，使用一行一个 program；显式转 FP32 便于固定精度口径。未修改上游文件，未编译验证。

```python
@triton.jit
def row_softmax(out_ptr, in_ptr, in_stride, out_stride,
                n_cols, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    row = tl.load(in_ptr + row_idx * in_stride + cols,
                  mask=mask, other=-float('inf')).to(tl.float32)
    centered = row - tl.max(row, axis=0)
    numerator = tl.exp(centered)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(out_ptr + row_idx * out_stride + cols, result, mask=mask)
```

host 端可以按以下**教学 wrapper**连接，假设 torch/triton/tl 与 DEVICE 已按 4.1 导入准备；这不是上游原样代码，也不是本章运行命令：

```python
def row_softmax_host(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    row_softmax[(n_rows,)](y, x, x.stride(0), y.stride(0),
                           n_cols, BLOCK=block)
    return y
```

这是理解参数流所需的完整小闭环：x 来自调用者，y 由 host 分配，shape 拆出2与3，block=4，grid=(2,)。输入/输出 stride 均为3，以元素计。y/x 分别对应 out_ptr/in_ptr，不能因数学写法习惯先写X就把调用顺序颠倒。

### 4. 第一组语句算的是索引与地址

`row_idx = tl.program_id(0)` 让 program0处理第0行，program1处理第1行；它是每个逻辑 program 内一致的标量，不是列索引。

`cols = tl.arange(0, BLOCK)` 得到 `[0,1,2,3]`，shape `(4,)`。常量 BLOCK=4 是不小于3的最小2的幂；多出来的索引3必须屏蔽。

`mask = cols < n_cols` 得到 `[true,true,true,false]`，shape仍为 `(4,)`。它没有把 cols 截成三项。对 program1，输入元素偏移为：

```text
row_idx * in_stride = 1*3 = 3
3 + cols             = [3,4,5,6]
mask                 = [T,T,T,F]
```

前3个偏移对应第二行的三个2；最后地址不允许读取，由 load 的 mask 阻止。计算出地址表达式不等于对每个地址进行访问。

输入指针相加按元素计步：FP32 的偏移3相当于12字节。这个 kernel 不带列stride，因此不能原样用于转置后列不连续的视图；x.stride(0) 正确不代表所有维度都支持任意步长。

### 5. 为什么无效列必须补负无穷

load 后第一行在逻辑上是 `[1,2,3,-inf]`。转 FP32 后 shape 不变。max 沿唯一的轴0，把四项归约成标量3，centered 变成 `[-2,-1,0,-inf]`。

exp 后是 `[exp(-2),exp(-1),1,0]`。无效列既不改变最大值，又不改变分母。最后 store 再用 mask 丢弃该列的输出，不会写到下一行。

只在 store 处加 mask 不够：若无效列的 load 未受保护，已经发生越界读取；若 load 受保护但补错值，虽然写入安全，归约的分母也可能错。

例如所有有效值是 `[-3,-2,-1]`，若补0，max就变成0，补入项 exp(0)=1 还加入分母。即使数学上的减常数本身不改变有效项比例，多出来的指数项仍会降低三个有效输出的总和。负无穷是 max 的中性填充值，而且指数后变成 sum 的中性值0。

全行有效数据若也是负无穷，`-inf - (-inf)` 会产生非有限结果；这就是前面的有限输入合同不能省略的原因。输入本来含NaN时，还需专门规定传播策略并核对实现。

**确认题：** n_cols=4 时还需要哪列补齐？没有，mask 全真。但原语句仍然正确，只是没有无效列。

### 6. 归约改变 shape，广播再扩回来

对一维四元素块，`tl.max(row, axis=0)` 产生标量，逻辑 shape 为 `()`。`row - 标量` 通过广播把同一个最大值用于四个位置，因此 centered 仍是 `(4,)`。

`tl.sum(numerator, axis=0)` 再产生标量分母。`numerator / denominator` 按相同规则广播，结果恢复四个逻辑输出。归约既不是只取第0项，也不是按列独立计算四个和；axis=0 指本块唯一的维度。

归约内部要让不同硬件线程的部分结果结合，可能涉及跨线程交换；源码的一行不表示只用一条机器加法。把“逻辑上归约成一个标量”和“物理上怎样协作”分开，7.1/7.3再读具体布局与同步。

[language/standard.py](../../triton/python/triton/language/standard.py) 第354–358行，sum 的真实主体：

```python
def sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)
    input = input.to(out_dtype)
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)
```

sum 先选归约类型，再转换输入、调用 reduce。默认类型选择辅助函数位于同文件第337–348行：显式给dtype时使用它；小整数提升到32位，其余保留输入dtype。**本版本不能笼统说sum自动把小浮点提升到FP32。** max 第218–224行则会对无索引的低位宽浮点输入提升到FP32。教学版本额外显式转 FP32，所以减法、指数、求和的这一条路径容易说明；官方版本没有这条显式转换，不应伪称原样相同。

这仍不能保证任意长度求和精确：浮点加法不满足实数结合律，归约树与逐项CPU求和可能有不同舍入。官方第102行还明确提醒指数是快速近似；验证应规定输入分布、输出dtype与容差，不能用“稳定”代替正确性测试。

### 7. 官方版本多出来的循环究竟做什么

回到 [02-fused-softmax.py](../../triton/python/tutorials/02-fused-softmax.py) 第84–99行，保留执行语句、删除注释：

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
```

row_start 仍是 program编号。row_step 是 grid 第0维一共启动的program数P。每个 program 处理行号 `pid, pid+P, pid+2P, ...`，小于 n_rows 才继续。

例如 n_rows=5、P=2：program0处理0、2、4，program1处理1、3。不是每个program都从0重新扫描全部行，因此没有重复写入。P可以小于行数；这就是该教程所谓 persistent programs：program活着跨越多行工作，而非启动一次后永不结束。

`tl.range` 是 Triton 的循环接口，num_stages 给循环流水安排提供编译信息；它不是 Python 在 host 把五行依次 load 到CPU。具体是否形成某种硬件流水、资源怎样变化，要看实际编译结果，不能只看到参数4就声称四条GPU流。

同一函数第101–109行的计算与写回：

```python
row_minus_max = row - tl.max(row, axis=0)
numerator = tl.exp(row_minus_max)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
output_row_start_ptr = output_ptr + row_idx * output_row_stride
output_ptrs = output_row_start_ptr + col_offsets
tl.store(output_ptrs, softmax_output, mask=mask)
```

每轮只对当前 row_idx 做一次完整归一化。输入与输出行stride分开，让输出分配方式不必跟输入行间距相同。循环下一轮重新形成行基址，行内的算术结构仍与简化 kernel 一样。

### 8. host 为什么先 warmup，而不是马上用行数做grid

官方 wrapper 在第124–147行先选参数、分配输出、准备资源信息。删注释后摘录：

```python
n_rows, n_cols = x.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)
num_warps = 8
num_stages = 4 if SIZE_SMEM > 200000 else 2
y = torch.empty_like(x)
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                               num_stages=num_stages, num_warps=num_warps, grid=(1, ))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared
```

BLOCK_SIZE 根据列数决定整行逻辑块大小；num_warps是编译选项；num_stages在这个kernel签名中也是constexpr参数，传给tl.range。这里8与200000都是此教程的选择，不是适用于全部设备、全部行长的经验定律。

4.2 读过 warmup：它交给 JIT.run 的 warmup=True，使实际发射分支跳过，返回编译产物。grid=(1,) 只是这次准备调用带的网格信息，不表示已经算完第一行，也不决定最终P。

`kernel._init_handles()` 触发装载并取得资源信息。compiler.py 第485–486行从驱动 load_binary 接收 n_regs 等字段，因此本版本 n_regs 并非这个 wrapper 在 host 根据B算出来的。metadata.shared 则记录编译产物的共享存储需求。

### 9. 把 occupancy 的算式按单位读懂

教程第115–119行先从设备属性中取得 NUM_SM、NUM_REGS、SIZE_SMEM、WARP_SIZE。下面是第166–174行的 NVIDIA 分支与共同尾部，省略HIP分支：

```python
else:
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
num_programs = min(num_programs, n_rows)
kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
```

这段是**分支摘录**，else对应前面的is_hip判断。NUM_REGS 是每个多处理器的寄存器资源量；`n_regs * WARP_SIZE * num_warps` 估计一个program的寄存器需求。因此整除得到寄存器预算允许同时驻留多少个program。

再用共享存储总量除以每program共享存储量，取两者最小值；乘 NUM_SM 得到全设备候选program数量，最后不超过行数。这里变量 occupancy 表示教程估计的每多处理器驻留program数，不是通常以百分比描述的所有意义上的occupancy。

**仅作单位演示的假设数字：** NUM_REGS=65536、n_regs=32、WARP_SIZE=32、num_warps=8，寄存器限制为8；SIZE_SMEM=65536、size_smem=16384，共享存储限制为4，因此估计每SM为4。若NUM_SM=2、n_rows=5，则候选8，最终P=5。数字不是本机设备信息，也不是本章实际内核资源。

教程没有在此完整复现所有硬件分配粒度和驻留限制，且直接做 SIZE_SMEM//size_smem；不要把这个简短公式复制成任何编译产物都稳健的通用调度器。扩展到其他内核时，零共享存储、资源分配粒度及目标限制都需重新处理和验证。

HIP 分支位于第148–165行：根据CDNA架构处理通用寄存器池，并结合最大线程/波数估计，再除num_warps。它没有直接照搬NVIDIA算式；这正说明资源模型必须随目标核对。

### 10. 最后一行为什么直接调用 kernel

warmup 返回的是 CompiledKernel，所以最后 `kernel[(P,1,1)](...)` 走的是另一个方括号接口，不再是 JITFunction 的 binder 入口。

[compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) 第511–522行：

```python
def __getitem__(self, grid):
    self._init_handles()
    def runner(*args, stream=None):
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        launch_metadata = self.launch_metadata(grid, stream, *args)
        self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                 knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
    return runner
```

self 是已编译内核对象，grid固定三维。runner获取流并调用launcher；因此教程必须把实参按准备时相同的顺序传齐，并显式带上两个constexpr位置的值，让launcher按产物签名处理。这条快捷路径不会替你重新编译不匹配的B。

对2×3教学矩阵，假设资源允许P=2，则每program处理一行；若人为令P=1，则一个program依次处理两行，数学目标仍相同。P影响调度与复用，不改变每行分母的定义。

### 11. 融合省掉什么，不能由此宣称什么

官方第39–59行 naive_softmax 分别求max、相减、exp、sum、相除，创建多个中间张量；融合 kernel 可以在一行的一次处理中保留中间值，只把最终输出写回。这是要检验的访存收益来源。

但“逻辑中间值不写出”不保证没有寄存器溢出、没有共享存储或全部行长都快。大行带来更多逻辑值、资源压力和较少驻留program，实际结果依赖编译与设备。官方教程里的理论流量估算及速度图不是本章实验；其中流量文字也不能把“元素数”直接当“字节数”，必须乘dtype字节量。

**自测与答案：** 五行、P=2时最后一行由谁处理？program0的第三轮。BLOCK=4、n_cols=3时无效列的exp为何为0？有限最大值下 `exp(-inf)=0`。把mask只放在store是否安全？不安全，load越界和错误归约都可能先发生。是否能从num_warps=8推断一行只有8个元素？不能，warp数与逻辑列数属于不同层次。

> 回到论文：§3.2.1 Tile semantics（PDF第3页）与§4.2 Data-Flow（PDF第5页）帮助理解整行逻辑块和显式归约；见[定位表](../appendices/reference.md#sec-a-1)。本部分的 persistent 循环、现代 Python API 和资源查询来自当前教程，不宣称为 2019 论文同一实现。原论文矩阵乘实验也不能当softmax当前收益证据。

本章已闭合数学→索引/归约→host资源准备→persistent启动；6.3 节给出实际运行。

**第一部分交出的边界。** Softmax 处理的是“一个 program 负责一行”的一维块，归约轴是本行所有列。下一部分把这个想法扩展到二维：一个 program 负责 BM×BN 的输出块，沿 K 维循环加载 A/B 块并累加。归约不再是 `tl.sum` 一个轴，而是 `tl.dot` 的矩阵乘语义。

<a id="sec-6-2"></a>
## 6.2 矩阵乘法：二维地址、K 循环与分组调度

本章读 `C = A @ B` 怎样成为按块加载、沿K累加、最后写回的 kernel。先用能手算的矩阵追清每个地址，再看官方教程的分组顺序和 autotune；不从“使用 Tensor Core 就快”这样的结论开始。

源码固定为 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`，主要依据 [03-matrix-multiplication.py](../../triton/python/tutorials/03-matrix-multiplication.py)。摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。以下先做静态带读；6.3 节给出小块教学 GEMM 的实际运行。小块仅供算法教学，不冒充官方硬件配置或性能实测。

### 1. M、N、K 各自数的是什么

```text
A = [[1,2,3],       B = [[ 7, 8],
     [4,5,6]]            [ 9,10],
                         [11,12]]
```

A 的 shape 是 `(M,K)=(2,3)`；B 是 `(K,N)=(3,2)`；C 是 `(M,N)=(2,2)`。共同的K表示内积长度，必须相等。输出元素 `C[m,n] = sum_k A[m,k]*B[k,n]`，不是 A/B 同位置相乘。

```text
C[0,0] = 1*7 + 2*9 + 3*11 = 58
C[0,1] = 1*8 + 2*10 + 3*12 = 64
C[1,0] = 4*7 + 5*9 + 6*11 = 139
C[1,1] = 4*8 + 5*10 + 6*12 = 154
```

先约定输入连续、元素类型FP32、行优先存储，输入输出不重叠，维度都为正。它们在内存中的元素序列分别是 `[1,2,3,4,5,6]` 和 `[7,8,9,10,11,12]`。后面会解释官方wrapper的FP16输出；不要把两种精度口径混在一起。

### 2. stride 把二维坐标变成一维偏移

stride 是沿某个维度增加1时，地址前进的**元素数**。本例：

| 矩阵 | 行方向stride | 列方向stride | 坐标到偏移 |
|---|---|---|---|
| A，2×3 | stride_am=3 | stride_ak=1 | `m*3+k` |
| B，3×2 | stride_bk=2 | stride_bn=1 | `k*2+n` |
| C，2×2 | stride_cm=2 | stride_cn=1 | `m*2+n` |

例如 A[1,2] 的元素偏移是5，FP32时相对首地址20字节。B[2,1] 也是偏移5，但它属于另一份存储。相同整数偏移并不表示两个矩阵共用地址。

使用stride而不是写死 `m*K+k` 的意义，是让算法表达与实际存储步长分离；但能否支持任意视图，还要检查wrapper断言、mask及优化假设，不能看到六个stride就声称没有接口限制。

### 3. 一个 program 负责输出的一块

把每个program负责的输出块设为 BM行×BN列；沿K每次加载BK项。为手算暂取 BM=2、BN=2、BK=2。本例只需一个program覆盖全部2×2输出，沿K循环两轮。

这三个2是教学的逻辑块大小，**不是官方tl.dot可在任意后端直接运行的矩阵形状**。官方配置大得多，并受目标硬件及dot lowering约束。为了完整展示小矩阵算法，下面用逐元素乘法与sum表达小块乘积；稍后再换成官方tl.dot。

每轮取 A 的 BM×BK块和 B 的 BK×BN块，得到 BM×BN 的局部乘积，再加进累加器。循环结束才写C，所以重复读取/写入完整C不是算法必需步骤。

### 4. 完整教学 kernel：显式乘法与归约

下面是根据官方索引与K循环思路编写的**教学实现**，不属于上游摘录，未在本机编译验证。使用正维度与有效stride，输出FP32；目的为小例可手算，三维临时块会使它不适合作为高性能GEMM模板。

```python
@triton.jit
def teaching_gemm(A, B, C, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BK)):
        ks = kb * BK + kk
        ap = A + rows[:, None] * sam + ks[None, :] * sak
        bp = B + ks[:, None] * sbk + cols[None, :] * sbn
        av = tl.load(ap, (rows[:, None] < M) & (ks[None, :] < K), other=0).to(tl.float32)
        bv = tl.load(bp, (ks[:, None] < K) & (cols[None, :] < N), other=0).to(tl.float32)
        products = av[:, :, None] * bv[None, :, :]
        acc = acc + tl.sum(products, axis=1)
    cp = C + rows[:, None] * scm + cols[None, :] * scn
    tl.store(cp, acc, (rows[:, None] < M) & (cols[None, :] < N))
```

相应**教学host片段**假设a/b已创建在同一GPU、都是二维FP32连续张量；torch/triton/tl按4.1导入：

```python
M, K = a.shape
K_b, N = b.shape
assert K == K_b
c = torch.empty((M, N), device=a.device, dtype=torch.float32)
grid = (triton.cdiv(M, 2), triton.cdiv(N, 2))
teaching_gemm[grid](a, b, c, M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1), BM=2, BN=2, BK=2)
```

a/b/c三个host张量按顺序对应kernel的A/B/C指针参数；六个stride分别对应sam/sak/sbk/sbn/scm/scn。grid=(1,1)产生一个二维网格中的program。BM/BN/BK是编译常量；M/N/K是实际矩阵长度，尾块mask不能拿块大小替代长度。

### 5. None 插入一个长度1维度，广播构造二维地址

先看第0轮：rows=[0,1]，cols=[0,1]，kk=[0,1]，ks=[0,1]。它们都为一维块。`rows[:, None]` 把shape `(2,)` 变成 `(2,1)`；`ks[None,:]` 变成 `(1,2)`。

把两个形状相加时，长度1维度广播成对方长度，得到 `(2,2)`：

```text
rows[:,None]*sam = [[0],     ks[None,:]*sak = [[0,1]]
                   [3]]
相加偏移           [[0,1],
                   [3,4]]
```

因此 ap 指向 A 的值 `[[1,2],[4,5]]`。这个None不是“缺少指针”，也不是创建一列未知数据，而是让原索引在另一个轴上复用。

B的写法交换维度角色：ks沿行，cols沿列。

```text
ks[:,None]*sbk = [[0],       cols[None,:]*sbn = [[0,1]]
                 [2]]
相加偏移         [[0,1],
                 [2,3]]
```

所以bp加载 `[[7,8],[9,10]]`。ap/bp 的shape恰好都2×2只是本例巧合，一般分别是BM×BK与BK×BN；这两个块的第二维不能一概叫列N。

mask也使用同样广播规则。A检查m与k，B检查k与n。`&` 是逐元素与；不能用普通Python `and` 试图合并一整块条件。

**确认题：** 把A写成 `rows[None,:] + ks[:,None]` 而其它含义不改，会怎样？轴角色被交换，加载块与后续乘法使用的m/k对应错误；不能只因最终shape一样就认为等价。

### 6. 三维 products 怎样变回二维 acc

av是BM×BK，bv是BK×BN。`av[:,:,None]` 是BM×BK×1，`bv[None,:,:]` 是1×BK×BN；广播相乘后 products 是BM×BK×BN。

products[m,k,n]恰好等于av[m,k]*bv[k,n]。沿axis=1归约就是把K这一轴加掉，留下BM×BN。第0轮：

```text
输出(0,0)的两个乘积：[1*7, 2*9] = [7,18] → 25
输出(0,1)的两个乘积：[1*8, 2*10] = [8,20] → 28
输出(1,0)的两个乘积：[4*7, 5*9] = [28,45] → 73
输出(1,1)的两个乘积：[4*8, 5*10] = [32,50] → 82
```

acc开始全零，更新后 `[[25,28],[73,82]]`。`acc = acc + ...` 在源码层重新绑定局部名字；编译器用循环携带值表达它，不是违反SSA、让同一个IR结果同时拥有两个定义。5.2的iter_args/yield正是这种表达的阅读基础。

### 7. 第1轮：K尾部为什么填零

第1轮kb=1，ks=[2,3]，但K=3只允许2。A地址偏移是 `[[2,3],[5,6]]`，其中每行第二项的k无效；B偏移是 `[[4,5],[6,7]]`，第二行k无效。

这里 A 的偏移3在物理上碰巧是下一行首项，仍必须屏蔽：它不属于本次需要的A[0,3]。mask按**逻辑坐标**判断，不是只问地址是否落在整份分配内部。

加载得到：

```text
av = [[3,0],       bv = [[11,12],
      [6,0]]             [ 0, 0]]
本轮乘积和 = [[33,36],
             [66,72]]
acc最终     = [[58,64],
             [139,154]]
```

填零是乘加累积的中性选择：尾部虚构项贡献0。softmax用负无穷是因为要经过max和exp；GEMM用零是因为要经过乘法和sum。不能从“mask”这个共同名字推断所有算子都用同一个other值。

写回cp根据rows/cols和C的stride形成 `[[0,1],[2,3]]`；只在m<M且n<N处store。输入读取mask与输出写入mask解决不同问题，一个正确不能替代另一个。

### 8. 官方版本把小块乘法交给 tl.dot

现在回到 [03-matrix-multiplication.py](../../triton/python/tutorials/03-matrix-multiplication.py) 第291–312行，删除注释后的执行语句：

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

形参定义位于第238–253行：三个指针、M/N/K、六个stride、三个块尺寸、GROUP_SIZE_M和ACTIVATION。pid_m/pid_n来自下一节要读的调度计算，角色与教学版pm/pn一样。

这里有三处与教学代码不同。第一，A行/B列使用 `% M`、`% N` 取模，把边缘超界的行列映射回合法输入；所以load只检查K。这样为将被丢弃的输出位置计算一些重复结果，但不越界读取。

第二，k仍是“第几个K块”，offs_k一直为块内偏移；指针每轮向前移动BK个K元素。判断 `offs_k < K-k*BK` 等价于“本轮全局k坐标<K”。不能再额外给指针加一次k*BK，否则会重复前进。

第三，tl.dot直接表达二维块矩阵乘，并把结果加进已有accumulator；它不显式构造教学版的BM×BK×BN临时products。实现怎样选择硬件指令留给布局与后端章，源码一行dot并不自动保证任何输入shape都可用Tensor Core。

为什么只在输入行列取模、不把输出坐标也取模？输出边缘多出的逻辑位置必须丢弃。若写回也取模，多个program可能覆盖同一有效输出，制造错误或竞争。

### 9. 官方写回先保留原始坐标，再mask

同文件第315–325行，连续执行语句摘录：

```python
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

ACTIVATION是constexpr，默认空字符串则不进入激活分支。leaky_relu定义在第329–331行，用where选择非负值本身或0.01*x。它在FP32累加结果上进行，然后再转FP16写出；这样不必先写C到显存再启动第二个激活kernel。

输出坐标offs_cm/offs_cn没有取模，因此边缘位置正确被mask掉。以M=3、BM=2、pid_m=1为例，行坐标为[2,3]：输入行可取模为[2,0]安全读取，输出行仍[2,3]，第二行不写入。

这份教程固定把c转换为FP16，wrapper也分配FP16输出。教学代码保持FP32，是为了单独讲地址和数学；不能拿教学版的精度承诺替换官方接口。

### 10. program 编号怎样变成输出块坐标

官方不用二维grid，而是先生成一维pid，再按分组顺序映射到块坐标。第261–269行原样摘录：

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

num_pid_m/n是输出块的行列数量，常比元素M/N小。GROUP_SIZE_M规定每组最多包含多少块行；一组覆盖这些块行与所有块列。

group_id用整除找到第几组；first_pid_m定位组内第一块行；group_size_m用min处理最后不足一整组的块行。剩下两行让组内块行先变化、块列后变化。两个取模各有对象：先取组内pid，再取组内块行。

纸上令块行数3、块列数2、GROUP_SIZE_M=2，总共6个program：

| pid | group_id | group_size_m | (pid_m,pid_n) |
|---|---|---|---|
| 0 | 0 | 2 | (0,0) |
| 1 | 0 | 2 | (1,0) |
| 2 | 0 | 2 | (0,1) |
| 3 | 0 | 2 | (1,1) |
| 4 | 1 | 1 | (2,0) |
| 5 | 1 | 1 | (2,1) |

比如pid5：group_id=5//4=1，first_pid_m=2，group_size_m=1，组内余数1，得到行2、列1。最后一组若仍错误除以固定2，就可能映射到不存在的块行或漏列。

相邻块(0,0)/(1,0)使用同一块列对应的B数据，因此这个顺序旨在提高L2复用机会。GPU实际安排program的时间顺序仍受调度影响；编号重排不是保证所有相邻pid必定同时执行的同步机制。是否更快必须测量，不能把教学表直接当cache命中日志。

### 11. wrapper 怎样把矩阵对象变成这一组形参

同文件第339–357行，删注释摘录：

```python
def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c
```

两个shape赋值中的K表示共同内积长度，前面的assert先保证它们相等。只对a断言连续，不对b断言连续；b的步长作为实参传下去，但仍要满足kernel的正stride等合同。kernel第275–282行有tl.assume，帮助编译器分析编号与正stride；assume不是host输入校验，调用方不能给出违背条件的值。

grid lambda接收候选配置中的块大小，用两个方向的向上取整乘出program总数。没有直接显式提供BLOCK_SIZE_M等，是因为外层autotuner将配置参数补进去；不应把它们当作无来源全局变量。

### 12. autotune 是选配置，JIT 是编译配置

教程第233–237行：

```python
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
```

这里只摘两个装饰器，紧接的函数定义是前面已读的matmul_kernel；装饰器片段不能单独执行。

装饰器自下而上应用：先用JIT包装kernel，再用autotune包装它。get_autotune_config在第221–225行按cuda/hip选择候选列表；CUDA候选示例在第171–174行：

```python
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
              num_warps=8),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
              num_warps=4),
```

字典中的名字对应kernel constexpr形参；num_stages/num_warps是编译选项。它们不是“第一组配置算一次矩阵左半，第二组算右半”，而是完成同一任务的不同候选实现。

[runtime/autotuner.py](../../triton/python/triton/runtime/autotuner.py) 第238–258行根据指定key与参数dtype形成缓存键；缓存未命中且有多个剩余候选时，核心语句为：

```python
timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
self.cache[key] = builtins.min(timings, key=timings.get)
```

这是实际试跑并测量候选，不是编译器静态证明哪一个一定最快。第281–295行读取所选config，并把 `**config.all_kwargs()` 交给 `self.fn.run`；self.fn在这里是内层JIT包装对象。所以配置选择后才进入4.2讲的参数绑定/特化/编译缓存/启动。

autotune配置缓存保存“这类输入选哪个配置”；JIT缓存保存“这个编译条件下的产物”。两者不能合称一个缓存。改变M/N/K会影响调优键；配置与dtype变化也可影响JIT产物。第一次调优成本不能混进声称稳态kernel延迟的统计里。

### 13. 精度与适用边界要单独读

官方主测试第366–375行用FP16输入、FP16输出，与torch.matmul比较，容差为atol=1e-2、rtol=0。这是上游测试选择，不是本章已运行通过的结果，也不是所有大小都自动适用的误差保证。

accumulator为FP32有助于保留多轮加法精度，但不能恢复输入先前量化丢掉的信息；最终转FP16仍会舍入，也可能遇到输出范围限制。不同归约次序与硬件路径也可能造成差异。

如果改为FP32输入，还必须看dot的input_precision，而不能只看accumulator类型。[language/core.py](../../triton/python/triton/language/core.py) 第2294–2298行实际选择默认值：

```python
assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
if input_precision is None:
    supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
    input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                                 (allow_tf32 or allow_tf32 is None)) else "ieee")
```

因此默认值会受到目标支持与knob影响。FP32存储并不自动表示所有乘法都按完整IEEE FP32精度执行；要固定实验口径，应明确目标与选项，并用适合输入规模的误差指标验证。FP8教程分支还引入额外dtype/目标限制，不能把这里的小整数手算当作低精度通用证明。

教学版的三维 products 适合解释内积，官方 dot 适合把块矩阵乘交给更专门的编译路径。两者数学联系明确，性能与可运行形状约束不同。6.3 节测试了这个教学实现的 GPU 正确性；仍没有速度结论。

### 14. 自测与论文回读

**问题一：** BK=2、K=3跑几轮，最后一轮哪项补零？两轮，块内索引1对应全局k=3，补零。**问题二：** 为何A越界行取模不会改写有效C？多余行的计算结果最后被原坐标mask丢弃。**问题三：** `tl.dot(a,b,accumulator)`与普通逐元素`a*b`相同吗？不同；dot沿共同K维求和，输出BM×BN，逐元素乘法必须先处理shape与归约才能表达同一数学目标。

> 回到论文：Listing 1 矩阵乘（PDF第2页）可对照tile索引、广播指针、mask与累加；§3.2.2/Listing 4（PDF第3–4页）可对照广播；§5.3 Auto-tuner与§6.1 Matrix Multiplication/Figure 8（PDF第7页）可对照搜索与实验问题。见[定位表](../appendices/reference.md#sec-a-1)。现代Python装饰器、分组调度、精度接口与Tensor Core路径不能倒写成2019论文原样实现，历史性能数也不是当前结果。

本章完成了“一个输出块如何算出并写回”的闭环。还没回答的是同一个BM×BN逻辑块怎样分给物理线程、warp与寄存器；[7.1](07-layout-passes-memory.md#sec-7-1)从这里接着读布局。

## 6.3 实际运行：Softmax 与 GEMM 的数值检查

这一节运行 6.1、6.2 的两个教学 kernel，并与 PyTorch 参考结果比较。环境同 4.3：A800 80GB、PyTorch 2.10.0+cu129、从固定提交构建的 Triton 3.9.0。

**Softmax。**

```bash
python examples/actual_softmax.py
```

```text
input [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]
output [[0.09003056585788727, 0.2447284460067749, 0.6652408838272095],
        [0.3333333432674408, 0.3333333432674408, 0.3333333432674408]]
reference [[0.09003056585788727, 0.2447284758090973, 0.6652408838272095],
           [0.3333333432674408, 0.3333333432674408, 0.3333333432674408]]
max_abs_diff 2.9802322387695312e-08
block 4
```

第一行输出与 6.1 的纸笔表一致：`[0.0900, 0.2447, 0.6652]`；第二行三个相等的输入得到三个 `1/3`。最大绝对差约 `3e-8`，来自设备 `exp` 近似与浮点舍入，不是公式错误。`block=4` 是不小于三的最小 2 的幂；第四个逻辑位置由 `mask` 和 `other=-inf` 处理。

**GEMM。**

```bash
python examples/actual_gemm.py
```

```text
grid (1, 1)
output [[58.0, 64.0], [139.0, 154.0]]
reference [[58.0, 64.0], [139.0, 154.0]]
max_abs_diff 0.0
```

这个教学 kernel 按 6.2 的显式 `products` + `tl.sum(axis=1)` 计算，不调用 `tl.dot`。2×3 与 3×2 相乘得到与手算相同的四个整数结果。`max_abs_diff=0.0` 只说明这个极小例子的乘加顺序恰好一致；更大的矩阵、不同累加顺序或 Tensor Core 路径不能据此保证逐位相等。

**边界。** 本节验证的是两个教学 kernel 的数值合同，不是官方教程完整实现的性能；没有测延迟、带宽或 Tensor Core 利用率。

## 第 6 章小结

| 你现在应该能回答 | Softmax | GEMM |
|---|---|---|
| 输出由哪些输入决定？ | 同一行所有列 | 沿 K 的所有项 |
| 为什么需要归约？ | 分母是整行指数和，最大值也沿整行 | 每个输出是 K 项乘积之和 |
| 无效位置补什么？ | `-inf`，使 max 不受影响、exp 后为 0 | `0`，使乘加不受影响 |
| 数据在哪里复用？ | 一行加载后在寄存器中算 max/exp/sum | A/B 块在同一 K 轮内被多个输出位置复用 |
| host 怎样决定 grid？ | 行数、persistent 行数、资源估计 | 输出块数、分组顺序、autotune 配置 |
| 哪些东西不能从纸笔例子推出？ | 实际 exp 精度、occupancy、速度 | 实际 Tensor Core 可用性、精度、速度 |

### 综合检查

1. Softmax 的 `other=0` 为什么错？0 会参与 max 和 exp，改变分母；应使用 `-inf`。
2. GEMM 的尾部为什么可以补 0？因为乘加中 0 不贡献结果；但 mask 仍必须阻止越界读取。
3. `tl.dot(a, b, acc)` 和 `a * b` 有什么区别？`dot` 沿共同 K 轴求和并输出 BM×BN；逐元素乘法只得到同形状乘积。
4. 边界的输入行列取模、输出不取模，为什么？输入取模保证安全读取，输出保留原坐标再用 mask 丢弃多余位置。
5. autotune 和 JIT 缓存是同一个东西吗？不是；autotune 选配置，JIT 缓存编译产物。
