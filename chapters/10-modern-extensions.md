# 第 10 章 现代扩展：Gluon、低精度与 AMD

[返回总目录](../README.md) · [上一章：大模型算子](09-llm-kernels.md) · [下一章：两代对照](11-two-generations.md)

第 4–9 章主要走经典 `tl` + NVIDIA 路线。真实 Triton 还在几个方向扩展：

- **Gluon**：把布局等选择显式交给程序员，而不是完全依赖自动布局；
- **低精度**：block scaling、packed 存储、scale 布局和矩阵指令的输入约束；
- **AMD**：同一套编译职责，另一套目标阶段、汇编格式和启动路径。

这三者放在同一章，是因为它们都回答“当经典 `tl` 的默认选择不够用时，程序员或后端还掌握哪些旋钮”。但它们改变的是不同层：Gluon 改语言/布局接口，低精度改数据格式和矩阵指令合同，AMD 改目标后端。

## 本章地图

| 扩展 | 解决什么问题 | 主要改变哪一层 | 仍然共用的思想 |
|---|---|---|---|
| Gluon | 需要显式控制布局、共享存储或异步协议 | 前端语言 + 布局接口 | 仍是块编程，逻辑 tensor 与 program 语义保留 |
| 低精度 | 用更少位保存权重/激活，并用 scale 还原 | dtype、存储打包、scale 布局、MMA | 仍是“加载块 → 矩阵乘 → 累加 → 写回” |
| AMD | 在非 NVIDIA 目标上编译和启动 | TTGIR 之后的 lowering、汇编、链接、launcher | 前端和大部分优化职责与 NVIDIA 共用 |

**读法：** 先确定问题：普通 `tl` 能否正确表达？自动布局是否够用？目标硬件是否支持该精度/指令？不要为了“更底层”而默认改写 Gluon，也不要把 NVIDIA 的 PTX/cubin 术语套到 AMD。

**运行边界：** 本章的 Gluon、低精度和 AMD 示例受目标架构、指令与后端支持限制，正文以固定源码核对为主。书中实际运行记录集中在第 4–9 章，且使用 NVIDIA A800；它们不能替代本章各目标的兼容性验证。

<a id="sec-10-1"></a>
## 10.1 现代扩展：把具体选择交回程序员

本章固定现代源码 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。下列三页分别闭合一个问题，均静态核对、未执行GPU：

1. [Gluon：显式指定布局的一维copy](#sec-10-2)，从host `BlockedLayout` 到 `gl.arange`、指针、mask与输出，再看显式转换的接口。
2. [低精度：scale怎样进入矩阵乘](#sec-10-3)，从量化后的数与scale手算，到经典 `tl.dot_scaled` 的打包索引，再读Gluon异步scaled MMA的完成协议。
3. [AMD：同一编译职责，另一套目标阶段](#sec-10-4)，从warp_size选项到TTGIR、LLVM、amdgcn、hsaco与HIP launcher，不把CUDA术语机械替换成AMD名字。

这些不是2019基础论文中已完成能力的展开。Gluon、现代Tensor Core指令、block scaling及今天的AMD后端都按固定快照的真实代码解释；本书没有用旧论文填补新硬件协议，也没有复用旧设备上的性能数字。

选择学习路线时，先确定问题：普通tl能否正确表达、自动布局是否足够、是否需要显式内存/异步协议、目标是否实际支持指定精度。更多控制意味着更多合同要由程序员证明，不意味着每个kernel都需要改写Gluon。硬件/软件版本更新时须重新核验教程的检测逻辑、指令约束与实际测试，不能只凭产品名推定兼容。

**先看最像经典 `tl` 的扩展。** Gluon 仍然写块程序，只是把 `layout` 作为显式值交给 kernel。下一部分用一维 copy 手算 lane/warp 持有关系，再看二维 copy 为什么需要两套索引和 `convert_layout`。

<a id="sec-10-2"></a>
## 10.2 显式布局怎样进入一个 Gluon copy

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。主要来源[gluon/02-layouts.py](../../triton/python/tutorials/gluon/02-layouts.py)，静态核对与手算，未运行GPU。Gluon位于 `triton.experimental`，接口不应当作永久稳定的语言标准。

### 1. 与tl的第一个差别发生在哪里

4.1写 `tl.arange(0,B)`，后面的编译器选择分布。Gluon让程序显式给逻辑tensor附布局。不是绕过JIT、也不是在host启动每个lane的Python函数；它仍是块编程，scalar保持块内一致，逐元素差异由tensor表达。

官方教程[01-intro.py](../../triton/python/tutorials/gluon/01-intro.py)41–46行给出import：

```python
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

`gluon.jit`类似经典 `triton.jit` 的设备函数入口，`gl`提供该语言的操作及布局类型；不要混用 `tl.tensor` 与 `gl`布局表达式并假定自动互通。

### 2. host先构造布局，再交给kernel

真实[02-layouts.py](../../triton/python/tutorials/gluon/02-layouts.py)184–188行：

```python
def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel
```

`input/output`由调用方分配；`numel()`只在host查询元素数，没有逐元素读GPU。`grid`决定program数；`XBLOCK`为每program逻辑元素数，`layout`为编译期分布描述，`num_warps`为编译/启动选项。

测试233–239行创建输入/输出后：

```python
layout = gl.BlockedLayout([1], [32], [num_warps], [0])
memcpy_1d_impl(input, output, XBLOCK, layout, num_warps=num_warps)
torch.testing.assert_close(input, output, atol=0, rtol=0)
```

这四个位置参数依次是 `size_per_thread`、`threads_per_warp`、`warps_per_cta`、`order`。每项都是一维列表，因为逻辑tensor rank1。每线程基础1元素，每warp32个lane，CTA内num_warps个warp，只有dim0可作排布顺序。

`layout`与`num_warps`要一致，不能传入描述4warp的布局却随手用1warp发射。测试明确用num_warps=4；这页沿它手算，不推定适用于任意AMD wavefront目标。

### 3. kernel的每一句怎样承接

真实164–181行，删去注释/空行：

```python
@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)
    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel
    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)
```

host的input/output分别进入in_ptr/out_ptr，元素数进入xnumel。`XBLOCK/layout`标constexpr，参与编译期特化，而不是每元素运行时读取的数组。

`pid/start`是块内一致标量；`indices`是XBLOCK长的逻辑向量，显式附上layout。后续加start、构造指针、比较mask时，tensor布局从indices向前传播；因此不需要每行重新写layout。

`in_ptrs`是指针tensor，不是已经读取的数据。`gl.load`描述设备读入，`value`才是元素值tensor；无效位置未给other，不为它们推测具体值。`out_ptrs`对应相同逻辑offsets，store用同mask，只保存有效位置。

编译期间这些句子构造带类型/布局的IR；设备运行时才按该分布读写。返回给host wrapper的 `compiled_kernel` 是编译产物/启动相关对象，不是“copy结果数组”；输出数据仍在调用方的output张量里。

### 4. 真正手算一次线程持有关系

选官方测试覆盖的N=200、XBLOCK=128、num_warps=4、上述layout。grid=2。第0块逻辑offset0..127，每线程1元素，`offset=32*warp+lane`。

第1块start=128，相同局部映射：

| warp | lane范围 | 全局offset | 有效情况 |
|---:|---|---|---|
| 0 | 0..31 | 128..159 | 全有效 |
| 1 | 0..31 | 160..191 | 全有效 |
| 2 | 0..7 | 192..199 | 有效 |
| 2 | 8..31 | 200..223 | mask为假 |
| 3 | 0..31 | 224..255 | mask为假 |

元素199由第1块warp2/lane7持有。若input[199]=17，设备读取并向output[199]写17；input/output的dtype决定地址按多少字节前进。

若XBLOCK=256，线程数仍128，shape超过基础布局，通常每线程需两个逻辑槽。例如一维该配置中局部i=199可分解为register1贡献128、warp2贡献64、lane7贡献7。与7.1的LinearLayout手算相接，但真实物理寄存器分配仍在后端。

### 5. 二维copy为什么会有两套索引

同文件713–737行 `memcpy_2d_inout_kernel` 的关键摘录：

```python
mask_in, in_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_in, ystride_in,
                                           XBLOCK, YBLOCK, layout_in)
mask_out, out_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_out, ystride_out,
                                             XBLOCK, YBLOCK, layout_out)
value = gl.load(in_ptr + in_offsets, mask=mask_in)
value = gl.convert_layout(value, layout_out)
gl.store(out_ptr + out_offsets, value, mask=mask_out)
```

辅助函数在同文件702–710行接收起始坐标、shape、两轴stride、块shape和layout，按SliceLayout创建行/列索引，广播得到二维offset与有效mask。输入/输出stride可能不同，所以相同逻辑坐标需要两套地址；布局也可分别优化两边访问。

`value`初始按layout_in分布，输出指针tensor按layout_out分布。`convert_layout`保留逻辑二维值，改变持有者，使每个线程在store时拿到自己负责输出位置的值。它不是转置数学结果，也不改变输出shape。

例如逻辑矩阵 `[[1,2],[3,4]]`，输入row-major stride `[2,1]`，输出列优先stride `[1,2]`，坐标(1,0)的值3从输入偏移2写到输出偏移1。即使物理存储顺序改变，逻辑输出(1,0)仍是3。线程分布转换与全局stride变换是两层决策。

### 6. 显式不代表所有转换都昂贵

layout相同可能消除转换；某些改变仅在线程内重排，另一些需要warp shuffle或共享内存。要看实际lowering/dump，不能按源码 `convert_layout` 行数估计通信成本。

Gluon还暴露共享存储及目标特定异步原语。显式操作仍要遵守完成、可见性和消费者交接协议；手写一个barrier不能自动修复错误布局或无效descriptor。下一节用scaled MMA的真实等待链展示这种责任。

### 7. 自测

1. XBLOCK=128是否表示128个warp？不是，是128个逻辑元素，本例4warp共128线程。
2. N=200第二块warp2/lane8能读input[200]吗？不能，mask为假。
3. layout改变是否必然改全局张量值？不应该，改变持有方式，逻辑计算合同不变。
4. host的layout对象是不是GPU数据？不是，是特化/类型推导用的描述。

Gluon是2019之后的语言接口；论文§5.2.1/Figure 5帮助理解层次分布思想，不包含本部分API与可用硬件证明。见[论文定位表](../appendices/reference.md#sec-a-1)。

**从布局到数据格式。** Gluon 解决“值放在哪”，低精度解决“值用什么格式存、scale 存在哪、矩阵指令怎样消费”。下一部分先手算 block scaling 的乘加，再读经典 `tl.dot_scaled` 和 Gluon 的异步 scaled MMA 完成协议。

<a id="sec-10-3"></a>
## 10.3 低精度矩阵乘中的 scale 到底乘在哪里

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。本部分读[10-block-scaled-matmul.py](../../triton/python/tutorials/10-block-scaled-matmul.py)和[Gluon 11](../../triton/python/tutorials/gluon/11-tcgen05-mma-scaled.py)。静态核对、纸笔推演，未运行GPU、量化模型或性能测试。

### 1. dtype、scale、累加精度是三个决定

把实数x近似成 `q*s`，q用较少位保存，s是一组元素共享的尺度。q的格式决定可表示值，s决定还原范围，累加器类型决定矩阵乘中间和的表示。fp32累加不会恢复量化前丢失的信息。

block scaling按K归约维每VEC_SIZE个逻辑元素一个scale，不是整矩阵只有一个scale。若A为 `[M,K]`，尺度为 `[M,K/VEC_SIZE]`；B在教程中按 `[N,K]`存，尺度 `[N,K/VEC_SIZE]`，乘法时将B转置。

教学例不采用真实硬件最小块，只用VEC_SIZE=2方便手算：A量化值 `[1,2,1,2]`，A尺度 `[2,1]`，还原A=`[2,4,1,2]`；B量化值 `[1,1,2,2]`，尺度 `[1,3]`，还原B=`[1,1,6,6]`。点积为 `2+4+6+12=24`。

不能先做量化整数点积再乘一个全局scale。正确分块为第一块 `(1*1+2*1)*2*1=6`，第二块 `(1*2+2*2)*1*3=18`，合计24。两个K块尺度乘积不同。

### 2. 教程先检测目标，再决定可用格式

经典教程133–143行真实检测：

```python
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'

def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]) or is_hip_cdna4()
```

这是固定教程接受哪些目标的逻辑，不是所有目标、所有格式、所有布局的兼容性表。该文件756行还限制AMD教程当前只支持mxfp4。Gluon 11的86–92行则只接受CUDA capability主版本10，不能因为经典教程支持AMD，就把NVIDIA `tcgen05` API搬过去。

格式同样要区分：Gluon 11的21–29行说明nvfp4使用VEC_SIZE16、e4m3尺度；MX使用VEC_SIZE32、e8m0尺度。e8m0按指数编码，字节值不是浮点scale本身；不要把存储字节2直接解释为尺度2。下面数例中的2/3表示已解码尺度。

### 3. 逻辑K与uint8存储K不能混为一谈

经典教程338–345行：

```python
VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2
```

fp4每逻辑元素4bit，两个打包进一个uint8，fp8通常一字节一个。`ELEM_PER_BYTE`决定descriptor的K地址步进单位；它不改变数学归约长度K。

例如BLOCK_K=256逻辑fp4元素，A/B descriptor沿存储K前进128个字节槽；若A是fp8、B是fp4，则A前进256、B前进128。scale每VEC_SIZE个逻辑元素一项，因此VEC_SIZE32时两边都需要8项scale，不能再因fp4除2变成4项。

### 4. 从真实load到dot_scaled

经典教程216–241行，非连续摘录，保留fp4×fp4分支：

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES,
                  disallow_acc_multi_buffer=disallow_acc_multi_buffer):
    a = a_desc.load([offs_am, offs_k_a])
    b = b_desc.load([offs_bn, offs_k_b])
    scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
    scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])
    scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
    scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
    accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
    offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
    offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
    offs_scale_k += rep_k
```

注意这段为教学选支摘录，去掉条件后不是可替代上游完整kernel的通用代码。`a_desc/b_desc`由host构造，负责packed矩阵块；`offs_am/offs_bn`在207–212行由pid_m/pid_n决定，`offs_k_*`初始0。scale descriptor采用为目标访问设计的物理打包，并非直接二维连续矩阵。

`reshape/trans/reshape`把物理scale块恢复为 `dot_scaled`要求的逻辑 `[BLOCK_M,BLOCK_K/VEC_SIZE]` 与 `[BLOCK_N,BLOCK_K/VEC_SIZE]`，不是对量化矩阵本身做反量化。比如BLOCK_M128、BLOCK_K128、VEC_SIZE32，逻辑scale应为 `[128,4]` 共512项，`rep_m=rep_k=1` 时中间 `[1,1,32,4,4]` 也有512项。转置重新排列对应关系，元素数保持不变。

`"e2m1"`指示packed数值的格式含义，不表示a这个uint8指针已经变成普通fp4数组。B转置使数学乘法为 `[M,K]@[K,N]`，但B尺度仍按输出N行组织。第三个矩阵实参 `accumulator` 接收历史K块的和，结果继续更新它。

`offs_scale_k += rep_k` 沿物理scale块推进；不能拿逻辑8项直接替换这个物理索引步长。读者要同时保留逻辑shape和descriptor实际打包shape，不能只凭变量名猜单位。

### 5. 参考实现也要先解包尺度

经典教程419–426行，摘录最后三句：

```python
a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)
```

`unpack_scale`在前面把物理格式还原为二维逻辑尺度；`repeat_interleave`把每个尺度沿K复制VEC_SIZE次，变成与每个量化元素一一对应的尺度；B再转置。最后才做反量化值的参考matmul。

这个reference检查“给定量化值与尺度的scaled matmul是否正确”，并不直接证明“量化后与原始全精度模型一样准确”。后者还要比较原始权重/激活、模型输出与任务质量，不能只看此局部误差。

### 6. Gluon显式scaled MMA多管理哪些对象

[Gluon 11](../../triton/python/tutorials/gluon/11-tcgen05-mma-scaled.py)147–160行，非连续摘录：

```python
scale_layout: gl.constexpr = TensorMemoryScalesLayout()
a_scale_tmem = allocate_tensor_memory(a_scale_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
b_scale_tmem = allocate_tensor_memory(b_scale_ptr.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
use_acc = False
```

Tensor Memory是此NVIDIA路径的特定存储空间，不是普通tl逻辑tensor的别名。前两对象保存尺度，acc_tmem保存fp32输出累加；布局类型不同，不能随意交换。a/b操作数另在shared memory中，descriptor与实际存储都由kernel前文准备。

循环中175–178行用TMA异步加载操作数：

```python
mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
tma.async_load(a_desc, [off_m, off_k_a], bar, a_smem)
tma.async_load(b_desc, [off_n, off_k_b], bar, b_smem)
mbarrier.wait(bar, phase)
```

`bar`在157–160行分配并初始化，计数1；expect声明本轮预计到达字节数，两个load把数据送到a_smem/b_smem并关联bar。wait等待相应phase完成，保证后续MMA使用该批数据，而非把“提交了load”当作已读完。

尺度由188–204行普通load进寄存器，`get_reg_layout/convert_layout`调整为tensor memory store接受的布局，再 `a_scale_tmem.store/b_scale_tmem.store` 写入尺度存储。

### 7. 发MMA、等完成、读acc、最终写回

214–239行，非连续摘录，省略注释：

```python
tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                   use_acc=use_acc)
tcgen05_commit(mma_bar)
mbarrier.wait(mma_bar, phase)
use_acc = True
phase ^= 1
```

第一个调用传共享操作数、tensor memory累加器和尺度，格式字符串来自208–209行fp4/fp8分支。`use_acc=False`使第一次不读取未初始化累加器；以后True延续K块和。commit关联完成barrier，wait等待该阶段，`phase ^= 1` 在0/1之间翻转供循环复用。

循环结束227–239行：

```python
mbarrier.invalidate(bar)
mbarrier.invalidate(mma_bar)
acc = acc_tmem.load()
acc = acc.to(c_desc.dtype)
acc_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
acc_smem.store(acc)
fence_async_shared()
tma.async_store(c_desc, [off_m, off_n], acc_smem)
tma.store_wait(0)
```

先invalidate结束barrier生命周期，避免后面共享分配复用其字节时仍留活动状态。acc从tensor memory加载到寄存器并转换输出dtype，再写到输出shared缓冲。`fence_async_shared`建立普通shared写入与异步代理之间的可见性，TMA提交写回后 `store_wait(0)`等待相关存储。不能删掉等待后仍声称临时shared可立即释放/复用。

这是从量化数据和尺度到最终输出的对象闭环；真实可执行kernel还包含合法tile/descriptor、初始化、布局与架构约束，不能拿几句API拼成任意硬件可跑的kernel。

### 8. 自测与适用边界

1. fp4一个字节装两个值，BLOCK_K256时存储K步长？128字节槽；尺度仍按逻辑K分组。
2. fp32累加能否消除量化输入误差？不能。
3. 两个相邻K块scale不同，可在最后只乘一次scale吗？一般不行。
4. `tcgen05_commit` 是否代表CPU收到结果？不是，设备异步MMA的完成协议，后面还要wait/读回/输出。
5. AMD gfx950支持教程mxfp4，能否调用NVIDIA Gluon tcgen05_mma_scaled？不能，由不同目标API与lowering实现。

这些格式、指令与Tensor Memory属于2019之后的实现。正文以实际教程API和源码检测为依据，不照抄旧论文硬件说明。Gluon 06的早期注释甚至还写“shared→tensor copy未暴露”，而同快照Gluon 10/11已经演示tcgen05_copy，说明跨教程概述也须以具体当前代码交叉核验。

**换一个目标后端。** 前面两部分仍然假设 NVIDIA 的 PTX/cubin 路线。AMD 复用前端和大量 TTGIR 优化，但从目标转换开始分开：warp size、矩阵指令 pass、LLVM triple、amdgcn、hsaco 和 HIP launcher 都有自己的选择。下一部分用同一套编译职责，逐段标出哪些层共用、哪些层必须重读。

<a id="sec-10-4"></a>
## 10.4 AMD 后端：从同一 tile 到 hsaco

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。依据[AMD compiler.py](../../triton/third_party/amd/backend/compiler.py)和[driver.py](../../triton/third_party/amd/backend/driver.py)静态带读；未在AMD设备编译/运行。

本部分用同一个连续copy的编译对象，回答“哪些层继续共用，哪些层从目标转换开始分开”。它没有把CUDA PTX名字替换一下就称完成移植。

### 1. 同样的num_warps，不一定同样多线程

AMD编译选项423–433行：

```python
base_arch = self.arch.removesuffix("-strict")
object.__setattr__(self, 'base_arch', base_arch)
gfx_major = int(base_arch[3:-2])
warp_size = 32 if gfx_major >= 10 else 64
object.__setattr__(self, 'warp_size', warp_size)
assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
    "num_warps must be a power of 2"
```

`self.arch`从目标/解析选项来，例如gfx950。去掉strict后缀，再去掉gfx前缀和末尾minor/patch位，取major。gfx950得到9，因此该快照选warp_size64；gfx1100得到11，选32。这是**本后端的选择规则**，不是对所有AMD硬件全部可选wave模式的概括。

`object.__setattr__`用于更新选项对象属性。`num_warps`要求正的2次幂；后续实际线程数由它乘warp_size计算。在gfx950上num_warps4对应256线程，在10.2的NVIDIA例中对应128线程，不能直接复用相同lane映射公式。

### 2. TTIR到TTGIR拿到目标warp_size

585–599行，非连续摘录：

```python
pm = ir.pass_manager(mod.context)
pm.enable_debug()
passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                   options.num_ctas)
pm.run(mod, 'make_ttgir_early')
pm = ir.pass_manager(mod.context)
pm.enable_debug()
emuTF32 = False
passes.ttgpuir.add_coalesce(pm)
passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_thread_locality(pm)
amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
```

`mod`仍是公共前端交来的MLIR module。转换函数与NVIDIA入口相似，但目标字符串是hip，warp_size由选项传入；先run早期转换，再新建pm追加后续优化。

coalesce等一部分pass共用；矩阵指令加速选择交给AMD专属pass，额外选项包括目标arch、非K指令维度与K打包。这里即使出现该pass，copy kernel也不会凭空变成矩阵乘；pass处理符合条件的操作，不能按pipeline列表推断每个kernel的指令。

**教学映射：** 若选择一维基础布局 `sizePerThread1,threadsPerWarp64,warpsPerCTA4`、shape256，则可按 `i=lane+64*warp` 手算。元素199在warp3/lane7。它只说明这一明确配置；实际自动布局要看真实TTGIR，不声称4.1或Gluon会自动生成该配置。

### 3. LLVM方言之前仍需要分配和同步

679–710行，非连续摘录：

```python
mod = src
pm = ir.pass_manager(mod.context)
pm.enable_debug()
amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
amd.passes.ttgpuir.add_warp_pipeline_conversion(pm, options.arch)
passes.convert.add_scf_to_cf(pm)
amd.passes.ttgpuir.add_allocate_shared_memory(pm, options.arch)
amd.passes.ttgpuir.add_membar(pm, options.arch)
passes.ttgpuir.add_allocate_global_scratch_memory(pm)
amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ, options.wgp_cu_mode == "cu")
```

`src`是TTGIR，不是AMDGPU汇编。异步wait计数与pipeline转换依赖arch；共享存储在AMD硬件语境通常称LDS，仍需合法分配与同步。`membar`是AMD实现，不是NVIDIA的mbarrier协议。

`__HIP_FTZ`在同函数709行设置True，控制exp2相关denorm行为；`wgp_cu_mode`是目标执行模式选择。这里写出这些参数是为了看到数值/目标合同也进入lowering，不能假设源代码相同就保证跨目标逐bit输出相同。

真实pipeline还包含插桩、warp specialization、CF/arith转换和清理，随后run；本摘录只保留对象衔接，不是推荐删除其他pass的简化实现。

### 4. 导出LLVM Module，再编译成AMDGPU汇编

749–758行：

```python
llvm.init_targets()
context = llvm.context()
llvm_mod = llvm.to_module(mod, context)
target_triple = amd.get_target_triple(options.arch)
amd.attach_target_triple(llvm_mod, options.arch)
target_features = ''
if knobs.compilation.enable_asan:
    target_features = '+xnack'
llvm.attach_datalayout(llvm_mod, target_triple, options.arch, target_features, get_llvm_flags(options))
```

与8.1一样，先由MLIR LLVM方言转真正LLVM module，但triple、arch、features是AMD路径。后面设置设备库控制常量、链接库、优化，再在871行返回 `str(llvm_mod)`。`metadata["shared"]`等在858行附近保存，供运行时消费。

`make_amdgcn`874–888行先从LLVM文本找 `amdgpu_kernel` 函数名，保存metadata，计算目标triple和IR hash。默认分支907–920行调用 `compile_amdgpu(src,target_triple,options.arch,features,...)`，返回amdgcn汇编文本；特殊调试选项可走MIR替换路径，本部分不把它当默认执行。

在这一阶段没有PTX，也没有调用NVIDIA ptxas。

### 5. 汇编、链接，再返回hsaco二进制

928–940行 `make_hsaco`，真实主体：

```python
target_features = []
if knobs.compilation.enable_asan:
    target_features.append('+xnack')
hsaco = assemble_amdgcn(src, options.arch, ','.join(target_features))
with tempfile.NamedTemporaryFile() as tmp_out:
    with tempfile.NamedTemporaryFile() as tmp_in:
        with open(tmp_in.name, "wb") as fd_in:
            fd_in.write(hsaco)
        amd.link_hsaco(tmp_in.name, tmp_out.name)
    with open(tmp_out.name, "rb") as fd_out:
        ret = fd_out.read()
return ret
```

此时 `src`是汇编字符串；`assemble_amdgcn`先生成对象bytes（局部名hsaco不代表已经完成最终链接），写临时输入，`amd.link_hsaco`链接到临时输出，最后read得到最终产物bytes。公共编译器将其缓存，CompiledKernel按后端 `binary_ext`选择它。

942–951行注册明确串起这几个函数：

```python
stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
```

lambda捕获选项，共用metadata；上一阶段输出作为下一阶段src，三处src的类型依次为MLIR module、LLVM文本、AMDGPU汇编文本。

### 6. 同一CompiledKernel，选择HIP launcher

[AMD driver.py](../../triton/third_party/amd/backend/driver.py)334–339行 `HIPDriver.__init__`：

```python
super().__init__()
self.utils = HIPUtils()
self.launcher_cls = HIPLauncher
```

公共CompiledKernel的惰性 `_init_handles`调用活动driver的launcher_cls和load_binary；选HIPDriver便落到这里。不是Python代码硬编码选择CudaLauncher。

`HIPLauncher.__init__`278–289行：

```python
signature = dict(src.signature)
tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
tensordesc_abi = TensorDescABI.HIP_TDM if tensordesc_meta else TensorDescABI.DECOMPOSED
utils = triton.runtime.driver.active.utils
expanded_signature = expand_signature(signature.values(), tensordesc_abi)
self.arg_annotations = annotate_arguments(expanded_signature)
self.launch = wrap_handle_tensordesc(utils.launch, signature, tensordesc_meta)
self.launch_cooperative_grid = metadata.launch_cooperative_grid
self.warp_size = metadata.warp_size
```

`src.signature`仍来自kernel参数合同；descriptor按HIP ABI展开。`warp_size`从编译元数据进入launcher，确保发射参数没有误按CUDA固定32解释。

303–331行 `__call__`准备scratch后提交：

```python
self.launch(self.launch_cooperative_grid, gridX, gridY, gridZ, stream, function, global_scratch,
            profile_scratch, kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook,
            self.warp_size, self.arg_annotations, self.kernel_signature, args)
```

function来自HIP装载句柄，args为本次调用的参数；C驱动[driver.c](../../triton/third_party/amd/backend/driver.c)680行在普通分支调用 `hipModuleLaunchKernel`。和CUDA路径一样，编译、装载、提交与设备完成要分开，返回Python不是隐式全设备同步。

### 7. 自测与移植检查

1. gfx950、num_warps4在此快照多少线程？4×64=256；不是固定128。
2. hsaco阶段是否输入PTX？不是，输入amdgcn汇编。
3. 同一个tl.load是否保证两目标相同机器指令？不保证，由目标lowering决定。
4. classic block-scaled教程接受gfx950，是否全部fp4/fp8格式都接受？不能据此推断；固定教程AMD分支只演示mxfp4。

移植一个kernel先保留数值合同、shape/stride和有效范围，再核对warp/layout、目标特定API、精度选项及资源要求；然后按8.2分开验证正确性与计时。本书没有跨厂商速度对比，也不声明这两条pipeline的所有优化等价。

2019论文没有本部分现代AMD/MLIR路线。可对照§5的编译职责理解共同问题，不能套用其硬件参数、指令或实验结论。见[论文定位表](../appendices/reference.md#sec-a-1)。

## 第 10 章小结

| 扩展 | 你新增要维护的合同 | 不能从经典 `tl` 直接搬来的东西 |
|---|---|---|
| Gluon | layout 与 `num_warps` 一致、显式转换、共享/异步协议 | 自动布局的假设、`tl.tensor` 与 `gl` 布局的混用 |
| 低精度 | dtype/scale 布局、packed K 步长、格式支持检测 | 全局 scale、fp32 累加能恢复量化误差、NVIDIA API 可跨到 AMD |
| AMD | 目标 arch、warp size、矩阵指令选项、hsaco/HIP launcher | PTX/cubin 阶段名、固定 32 线程 warp、NVIDIA 性能数字 |

### 综合检查

1. Gluon 是否绕过 JIT、让每个 lane 跑一个 Python 函数？不是；它仍是块编程，只是布局显式化。
2. `layout` 与 `num_warps` 不一致会怎样？编译/启动合同不匹配，不能假设自动修正。
3. fp4 一个字节装两个值，BLOCK_K=256 时存储 K 步长是多少？128 个字节槽；但 scale 仍按逻辑 K 分组。
4. fp32 累加能消除量化输入误差吗？不能；它只减少累加阶段的误差。
5. gfx950 的 `num_warps=4` 一定是 128 线程吗？不一定；该后端 warp_size 可能是 64，实际线程数由两者相乘。
6. hsaco 阶段输入是 PTX 吗？不是；输入是 AMDGPU 汇编，经过汇编和链接得到 hsaco。
