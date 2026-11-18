# §6–§8：从实验图回到实现与输入

[上一篇：编译器](03-compiler.md) · [路线目录](README.md)

## §6 开头（PDF 第 7 页）

段首说要比较深度学习中若干计算。接下来的句子规定了测量环境和对照对象：GTX1070，cuBLAS 10.0、cuDNN 7.0，以及当时的几套编译技术，适用时各问题大小分别调优。从[旧基准入口](../../triton-legacy/tests/bench)了解快照里有哪些可追踪的代码；复现图表还需要确认具体配置、版本和计时边界。读 TFLOPS 时把它理解成每秒完成多少万亿次浮点操作。输入形状、精度和运算计数必须一致才适合直接比较。

## §6.1 与 Figure 8：矩阵乘（PDF 第 7 页）

| 原文位置 | 怎样读 | 源码入口 |
|---|---|---|
| 第 1 段前半，公式与两个输入 | M 为 A 的行数，N 为另一矩阵的行数，K 是两个矩阵共享的乘加维。每个输出对应一对行，总共 M×N 个输出 | 打开[旧 dot 源码](../../triton-legacy/tests/common/src/dot.h)的输出块、两组地址和 K 循环；逐句参照[Listing 1](01-language.md#listing-1) |
| 第 1 段后半，任务与 Figure 8 | 横向不同组分别给方阵、DeepSpeech2 和 Transformer 相关形状；读某个柱组先抄下三个尺寸，再对照图例认实现 | [dot.cc](../../triton-legacy/tests/bench/dot.cc)是基准驱动，[common/dot.h](../../triton-legacy/tests/common/dot.h)包含输入布局、候选与参考路径，cuBLAS 计时块在此快照被注释。快照基准的列表要与图逐项核对 |
| 第 2 段关于接近峰值和 cuBLAS 的例外 | M 和 N 小时，可分出的输出块少。若 K 很长，可以让多个工作实例分别做一段 K，再合并部分和，以增加并行工作 | 在旧 kernel 找一个输出块怎样沿 K 顺序累加，再到[现代 GEMM 章](../chapters/06-classic-kernels.md#sec-6-2)看按输出块分工。论文把 cuBLAS 的优势归于 3D 算法，快照中被注释的 cuBLAS 调用说明了基准原本设想的接口 |
| 第 2 段最后一句 | 记录当时 DSL 的相对性能以及尺寸为 32 的倍数时的例外。应用于任何新设备前都需要新测量 | 当前训练或推理任务可从[现代矩阵乘教程](../../triton/python/tutorials/03-matrix-multiplication.py)另选对照和计时条件 |

## §6.2 与 Figure 9：把卷积窗口当作乘法的一行（PDF 第 7 页）

第 1 句介绍算子背景。接下来看图 9(a) 的数据、过滤器、输出；再看图 9(b)，把每个输出位置对应的窗口展成一行，过滤器展成矩阵的列。若输入只有一个通道、窗口是 2×2，一个窗口就在乘法里贡献四个乘积。implicit GEMM 在 kernel 内按坐标现取窗口里的值，省去预先写出整张展开矩阵。shift 卷积在取输入时先为各通道改变坐标，然后对通道做点积。末句引到 Listing 8 的偏移表。

[旧卷积的 forward](../../triton-legacy/python/triton/nn/conv.py#L20) 先计算输出高、宽并分配张量，然后调用 `einsum(..., output=output, values={...})`。这里 `output=output` 左边是接口的参数名，右边是刚分配的张量变量；`values` 是按名字保存参数值的 Python 字典。表达式中的 `stride_h`、`pad_h` 是索引中的名字，字典提供这一调用中对应的值。比如没有 padding、`stride_h=2` 时，输出第 1 行、窗口第 0 或第 1 行，输入行分别为 `1*2+0=2` 和 `1*2+1=3`。所以源码里的 `h*stride_h + r - pad_h` 就是在确定从输入哪一行取数。

再用其中一条语句连接索引和步长。[symbolic_delta](../../triton-legacy/python/triton/ops/einsum.py#L363) 建立一个 `nexts` 字典，将归约轴名字换成相应的下一步名字。`symbols[i].subs(nexts)` 对第 i 个坐标表达式执行这个替换。以输入行表达式 `h*2+r` 为例，若归约轴里有 `r`，替换后为 `h*2+nextr`；两式相减，只剩 `nextr-r`。代码再乘 `strides[i]`，即地址在该输入轴移动一格要跨的元素数。比如 `r=0`、`nextr=1`、行跨度为 4 项，表达式的这部分给地址增量贡献 4。循环把输入各轴的贡献加起来，才能得出完整地址差。这里的 `strides[i]` 是张量的存储步长，与卷积表达式里让窗口移动的 `stride_h` 是两个不同的量。

### §6.2.1、Table 1 与 Figure 10（PDF 第 7–8 页）

- 第 1 段指向任务表。区分输入 H/W、通道 C、批大小 B、输出通道 K 与窗口 R/S。输出大小还受 stride 和 padding（边界补齐）影响，追[旧 conv.py](../../triton-legacy/python/triton/nn/conv.py)参数时要一起看。
- 第 2 段先报告对 cuDNN 的 IMPLICIT_GEMM 路径的比较，再提出为什么 ResNet 结果可能如此，最后指出 DeepSpeech2 的差异。图的基线是表明了算法名的特定路径；论文对维护资源的解释属于推测。仓库中的[卷积示例](../../triton-legacy/python/examples/einsum.py#L95)帮助理解生成和对照逻辑，其中二维配置列表在该快照被注释。

<a id="listing-8"></a>
### §6.2.2、Listing 8 与 Figure 11（PDF 第 8 页）

这一段依次给任务、基线和结论。先看任务 1–5 如何换成 shift 卷积，再看显式 shift 后调用库的参照，以及跳过 shift 时的 1×1 卷积参照。Figure 11 的问题是融合版本离「没有 shift 成本」还有多远；这是论文给定任务上的结果。v0.1 的[示例](../../triton-legacy/python/examples/einsum.py#L135)也有独立 CUDA shift 后调用卷积的人为对照和一个索引内包含 `sh[c]`、`sw[c]` 的 einsum 路径，示例的 shift 形状在该快照被注释。

| Listing 8 的顺序 | 每组操作的含义 | v0.1 中可跟的机制 |
|---|---|---|
| 选 TM/TN/TK，填 delta 表 | 每个通道有自己的输入平面和位移。元素偏移由通道基址、行位移乘行宽、列位移相加 | 对应机制见 [`symbolic_delta` 与 `make_delta`](../../triton-legacy/python/triton/ops/einsum.py#L363)；这里生成的是地址增量，先确认表项表示绝对偏移还是相邻迭代的差 |
| 准备坐标、累加块、数据/权重/表的起始指针 | 输出位置、输出通道、归约通道是不同方向。表指针索引归约通道，取出对应偏移 | [源码生成器](../../triton-legacy/python/triton/ops/einsum.py#L181)找 `padelta`，跟 `*padelta` 变成 `incda`。模式选择见 `lut_mode`，该函数中选择 CONSTANT 的语句被注释 |
| 每轮读取偏移、取得输入/权重、更新累加，再推进 | 如某通道要横移一格，输入地址就按该通道偏移取值；乘加后表与权重指针前进到下一个 K 块 | 在[循环生成代码](../../triton-legacy/python/triton/ops/einsum.py#L220)追指针更新，再结合[旧 dot 源码](../../triton-legacy/tests/common/src/dot.h)理解累加与 K 循环 |
| 重建输出地址，计算两个方向的边界，再写回 | shift 改变输入取数位置，结果仍写到原输出坐标。检查两轴能避免写到尾块之外 | 在旧 dot 的 `offc`、`checkc`、masked store 追相同的写回职责 |

### 用两个通道追一次 shift 的地址

先把[示例表达式](../../triton-legacy/python/examples/einsum.py#L174) `nc(h + sh[c])(w + sw[c]),ck->nkhw` 拆开。逗号左边是输入 A 的下标，右边是权重的下标，箭头后是输出的下标。输入和权重都出现了 c，输出里没有 c，因此把所有通道的乘积加起来。式子里的 `sh[c]`、`sw[c]` 按通道查各自的位移。这里的 `k` 是输出通道，别和 GEMM 常用来表示归约长度的 K 混淆。

例如输入宽四格，取输出位置 `(h,w)=(1,1)`。假设通道 0 的偏移是 `(0,1)`，通道 1 是 `(1,0)`。这个输出的两个输入分别从通道 0 的 `(1,2)`、通道 1 的 `(2,1)` 读取。若两个读出来的值是 5、7，所选输出通道的两个权重是 2、3，输出是 `5*2 + 7*3 = 31`。**结果仍写给 (1,1)**。这个例子演示合法坐标内的取数和累加，边界要以被读的 kernel 合同为准。

现在读两个仓库分支：

1. **显式 shift 的参照。** 示例里的 `shift_conv(a, b, **kwargs)` 接收输入、权重以及按名字传的额外值。用 `kwargs['sh']`、`kwargs['sw']` 从字典按键读取对应的位移数组，`sh` 对应高度，`sw` 对应宽度。`shift.apply` 先生成位移后的输入，再把权重改成 1×1 形状交给 `conv2d`。[星号和字典语法](../chapters/00-foundations.md#python-launch-syntax)可回查。
2. **在取数时完成位移。** 示例末尾把表达式与位移数组交给 `triton.ops.einsum`。沿 [`make_delta`](../../triton-legacy/python/triton/ops/einsum.py#L379) 找到 `k` 和 `nextk`，它们记录当前与下次要访问的归约位置。函数把归约位置拆回各个轴坐标，再用符号表达式求两次地址的差。可先用两个假地址 10 和 25 理解，表里记录差 15，就可以在原指针上加 15 找到下次地址。
3. **从表项到 kernel。** [生成器](../../triton-legacy/python/triton/ops/einsum.py#L181) 用 `src += f"""..."""` 把多行文字拼到已有源码末尾。Python 先替换花括号里的 `spec` 和 `cast`；在 DRAM 路径里两者都是空字符串，因此交给 parser 的主体是 `int *padelta[TK] = AD + offadelta;`。`AD` 指向增量表，`offadelta` 是 TK 个表下标。若 `off_k=0`、TK=2，就得到指向表第 0、1 项的两个地址。下一行 `(*padelta)[newaxis, :, newaxis]` 先读出两个增量，再把形状由 `[2]` 变为 `[1,2,1]`，然后可广播到 M、K、批次对齐的三维块。后续 `pa += incda` 用这组增量推进数据地址，`padelta += TK` 则让表指针去下一组表项。论文清单的 `pxa + delta` 是从基址加偏移，v0.1 这里展示的是沿地址增量更新的路径。

如果只想确认理解，可以把通道 1 的位移改成零，重新写两个输入坐标。变的是其中一个输入地址，输出位置仍为 `(1,1)`。

<a id="read-shift-call-chain"></a>
### 从调用追到生成的 kernel，在哪里停下来

此前的两通道例子回答了位移地址的数学问题。现在要回答源码问题：**谁把 `sh`、`sw` 交给生成器，生成出来的源码或参数又交给谁？** 从 [示例末尾](../../triton-legacy/python/examples/einsum.py#L191)的 `triton.ops.einsum(expr, ta, tb, tc, arrays=arrays, bench=True)` 开始。先认实参：`expr` 是索引表达式的文字，`ta/tb` 是输入张量，`tc` 是预先准备的输出，`arrays` 给表达式里的名字提供数据。

| 当前在哪 | 这一站只读这些操作 | 什么时候往下，什么时候返回 |
|---|---|---|
| [ops/__init__.py](../../triton-legacy/python/triton/ops/__init__.py#L1) | `from .einsum import _einsum, einsum` 说明公共的 `einsum` 从同一个包内的 `einsum.py` 导出 | 当前只追 `einsum`，继续去对应定义。文件里另一个 `batchnorm` 导入和这次调用无关 |
| [einsum.py 的公共函数](../../triton-legacy/python/triton/ops/einsum.py#L700) | 最终调用 `_einsum.apply(...)`。在本例要找前向计算，就去同一文件的 [`_einsum.forward`](../../triton-legacy/python/triton/ops/einsum.py#L644) | `forward` 先按表达式、张量类型、shape 和 stride 等查现有实例。第一次生成 kernel 的问题走 `key not in cache` 的分支，进入 `_einsum.instance` 的构造 |
| [instance 中准备生成参数的位置](../../triton-legacy/python/triton/ops/einsum.py#L539) | `[(x, arrays[x]) for x in subscripted]` 依照用到的名称拿数据。若 `x` 是 `"sh"`，就把名称与位移数组配成一对。接着 `make_delta` 计算增量并返回使用方式。两份结果都交给进一步的 kernel 构造 | 想确认增量表怎样取数时才回到上一节的 `make_delta` 走读。确认 `delta_a` 与 `lut_mode_a` 各是什么后，回到 `_einsum.make_kernel` 的调用 |
| [生成函数开头](../../triton-legacy/python/triton/ops/einsum.py#L72) 和[末尾](../../triton-legacy/python/triton/ops/einsum.py#L339) | 开头 `src = ""` 准备空字符串，中间逐段拼进 Triton-C，末尾 `triton.kernel(src, defines=defines)` 把源码和宏配置交给运行时 | `padelta` 指向增量表，`*padelta` 取增量，`padelta += TK` 切到下一组表项，上面的两通道示例已展开这一分支。继续追编译时可沿[旧运行时入口](00-how-to-read.md) |
| [instance 的参数与 run](../../triton-legacy/python/triton/ops/einsum.py#L591) | 若表用 DRAM 模式，构造函数将增量表转成 GPU 张量并加入 `self.args`；`run` 填进去这次调用的 a、b、c，并调用 `self.kernel(*self.args, ...)` | 到这里可以回答谁把参数交给谁。想追设备执行，回到本文引用的生成代码里找从表地址读增量的语句 |

最后试着合上除了示例以外的文件，口头复述一遍：「示例交给 Python 的是表达式和数组。Python 根据索引的变化算出增量，准备 kernel 源码与参数。设备执行时读增量来推进输入地址。」如果缺的是某一步从哪行发生，就只回到那一站查。若你已能解释论文里的查表为何在这里表现为这样的指针更新，可以回到 Listing 8 和 Figure 11。

## §7–§8（PDF 第 8 页）

§7 首段依次回顾三件事：提出系统、解释扩充 IR 如何支持优化、用 Triton-C 表达新算子。可分别回到 [Figure 2](01-language.md)、[§4 的 IR](02-ir.md)和[本篇 Listing 8](#listing-8)检查证据。次段把 Tensor Core、量化与更高层 DSL 集成列为未来方向；对照今天的实现时，可从[现代低精度与 MMA](../chapters/10-modern-extensions.md#sec-10-3)开始。§8 是资助致谢；后面的参考文献用来追方法来源。

读完后试着从结果逆向走一遍：选 Figure 11 一组，找到它的比较路径，写下 fused 版本每个输出位置从哪里取输入，再回到源码找到决定那个地址的变量。
