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

沿[旧卷积实现](../../triton-legacy/python/triton/nn/conv.py#L6)的 `_conv2d.forward` 找 `einsum` 调用，再沿[旧 einsum 实现](../../triton-legacy/python/triton/ops/einsum.py#L363)的 `symbolic_delta`、`make_delta` 追索引表达式如何得到地址增量。在表达式 `h*2+r` 中，`h` 是输出坐标，2 是 stride，`r` 是过滤器窗口内的坐标。三者合起来才选到真实的输入行。

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

先把[示例表达式](../../triton-legacy/python/examples/einsum.py#L166) `nc(h + sh[c])(w + sw[c]),ck->nkhw` 拆开。逗号左边是输入 A 的下标，右边是权重的下标，箭头后是输出的下标。输入和权重都出现了 c，输出里没有 c，因此把所有通道的乘积加起来。式子里的 `sh[c]`、`sw[c]` 按通道查各自的位移。这里的 `k` 是输出通道，别和 GEMM 常用来表示归约长度的 K 混淆。

例如输入宽四格，取输出位置 `(h,w)=(1,1)`。假设通道 0 的偏移是 `(0,1)`，通道 1 是 `(1,0)`。这个输出的两个输入分别从通道 0 的 `(1,2)`、通道 1 的 `(2,1)` 读取。若两个读出来的值是 5、7，所选输出通道的两个权重是 2、3，输出是 `5*2 + 7*3 = 31`。**结果仍写给 (1,1)**。这个例子演示合法坐标内的取数和累加，边界要以被读的 kernel 合同为准。

现在读两个仓库分支：

1. **显式 shift 的参照。** 示例里的 `shift_conv(a, b, **kwargs)` 接收输入、权重以及按名字传的额外值。用 `kwargs['sh']`、`kwargs['sw']` 从字典按键读取对应的位移数组，`sh` 对应高度，`sw` 对应宽度。`shift.apply` 先生成位移后的输入，再把权重改成 1×1 形状交给 `conv2d`。[星号和字典语法](../chapters/00-foundations.md#python-launch-syntax)可回查。
2. **在取数时完成位移。** 示例末尾把表达式与位移数组交给 `triton.ops.einsum`。沿 [`make_delta`](../../triton-legacy/python/triton/ops/einsum.py#L379) 找到 `k` 和 `nextk`，它们记录当前与下次要访问的归约位置。函数把归约位置拆回各个轴坐标，再用符号表达式求两次地址的差。可先用两个假地址 10 和 25 理解，表里记录差 15，就可以在原指针上加 15 找到下次地址。
3. **从表项到 kernel。** 从同文件的[生成器](../../triton-legacy/python/triton/ops/einsum.py#L181)找 `*padelta`。这里生成的 Triton-C 会沿表指针读取增量，随后在循环中更新输入指针。结合 Listing 8 的 `pxa + delta` 来看，论文清单按基址加选中偏移写，v0.1 生成器有按前后两次访问的增量推进的路径。Figure 11 里的融合收益要回到论文测量条件理解。

如果只想确认理解，可以把通道 1 的位移改成零，重新写两个输入坐标。变的是其中一个输入地址，输出位置仍为 `(1,1)`。

## §7–§8（PDF 第 8 页）

§7 首段依次回顾三件事：提出系统、解释扩充 IR 如何支持优化、用 Triton-C 表达新算子。可分别回到 [Figure 2](01-language.md)、[§4 的 IR](02-ir.md)和[本篇 Listing 8](#listing-8)检查证据。次段把 Tensor Core、量化与更高层 DSL 集成列为未来方向；对照今天的实现时，可从[现代低精度与 MMA](../chapters/10-modern-extensions.md#sec-10-3)开始。§8 是资助致谢；后面的参考文献用来追方法来源。

读完后试着从结果逆向走一遍：选 Figure 11 一组，找到它的比较路径，写下 fused 版本每个输出位置从哪里取输入，再回到源码找到决定那个地址的变量。
