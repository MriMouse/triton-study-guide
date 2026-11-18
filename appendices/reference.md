# 附录：论文定位、术语与练习

[返回总目录](../README.md) · [上一章：两代对照](../chapters/11-two-generations.md)

这里集中三样查阅材料，正文不要求按顺序读完：

1. **论文定位表**：2019 论文的节号、Figure/Listing 和 PDF 页序，对应到历史源码与现代实现；
2. **术语速查**：遇到陌生名词时恢复阅读上下文；
3. **配套练习与运行边界**：只用 Python 标准库就能做的索引/数学检查，以及将来运行 GPU 时的资源与版本核查。

**使用建议：** 读正文时遇到“论文 §5.2.2”“TTGIR”“RAW/WAR”这类词，先查本附录，再回到正文看具体代码。附录只做索引和练习，不替代正文的对象流带读。

<a id="sec-a-1"></a>
## A.1 论文、历史源码与现代实现定位表

按顺序跟读可进入[论文逐段带读](../paper-reading/README.md)。

核对原文：Philippe Tillet、H. T. Kung、David Cox，*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*，MAPL 2019，[作者站 PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)，[DOI](https://doi.org/10.1145/3315508.3329973)。下表页码是 **PDF 文件从 1 开始的页序**，不是某个会议合订本的印刷页码。共 10 页。

这里提供节号和阅读问题，不复制整篇论文。图和 Listing 请打开原 PDF 对照；`Listing` 指论文里的编号代码清单。

### 怎样使用这张表

“论文讨论”只陈述原文主题。“历史入口”是目前固定的 **2020-05-04 v0.1** 源码，它保留旧架构，但不能证明与论文版本一致。“现代入口”则用于比较今天的设计，**不表示旧 pass 的代码仍原样存在**。

| 论文位置 | PDF 页 | 阅读问题 | 历史入口 | 现代入口与边界 |
|---|---|---|---|---|
| §1 Introduction，Figure 1 | 1–2，图在 1 | 为什么需要自定义 kernel？怎样用 Roofline 理解结果？ | [README](../../triton-legacy/README.md) | [入门导读](../chapters/00-foundations.md#sec-0-1)；历史性能图不能当作当前实测 |
| Figure 2 系统总览 | 2 | 语言、IR、JIT 与调优器怎样分工？ | [function.cc](../../triton-legacy/lib/runtime/function.cc) | [现代编译链](../chapters/04-vector-add-jit.md#sec-4-2)；现代前端和 IR 已变化 |
| §2 Related Work，Listing 2 | 2–3 | Triton 的抽象层次与当时其他 DSL 有什么不同？ | [旧说明](../../triton-legacy/README.md) | 比较必须限定年代，不能当作今天 TVM/XLA 的完整能力评价 |
| §3.1 Syntax，Listing 3 | 3 | tile、range、broadcast、predication 怎样写？ | [parser.cc](../../triton-legacy/lib/lang/parser.cc) | [core.py](../../triton/python/triton/language/core.py)；今天用 Python 风格接口，旧语法已不同 |
| §3.2.1 Tile semantics | 3 | 为什么保留成块的操作？ | [旧 type.h](../../triton-legacy/include/triton/ir/type.h) | [4.1向量加法](../chapters/04-vector-add-jit.md#sec-4-1)；逻辑值不是单个硬件线程 |
| §3.2.2 Broadcasting，Listing 4 | 3–4，Listing 在 4 | 不同 shape 怎样参与同一计算？ | [code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) | [semantic.py](../../triton/python/triton/language/semantic.py)；对应广播思想而非语法相同 |
| §3.3 Programming Model，Figure 3 | 4 | 源程序的一块工作怎样对应并行执行？ | [旧 CUDA 对照](../../triton-legacy/docs/tutorials/triton-vs-cuda.rst) | [GPU 基础](../chapters/00-foundations.md#sec-0-4)；旧文“single-threaded”不是只有一个物理线程 |
| Listing 1 矩阵乘 | 2 | tile 索引、指针广播、mask 与累加怎样配合？ | [旧 dot benchmark](../../triton-legacy/tests/bench/dot.cc) | [现代 GEMM 教程](../../triton/python/tutorials/03-matrix-multiplication.py)；后续 GEMM 章详细对照 |
| §4 The Triton IR，Listing 5 | 4–5，Listing 在 4 | 怎样保留 tile 运算供优化？ | [旧 IR 头文件](../../triton-legacy/include/triton/ir) | [现代 Triton dialect](../../triton/include/triton/Dialect/Triton/IR)；旧文本不是现代 TTIR |
| §4.1 Structure | 4–5 | module/function/basic block/SSA 怎样组织程序？ | [basic_block.h](../../triton-legacy/include/triton/ir/basic_block.h) | [AST code generator](../../triton/python/triton/compiler/code_generator.py)；共同概念不意味着共用类型 |
| §4.2 Data-Flow，Figure 4 | 5 | shape、reshape、broadcast、dot 怎样显式表达？ | [instructions.h](../../triton-legacy/include/triton/ir/instructions.h) | [TritonOps.td](../../triton/include/triton/Dialect/Triton/IR/TritonOps.td)；逐项比对操作和类型 |
| §4.3 Control-Flow，Listing 6 | 5 | 怎样表达 tile 内逐元素条件？ | [旧 masked load 构造](../../triton-legacy/lib/lang/code_gen.cc) | [load/store API](../../triton/python/triton/language/core.py)；PSSA/psi 尚未在此旧快照核实，不能说 masked load 就是 psi |
| §5 The Triton-JIT compiler | 5–7 | 从逻辑计算到目标代码需要哪些决定？ | [make_bin](../../triton-legacy/lib/runtime/function.cc) | [NVIDIA pipeline](../../triton/third_party/nvidia/backend/compiler.py)；对照职责而非同名类 |
| §5.1.1 Pre-Fetching，Listing 7 | 5 | 怎样提前加载，隐藏循环中的访存延迟？ | [旧 dot 的手写预取](../../triton-legacy/tests/common/src/dot.h)，自动变换尚未核实 | [现代流水线目录](../../triton/lib/Dialect/TritonGPU/Transforms/Pipeliner)；现代异步流水线不是论文原实现 |
| §5.1.2 Tile-Level Peephole | 6 | 局部等价变换为何需要 tile 语义？ | [peephole.cc](../../triton-legacy/lib/codegen/transform/peephole.cc) | [现代 Triton transforms](../../triton/lib/Dialect/Triton/Transforms)；需选具体模式验证 |
| §5.2.1 Hierarchical Tiling，Figure 5 | 6 | 逻辑 tile 怎样匹配计算和存储层次？ | [layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc) | [TritonGPU transforms](../../triton/lib/Dialect/TritonGPU/Transforms)；现代 layout 系统已经发展 |
| §5.2.2 Memory Coalescing，Figure 6 | 6 | 相邻线程的地址怎样影响事务利用率？ | [coalesce.cc](../../triton-legacy/lib/codegen/transform/coalesce.cc) | [Coalesce.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)；同名问题不证明算法相同 |
| §5.2.3 Shared Memory Allocation，Figure 7 | 6 | 哪些数据存放片上，生命周期怎样影响分配？ | [allocation.cc](../../triton-legacy/lib/codegen/analysis/allocation.cc) | [Allocation.cpp](../../triton/lib/Analysis/Allocation.cpp)；现代别名、descriptor 与存储层次更多 |
| §5.2.4 Shared Memory Synchronization | 6 | RAW/WAR 依赖怎样要求插入同步？ | [membar.cc](../../triton-legacy/lib/codegen/transform/membar.cc) | [Membar.cpp](../../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Membar.cpp)；现代 async wait/barrier 需额外语义 |
| §5.3 Auto-tuner | 7 | 哪些参数被搜索？搜索与编译的关系？ | [precompile 与选项](../../triton-legacy/lib/runtime/function.cc) | [autotuner.py](../../triton/python/triton/runtime/autotuner.py)；现代用户配置搜索不是论文空间的原样延续 |
| §6.1 Matrix Multiplication，Figure 8 | 7 | 输入 shape 与对照环境是什么？ | [bench](../../triton-legacy/tests/bench) | [现代 GEMM](../../triton/python/tutorials/03-matrix-multiplication.py)；后续另定当前验证，不复用旧倍数 |
| §6.2 Convolutions、§6.2.1 Dense、§6.2.2 Shift | 7–8 | 系统如何处理常规与新型算子？ | [历史卷积示例与查表增量](../paper-reading/04-experiments.md#listing-8)，实验配置仍需逐项核对 | 用于理解原论文动机；本书主练习按当前 LLM 算子安排 |
| §7 Conclusions | 8 | 哪些是已实现内容，哪些是未来方向？ | 与所选快照逐项核验 | Tensor Core、现代硬件特性、Gluon 等需另读今天的来源 |

### 目前需要特别保留的三个差异

1. **Triton-C 已不再是现代主线前端。** 学它是为了理解原论文的表达方式与前端工作；实际新 kernel 先学经典 `tl`，再学 Gluon。
2. **旧 Triton-IR 不是 TTIR 的旧拼写。** 两者的基础设施、语法和指令体系不同；SSA、tile、数据流是可迁移的理解。
3. **2019 论文与 2020 快照有待追溯的差距。** 尤其 prefetch/PSSA，当前没有足够证据写“原论文实现已完整保存”。具体机制及版本差异见[论文 IR 篇](../paper-reading/02-ir.md)与[编译篇](../paper-reading/03-compiler.md)。

未来新增章节时，在相应段落标注具体节号，而不是只在参考文献中笼统列论文。若你实际阅读的是另一篇后续 Triton 论文，这张表继续保留为 2019 基础路线，再增加独立索引，不能混用节号。

<a id="sec-a-2"></a>
## A.2 术语速查

| 术语 | 本书中的简明解释 | 首读 |
|---|---|---|
| DNN / LLM | 深度神经网络 / 大语言模型；LLM 是本书的应用背景 | [0.1](../chapters/00-foundations.md#sec-0-1) |
| operator / kernel | 算子规定数学计算；kernel 是设备执行程序，二者不必一一对应 | [0.1](../chapters/00-foundations.md#sec-0-1) |
| DSL | 领域专用语言，针对特定计算领域表达程序 | [0.1](../chapters/00-foundations.md#sec-0-1) |
| tile / block | 一组逻辑元素；注意与 CUDA thread block 区分 | [0.3](../chapters/00-foundations.md#sec-0-3) |
| program instance | kernel 的一个逻辑实例，拥有自己的 program id | [4.1](../chapters/04-vector-add-jit.md#sec-4-1) |
| grid | 启动的 program/thread-block 组织，含一个至三个维度 | [4.1](../chapters/04-vector-add-jit.md#sec-4-1) |
| CTA | Cooperative Thread Array，通常即 CUDA thread block | [0.4](../chapters/00-foundations.md#sec-0-4) |
| warp / lane | NVIDIA 的 32 线程执行组 / 组内位置 | [0.4](../chapters/00-foundations.md#sec-0-4) |
| stride | 某一维前进一步跨过的元素数，不是字节数 | [0.3](../chapters/00-foundations.md#sec-0-3) |
| mask / predication | 用逐元素条件控制某些操作；具体语义依操作定义 | [4.1](../chapters/04-vector-add-jit.md#sec-4-1) |
| coalescing | 让一组线程的访存更有效利用内存事务 | [0.4](../chapters/00-foundations.md#sec-0-4) |
| reduction | 归约，把多个元素按求和、最大值等运算合并 | [6.1](../chapters/06-classic-kernels.md#sec-6-1) |
| GEMM | 通用矩阵乘法，常见核心计算为 C=AB，接口也可能含缩放和累加 | [6.2](../chapters/06-classic-kernels.md#sec-6-2) |
| AST | 抽象语法树，描述源码语法结构 | [5.1](../chapters/05-frontend-ir.md#sec-5-1) |
| IR / dialect | 中间表示 / MLIR 中的一组相关操作、类型与规则 | [5.2](../chapters/05-frontend-ir.md#sec-5-2) |
| SSA | 静态单赋值形式，IR 的一个值只定义一次，便于追踪数据依赖 | [5.2](../chapters/05-frontend-ir.md#sec-5-2) |
| pass | 一次分析或变换步骤，不一定改写程序 | [7.2](../chapters/07-layout-passes-memory.md#sec-7-2) |
| lowering | 降低抽象层次，把语义细化为更接近实现的表达 | [8.1](../chapters/08-backend-debug.md#sec-8-1) |
| JIT / specialization | 按需编译 / 针对已知条件生成特定版本 | [4.2](../chapters/04-vector-add-jit.md#sec-4-2) |
| constexpr | 编译时已知的值，可用于形状、分支等决策 | [4.1](../chapters/04-vector-add-jit.md#sec-4-1) |
| layout | 逻辑元素到线程、寄存器或存储结构的分布方式 | [7.1](../chapters/07-layout-passes-memory.md#sec-7-1) |
| TTIR / TTGIR | 现代经典 Triton 的逻辑张量 IR / 带 GPU 实现信息的 IR | [5.2](../chapters/05-frontend-ir.md#sec-5-2) |
| PTX / cubin / SASS | NVIDIA 虚拟指令文本 / GPU 二进制产物 / 实际机器指令的汇编表示 | [8.1](../chapters/08-backend-debug.md#sec-8-1) |
| RAW / WAR | 写后读 / 读后写的依赖，影响能否安全重排或复用存储 | [7.3](../chapters/07-layout-passes-memory.md#sec-7-3) |
| prefill / decode | 处理输入序列 / 在已有上下文上继续生成 token | [9.4](../chapters/09-llm-kernels.md#sec-9-4) |
| KV cache | 缓存先前 token 的 key/value，供后续 Attention 复用 | [9.4](../chapters/09-llm-kernels.md#sec-9-4) |

精确定义以具体章节与相应 API/硬件语义为准，速查表只用于恢复阅读上下文。

<a id="sec-a-3"></a>
## A.3 配套练习与运行边界

本书正文是源码带读，纸笔数值和教学改写不假称 GPU 执行结果。你可以先完成下面标准库练习，再按 8.2 节准备实际环境。路径以三个并列目录为基础。

### 1. 不安装任何深度学习库，先核对索引

从手册父目录执行：

```bash
python3 triton-study-guide/tools/trace_vector_add.py
python3 triton-study-guide/tools/trace_tensor_math.py
```

第一个脚本沿[4.1](../chapters/04-vector-add-jit.md#sec-4-1)打印三个program的offset/mask/load/add与最终写入次数。它使用Python列表和显式补零，不预测原教程未定义位置的实际数值。

第二个脚本对应[0.3](../chapters/00-foundations.md#sec-0-3)、[6.1](../chapters/06-classic-kernels.md#sec-6-1)、[6.2](../chapters/06-classic-kernels.md#sec-6-2)和[Attention](../chapters/09-llm-kernels.md#sec-9-2)的数学：

- 坐标(1,2)、stride=(3,1)得到5元素/20字节偏移。
- softmax的平移不变性，用稳定指数计算核对。
- 2×3和3×2矩阵相乘得到显式整数参考值。
- 将一行score分成两块，用在线max/分母/加权分子更新，对照完整softmax的加权和。

它使用Python双精度math，与GPU近似指数/低精度累加有不同误差行为。通过数学练习不表示GPU实现、layout、并发或性能已经验证。

### 2. 阅读时自己做三张表

**对象表：** 名字、类型/shape、由哪行创建、交给哪个调用。host张量、指针、编译期IR value分开写。

**地址表：** 逻辑坐标、元素偏移、字节地址、mask。遇到stride或广播，先算这张表再谈吞吐。

**调用表：** 调用者文件/函数、实参与形参、返回对象、下一步。对visitor与注册接口，注明实际连接方式，不靠同名猜测。

正文都提供了至少一条具体示范；表格是帮助你检查理解，不要求你脱离讲解自行补完所有代码。

### 3. 运行 GPU 示例时先确认真正导入了什么

本书部分章节附有实际运行记录，它们会标明设备、Triton 版本、源码版本和命令。换一个环境复现时，仍要先确认实际导入的 Triton 路径与版本：

```bash
python3 -c 'import torch, triton; print(torch.__version__); print(triton.__version__); print(triton.__file__)'
```

源码可见不等于已构建安装；版本字符串相同也不保证是同一个 commit。不要直接 import 上游教程来“只看定义”：它们可能激活设备并在尾部执行大范围测试或 benchmark。真正动手时提取最小例子，先做小 shape 正确性，再考虑计时和配置搜索。

### 4. 怎样判断一次正确性验证足够具体

至少记录输入shape/dtype/stride/device、输入值范围/随机种子、目标公式、参考实现、误差标准、边界样例，以及实际导入版本。NaN/Inf、空维、别名/并发与非连续输入是否支持应由接口合同决定，不默默混入未覆盖情形。

随后才区分编译时间、分配时间、host提交和设备执行时间；同一份kernel测量不自动代表模型请求延迟。具体方法与源码在[8.2](../chapters/08-backend-debug.md#sec-8-2)。
