# §5：沿着编译顺序读优化

[上一篇：IR](02-ir.md) · [路线目录](README.md) · [下一篇：实验](04-experiments.md)

## §5 开头（PDF 第 5 页）

把开头的两类优化和调优分别放到[旧 function.cc](../../triton-legacy/lib/runtime/function.cc#L199)。`make_bin` 收到一个已经生成的 IR 模块，先创建 pass（对这个模块做一次分析或改写的编译步骤），再依次执行。读到 `align.run`、`layouts.run` 时查看它们算出的信息由谁消费，读到 `peephole.run`、`coalesce.run` 时查看它们改写了什么。后半段根据布局计算存活区间、分配共享内存、插入同步，然后 `isel.visit` 把旧 IR 变成 LLVM 模块。真实顺序见[第 2 章](../chapters/02-legacy-ir-passes.md#sec-2-2)与[第 3 章](../chapters/03-legacy-layout-backend.md#sec-3-2)。现代 NVIDIA 对照入口是 [`make_ttir`、`make_ttgir`、`make_llir`](../../triton/third_party/nvidia/backend/compiler.py#L292)。

## §5.1 Machine-Independent Passes（PDF 第 5–6 页）

### §5.1.1 与 Listing 7：什么时候准备下一轮的值

首句说循环刚开始便 load，后续计算往往要等 load 返回。第二句提出识别循环并把准备数据的时间提前。看 Listing 7 左右两部分：

| 读到哪里 | 左右两侧各做什么 | 可跟读的代码 |
|---|---|---|
| 进入 B1 前的 B0 | 左边只算初始地址；右边在 B0 已经准备第一个 load 值 `x0` | [旧 dot kernel](../../triton-legacy/tests/common/src/dot.h)中 prologue 先构造首轮指针，再加载 a、b |
| B1 开头 | 左边每轮从当前指针读；右边用 phi 选取首轮预载或上一轮准备的值 | 旧 kernel 的循环开头先消费 a、b；[旧模块](../../triton-legacy/include/triton/ir/module.h)与[VisitForStmt](../../triton-legacy/lib/lang/code_gen.cc#L406)提供理解控制流的入口 |
| B1 末尾 | 先决定下一轮地址，再安排后续 load。判断这种移动的正确性要检查最后一轮边界和指针依赖 | 旧 kernel 在累加后推进 `pa/pb` 再条件加载，条件 `k > TK` 决定下一轮是否存在。印刷清单最后的 load 写成 `%p`；判断下一轮取谁时按真实源码的更新后指针追 |

v0.1 这里看到的是手写软件预取，`make_bin` 的 pass 列表尚未核实论文所示的自动改写。现代[流水线目录](../../triton/lib/Dialect/TritonGPU/Transforms/Pipeliner)处理相近的延迟隐藏问题。可以从 [NVIDIA 编译流程](../../triton/third_party/nvidia/backend/compiler.py#L308)寻找调度和 pipeline 的调用，再追具体变换。

### §5.1.2：为什么知道 shape 就能删操作

作者用「转置两次回到原块」说明局部化简，并把其他代数关系列为可继续探索的方向。v0.1 有一个可完整追踪的不同例子。[旧 peephole.cc](../../triton-legacy/lib/codegen/transform/peephole.cc#L90)查找归约轴长度为 1 的情况。比如 `[[7],[9]]` 沿列求和，答案仍为 7 和 9，只少了一个长度为 1 的轴。看代码如何检查形状、创建 reshape、替换用户的旧操作数，再看 `make_bin` 后续怎样跑 DCE（清理无用指令）。现代逐条追改写见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-2)。

## §5.2 Machine-Dependent Passes（PDF 第 6 页）

开头列出四项工作，对应下面四小节。读 v0.1 时从 `make_bin` 看前一项产出的信息怎样供下一项使用。

### §5.2.1 与 Figure 5：一个逻辑块怎样分给硬件

第一句将一大块分为小块再分为更小块。用图从大到小追「设备中的多个程序实例 → 实例内部协作的执行单元 → 更细的计算与寄存器分担」。第二句说明作者希望从 IR 中枚举合法组合。v0.1 [layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc#L388)先按相关轴组织值，再给组选择布局；回查同文件的 scanline 和 MMA 构造，圈出 shape、连续维与 warp 配置从哪里进入。选一种布局和枚举全部布局是两个动作，论文的枚举范围要回看 §5.3。现代显式坐标例子见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-1)。

#### 先算出一组线程，再问它们访问什么

在 NVIDIA 这段旧代码里，一个 warp 是 32 个线程，lane 指某个线程在组内的编号。layout 描述的是逻辑元素交给谁处理。例如元素 12 对应的数组地址可以一直保持不变，布局却可以把这个元素交给别的 lane。

打开 [scanline_layout 构造函数](../../triton-legacy/lib/codegen/analysis/layout.cc#L167)并设一维形状为 `[128]`，`num_warps=1`，最连续轴为 0，连续访问分析允许每线程在该轴处理 4 项。顺着三行计算：

1. `size` 是各轴长度的乘积，得到 128。`num_threads = num_warps * 32` 得到 32。
2. `nts_[i] = clamp(size / num_threads, ...)` 计算每线程在这根轴上的连续元素数，取 128/32=4，并将它限制在源码给出的上下界之间。这里 `i=0`，上下界为 1 和 4。
3. `mts_[i] = clamp(num_threads, 1, shape_[i] / nts_[i])` 计算这根轴上用多少个线程位置，取 32。源代码还会把已分配的规模从 `size` 和 `num_threads` 的剩余配额中除去，以便处理后续轴。

这一步还没发出任何 load。先算出 `nts_`、`mts_` 再读下一节，就能问出「当地址来自转置或带 stride 的视图时，哪一轴让邻近线程真正读到邻近地址？」。

### §5.2.2 与 Figure 6：访问的是一块数据，付费的是一批事务

第一段定义相邻线程同时访问附近地址时更有机会让同一批事务带来的数据被充分使用。第二段说编译器能改变微块内的线程安排。图左读「用到哪些位置」「同时搬来了哪些没用到的位置」，再看右边邻近访问怎样减少浪费。

回到 [coalesce.cc](../../triton-legacy/lib/codegen/transform/coalesce.cc#L64)，先看 `extract_io_use` 如何只找把某个 IR 值当地址使用的 I/O，再看主函数怎样利用 `align` 和 `layouts` 所给的信息。现代 [Coalesce.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)同样从内存操作入手；逐步比较见[第 11 章](../chapters/11-two-generations.md#sec-11-3)。

### §5.2.3 与 Figure 7：什么时候能重用同一段共享内存

前两句说明重复使用的 dot 操作数值得放到便宜且快的共享存储。下一句说应为每块数据决定存多久和放哪里。图的横向是时间，纵向是存储占用。若两块各 4 KiB 且同时存活，就需要为它们留出不同空间；若生命周期错开，就有机会占用同一区间。

依次打开[旧 liveness.cc](../../triton-legacy/lib/codegen/analysis/liveness.cc#L14)与[allocation.cc](../../triton-legacy/lib/codegen/analysis/allocation.cc#L13)。先找到何时最后使用一块数据，再找重叠生命周期与重叠地址如何形成干涉边，最后找 first-fit 着色如何得到各布局的 offset。论文提到的线性时间存储分配与 v0.1 此处图着色属于具体算法差异。布局形成共享数据到使用 offset 的细节见[第 3 章](../chapters/03-legacy-layout-backend.md#sec-3-1)。

### §5.2.4 与数据流公式：哪些读写需要等其他线程

第一句给出机器执行假设。第二句提出编译器插 barrier（让协作线程在继续前等待彼此）。随后两类公式可从[旧 membar.cc](../../triton-legacy/lib/codegen/transform/membar.cc#L84)找对象：

| 公式 | 含义 | 源码追法 |
|---|---|---|
| 两条 in | 当前基本块从各前驱汇总尚需关心的写入和读取 | 读 `run` 如何收集各前驱的记录并调用 `join` |
| RAW（read after write） | 有线程刚写共享区间，另一些线程现在要读重叠部分，须保证数据已经可读 | 在 `transfer` 看已有 `new_written_to` 如何与当前 `read` 判重叠 |
| WAR（write after read） | 有线程还可能在读旧数据，下一批写入要复用同一地址，须先等读者 | 看 `new_read_from` 如何与当前 `written` 比较 |
| 两条 out | 没冲突就扩充记录；发生冲突则在同步处分开前后的访问 | 读 `transfer` 的清除及加入逻辑，再找 `insert_barrier` 如何落到 IR。实际实现以字节区间和代码分支为准 |

例如先向共享地址 0–15 写入，随后另一 warp 读取地址 8–15，覆盖区间就是查找 RAW 的依据。如果后续再复用地址 0–15 写新值，还要检查之前的读是否结束。现代异步与同步协议见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-3)。

### 看源码时模拟一次 transfer

在 [membar::transfer](../../triton-legacy/lib/codegen/transform/membar.cc#L84) 中，`written_to` 和 `read_from` 是扫描当前块之前从前驱带来的记录。前驱就是控制流中有边可以跳到当前块的基本块。`insert_loc` 是待插同步的指令集合。参数里的 `&` 是 C++ 引用，`insert_loc.insert(i)` 会修改调用者传入的集合，`const ... &` 则表示此次通过引用只读。

从两个空记录开始，用包含实际冲突的三个动作读循环。区间以「开始字节、结束字节」记，以下是纸笔推演：

| 指令 | 当前读/写区间 | 源码的判断和更新 |
|---|---|---|
| I0 | 写 [0,16) | `get_written_intervals` 取得该区间；此前没有读，随后 `std::copy` 把它加入写记录 |
| I1 | 读 [8,16) | `intersect(new_written_to, read)` 检出 RAW；把 I1 放进 `insert_loc`，清除两张旧记录，再把 I1 的读取加入读记录 |
| I2 | 写 [4,12) | 仍记得 I1 读过的区间；与本次写入交叠的部分是 [8,12)，因此判 WAR，记录 I2 再重新累计 |

回到 [run](../../triton-legacy/lib/codegen/transform/membar.cc#L117) 看为什么 `transfer` 接收的记录可能并非空：它从各前驱取出记录，经 `join` 合起来。最后另一次循环才调用 `insert_barrier`。这就是论文公式里的集合在 C++ 中变成「两个区间列表、一个遍历和一个插入位置集合」的具体过程。快照还有双缓冲特判，可先追通普通分支，再读 `safe_war`。

## §5.3 Auto-tuner（PDF 第 7 页）

第一段比较调优参数的来源：论文提出从各 pass 汇总可调参数。第二段限定其实验只搜索分层 tiling，不同层在各自的区间内枚举 2 的幂。读 v0.1 的 [`precompile`](../../triton-legacy/lib/runtime/function.cc#L301) 和 [`tests/common/dot.h`](../../triton-legacy/tests/common/dot.h)，找候选宏和 `num_warps` 从哪来，候选如何编译、哪些会被跳过。再搜同一运行时文件的 `autotune`，跟踪执行时间怎样决定所选 caller。同样三问可用于读现代 [autotuner.py](../../triton/python/triton/runtime/autotuner.py) 的 Config、缓存键和实际试跑。比如尝试 B=32 或 64 是两份候选；即便每个配置编译成功，还要测量才知道给定输入下该采用哪一个。


读 `precompile` 时可以在纸上画 2×2 表。假设 `space.num_warps=[4,8]`、`space.defines` 中只有 `TM=[32,64]`。这时 `ranges` 记录 `[2,2]`，`_loop_nest` 对四组下标调用 `do_make`。例如下标 `[1,0]` 会选 warp 数 8、TM=32，随后 `make(stream, opt)` 编译或取该配置的设备程序。`callers_[opt].reset(call)` 把得到的可启动对象存到以选项为键的表里。到 `autotune` 才按参数和候选计时，`precompile` 与 `autotune` 是前后两个动作。这里的 C++ `[&](...){ ... }` 创建一个可交给循环调用的函数，外层需要的名字用引用访问，具体值如 `params` 由每次调用传入。
