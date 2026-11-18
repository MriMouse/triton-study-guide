# §5：沿着编译顺序读优化

[上一篇：IR](02-ir.md) · [路线目录](README.md) · [下一篇：实验](04-experiments.md)

## §5 开头（PDF 第 5 页）

把开头的两类优化和调优分别放到[旧 function.cc](../../triton-legacy/lib/runtime/function.cc#L199)。`make_bin` 收到一个已经生成的 IR 模块，先创建 pass（对这个模块做一次分析或改写的编译步骤），再依次执行。例如 `align.run(module)` 会计算地址值在各轴上有多少连续元素，`coalesce` 读取这些数给内存操作分组。`peephole.run(module)` 则可能插入 reshape，并改掉后面某个 store 所用的输入。后半段根据布局计算存活区间、分配共享内存、插入同步，然后 `isel.visit` 把旧 IR 变成 LLVM 模块。真实顺序见[第 2 章](../chapters/02-legacy-ir-passes.md#sec-2-2)与[第 3 章](../chapters/03-legacy-layout-backend.md#sec-3-2)。现代 NVIDIA 对照入口是 [`make_ttir`、`make_ttgir`、`make_llir`](../../triton/third_party/nvidia/backend/compiler.py#L292)。

<a id="read-one-pass"></a>
### 在 make_bin 里选一条 pass 路线

第一次打开 [function.cc](../../triton-legacy/lib/runtime/function.cc#L199)，只读两个范围：开头创建各个 pass 的那几行，以及紧接着的 `// run passes`。前一个范围告诉你某些对象之间有什么依赖，例如 `allocation(&liveness)` 让分配器可以查存活信息；后一个范围才是本次执行顺序。两处都看到名字「allocation」，一处在构造，一处才实际计算。

把本轮问题定成「一个长度为 1 的归约怎样被换掉」，按下面读一趟：

1. **从执行顺序进去。** 在 [第一次运行 peephole 的位置](../../triton-legacy/lib/runtime/function.cc#L225) 确认传进去的是同一个 `module`，下一行会运行 DCE。跳去 [peephole::run](../../triton-legacy/lib/codegen/transform/peephole.cc#L188)，只看 `module → function → block → instruction` 的三个循环和 `rewrite_unit_red(i, builder)` 这一行。这里的 `i` 是每次取出来试着匹配的指令。
2. **在规则里找三件事。** 看 [rewrite_unit_red](../../triton-legacy/lib/codegen/transform/peephole.cc#L99)。`dynamic_cast` 尝试判断 `value` 是不是归约指令，失败就返回。`get_operand(0)` 取归约输入，`arg->get_type()->get_tile_shapes()` 从值走到类型再取形状，这就是前一篇读的 `value.h` 与 `type.h` 的实际使用处。`shapes[x->get_axis()] == 1` 说明沿要消掉的轴只有一项，才构造 reshape。
3. **去消费者验证改写。** `replace_all_uses_with(new_red)` 将原来消费归约结果的操作改成消费 reshape。回到 `make_bin`，看到 [DCE 的入口](../../triton-legacy/lib/codegen/transform/dce.cc#L12) 后只读两处：从 store、return 等必须保留的操作向输入做标记，最后删除未标记指令。原归约失去有效的使用者后，DCE 可以清理它。
4. **回到执行顺序。** 后续分析拿到改完的模块。现在可以说清这一次改写的触发条件、类型来自哪个对象、谁看见了新结果。其他 peephole 规则和其他 pass 留到各自的问题再进。

以后读 layout 或 membar，可以沿同一方法先在 `make_bin` 看它读谁的结果，再进一个函数核对一张表或一个判断。例如看共享内存时，从 `liveness.run → allocation.run → barriers.run` 这三行开始，手里始终追同一块共享数据。

## §5.1 Machine-Independent Passes（PDF 第 5–6 页）

### §5.1.1 与 Listing 7：什么时候准备下一轮的值

首句说循环刚开始便 load，后续计算往往要等 load 返回。第二句提出识别循环并把准备数据的时间提前。看 Listing 7 左右两部分：

| 读到哪里 | 左右两侧各做什么 | 可跟读的代码 |
|---|---|---|
| 进入 B1 前的 B0 | 左边只算初始地址；右边在 B0 已经准备第一个 load 值 `x0` | [旧 dot kernel](../../triton-legacy/tests/common/src/dot.h)中 prologue 先构造首轮指针，再加载 a、b |
| B1 开头 | 左边每轮从当前指针读；右边用 phi 选取首轮预载或上一轮准备的值 | 旧 kernel 的循环开头先消费 a、b；[旧模块](../../triton-legacy/include/triton/ir/module.h)与[VisitForStmt](../../triton-legacy/lib/lang/code_gen.cc#L406)提供理解控制流的入口 |
| B1 末尾 | 先决定下一轮地址，再安排后续 load。判断这种移动的正确性要检查最后一轮边界和指针依赖 | 旧 kernel 在累加后推进 `pa/pb` 再条件加载，条件 `k > TK` 决定下一轮是否存在。印刷清单最后的 load 写成 `%p`；判断下一轮取谁时按真实源码的更新后指针追 |

拿 K=16、TK=8 来读旧 kernel，把 `pa` 想成指向 A 的八个 K 位置的地址表。`for(int k = K; k > 0; k -= TK)` 用 `k` 记录从本轮开始还有多少项，循环体完成后再减 8。进入循环前，`rk < K` 对本轮八个位置都为真，先从 K=0–7 的地址取数据。第一轮 `k=16` 时先用 `c += a @ b` 乘加，再令 `pa += TK * STRIDE_AK`，把每个地址沿 K 推进 8 个位置。此时 `k > TK` 为真，循环末尾的 `*?(checka)pa` 才为第二轮准备 K=8–15 的数据。第二轮 `k=8` 消费这批值，之后 `k > TK` 为假，阻止最后一次预载。这里的 `+=` 把右侧加到原值上，`-=` 从原值上减去右侧。论文右侧的 phi 表示「第一次拿循环外准备的值，后来拿上一轮末尾准备的值」，和旧程序中 `a` 第一次、第二次取值的来源可以逐项对照。

v0.1 这里看到的是手写软件预取，`make_bin` 的 pass 列表尚未核实论文所示的自动改写。现代[流水线目录](../../triton/lib/Dialect/TritonGPU/Transforms/Pipeliner)处理相近的延迟隐藏问题。可以从 [NVIDIA 编译流程](../../triton/third_party/nvidia/backend/compiler.py#L308)寻找调度和 pipeline 的调用，再追具体变换。

### §5.1.2：为什么知道 shape 就能删操作

作者用「转置两次回到原块」说明局部化简，并把其他代数关系列为可继续探索的方向。v0.1 有一个可完整追踪的不同例子。[旧 peephole.cc](../../triton-legacy/lib/codegen/transform/peephole.cc#L90)查找归约轴长度为 1 的情况。比如 `[[7],[9]]` 沿列求和，答案仍为 7 和 9，只少了一个长度为 1 的轴。上面的[改写走读](#read-one-pass)已经定位了这条规则。代入 `arg` 的 shape 为 `[2,1]`、归约轴为 1，`shapes[x->get_axis()]` 就是 `shapes[1]=1`，条件成立。原归约结果的 shape 是 `[2]`，`create_reshape(arg, x->get_type()->get_tile_shapes())` 直接把这个目标 shape 交给新操作。若把输入换成 `[2,3]`，同一分支条件为假，因为沿轴求和会把三个元素真的加起来。现代逐条追改写见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-2)。

## §5.2 Machine-Dependent Passes（PDF 第 6 页）

开头列出四项工作，对应下面四小节。读 v0.1 时从 `make_bin` 看前一项产出的信息怎样供下一项使用。

### §5.2.1 与 Figure 5：一个逻辑块怎样分给硬件

图把工作分到不同程序实例，再分给实例内部的执行单元和各线程。v0.1 的 [layouts::run](../../triton-legacy/lib/codegen/analysis/layout.cc#L388) 先用 `make_graph(i)` 按指令建图，再用 `connected_components(&values_, &groups_)` 把彼此连通的值归到一组，例如计算一组地址和使用这组地址的值可以有关联。到具体布局时，下面直接代入一维 shape 看每线程拿到多少元素。选一种布局和枚举全部布局是两个动作，论文的枚举范围要回看 §5.3。现代显式坐标例子见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-1)。

#### 先算出一组线程，再问它们访问什么

在 NVIDIA 这段旧代码里，一个 warp 是 32 个线程，lane 指某个线程在组内的编号。layout 描述的是逻辑元素交给谁处理。例如元素 12 对应的数组地址可以一直保持不变，布局却可以把这个元素交给别的 lane。

打开 [scanline_layout 构造函数](../../triton-legacy/lib/codegen/analysis/layout.cc#L167)并设一维形状为 `[128]`，`num_warps=1`，最连续轴为 0，连续访问分析允许每线程在该轴处理 4 项。顺着三行计算：

1. `size` 是各轴长度的乘积，得到 128。`num_threads = num_warps * 32` 得到 32。
2. `nts_[i] = clamp(size / num_threads, ...)` 计算每线程在这根轴上的连续元素数，取 128/32=4，并将它限制在源码给出的上下界之间。这里 `i=0`，上下界为 1 和 4。
3. `mts_[i] = clamp(num_threads, 1, shape_[i] / nts_[i])` 计算这根轴上用多少个线程位置，取 32。源代码还会把已分配的规模从 `size` 和 `num_threads` 的剩余配额中除去，以便处理后续轴。

这一步还没发出任何 load。先算出 `nts_`、`mts_` 再读下一节，就能问出「当地址来自转置或带 stride 的视图时，哪一轴让邻近线程真正读到邻近地址？」。

### §5.2.2 与 Figure 6：访问的是一块数据，付费的是一批事务

第一段定义相邻线程同时访问附近地址时更有机会让同一批事务带来的数据被充分使用。第二段说编译器能改变微块内的线程安排。图左读「用到哪些位置」「同时搬来了哪些没用到的位置」，再看右边邻近访问怎样减少浪费。

在 [extract_io_use](../../triton-legacy/lib/codegen/transform/coalesce.cc#L19) 里，设 `v` 是 `pa` 那块地址，`v->get_users()` 给出使用它的操作。假如一项是 `load(pa)`，`dynamic_cast<ir::io_inst*>(u)` 会成功，返回可当作访存指令访问的指针 `i`；`i->get_pointer_operand() == v` 也为真，所以 `result.insert(i)` 把 load 记进集合。假如一项是 `pa + offset`，它是计算地址的指令，转换会失败，`i` 为空。`i && ...` 从左向右判断，遇到空指针就停止，右侧的 `i->...` 不会执行。

[紧随其后的 extract_ld](../../triton-legacy/lib/codegen/transform/coalesce.cc#L27) 读取 `contiguous(ptr)`。假设两个轴上的连续元素数是 `[1,4]`：`std::max_element` 找到 4 所在的位置，`std::distance` 算它离开头有几个位置，得到轴 1。`result[axis].push_back(i)` 把该访存指令追加到轴 1 的列表。回到 [run 的分组循环](../../triton-legacy/lib/codegen/transform/coalesce.cc#L104)，这张表用于挑出要重新构造数据路径的访存。稍后的 `rematerialize` 遇到 load 时可以创建 `copy_to_shared`；代码处理的是 IR，实际搬运由最终设备程序执行。现代 [Coalesce.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)同样从内存操作入手；逐步比较见[第 11 章](../chapters/11-two-generations.md#sec-11-3)。

### §5.2.3 与 Figure 7：什么时候能重用同一段共享内存

前两句说明重复使用的 dot 操作数值得放到便宜且快的共享存储。下一句说应为每块数据决定存多久和放哪里。图的横向是时间，纵向是存储占用。若两块各 4 KiB 且同时存活，就需要为它们留出不同空间；若生命周期错开，就有机会占用同一区间。

在 [liveness.cc](../../triton-legacy/lib/codegen/analysis/liveness.cc#L14)，`indices.insert({instr, index})` 先为每条指令存一个顺序编号。假设某共享块在第 3 条指令形成，被第 5 和第 8 条使用。`std::min(start, indices.at(v))` 从组成该布局的值里取最早的定义，`std::max(end, indices.at(u))` 从使用者里取最后一个编号；`intervals_[layout] = segment{start, end}` 把得到的 `{3,8}` 写到此布局对应的表项。

[allocation.cc 的干涉检查](../../triton-legacy/lib/codegen/analysis/allocation.cc#L54) 接过这张表。`XS = {X0, X0 + NX}` 用起始字节 X0 和大小 NX 记下 X 目前占用的区间。下面的 `if (... && ...)` 要求时间也重叠、字节地址也重叠，才把 Y 放进 `interferences[x]`。例如两个值都试放 0–15 字节，如果一个在编号 3–8 使用，另一个在 6–10 使用，就需要区分这两个分配。随后代码为相邻布局选不同的编号（源码称 color），并在最终写入 `offsets_[x]` 时用编号和附近布局计算出的偏移量调整起点。这张偏移表之后被 membar 用来把 IR 读写转换成实际的共享字节区间。论文提到的线性时间存储分配与 v0.1 此处图着色属于具体算法差异。布局形成共享数据到使用 offset 的细节见[第 3 章](../chapters/03-legacy-layout-backend.md#sec-3-1)。

### §5.2.4 与数据流公式：哪些读写需要等其他线程

第一句给出机器执行假设。第二句提出编译器插 barrier（让协作线程在继续前等待彼此）。随后两类公式可从[旧 membar.cc](../../triton-legacy/lib/codegen/transform/membar.cc#L84)找对象：

| 公式 | 含义 | 源码追法 |
|---|---|---|
| 两条 in | 当前基本块从各前驱汇总尚需关心的写入和读取 | `written_to[pred]` 取这个前驱的写记录；`push_back` 收集后，`join` 把多份记录合在一起交给 `transfer` |
| RAW（read after write） | 有线程刚写共享区间，另一些线程现在要读重叠部分，须保证数据已经可读 | `intersect(new_written_to, read)` 把早先的写区间和本条指令的读区间比较，返回真假 |
| WAR（write after read） | 有线程还可能在读旧数据，下一批写入要复用同一地址，须先等读者 | `intersect(new_read_from, written)` 比较早先的读与本条指令的写 |
| 两条 out | 没冲突就扩充记录；发生冲突则在同步处分开前后的访问 | `clear()` 清掉同步前的记录，随后 `std::copy(..., std::back_inserter(...))` 把本条访问追加到旧记录清空后的列表 |

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

第一段比较调优参数的来源：论文提出从各 pass 汇总可调参数。第二段限定其实验只搜索分层 tiling，不同层在各自的区间内枚举 2 的幂。v0.1 的 [dot 基准](../../triton-legacy/tests/common/dot.h#L112) 给出的候选有 TM 为 128、TN 为 32、warp 数为 4；TK 根据数值类型选定。这个基准给每个参数单项候选，下面另用两项教学候选演示运行时接口怎样遍历组合。同样三问可用于读现代 [autotuner.py](../../triton/python/triton/runtime/autotuner.py) 的 Config、缓存键和实际试跑。比如尝试 B=32 或 64 是两份候选；即便每个配置编译成功，还要测量才知道给定输入下该采用哪一个。


读 `precompile` 时可以在纸上画 2×2 表。假设 `space.num_warps=[4,8]`、`space.defines` 中只有 `TM=[32,64]`。这时 `ranges` 记录 `[2,2]`，`_loop_nest` 对四组下标调用 `do_make`。例如下标 `[1,0]` 会选 warp 数 8、TM=32，随后 `make(stream, opt)` 编译或取该配置的设备程序。`callers_[opt].reset(call)` 把得到的可启动对象存到以选项为键的表里。到 [autotune 的候选循环](../../triton-legacy/lib/runtime/function.cc#L362) 才计时。`x.first` 是这一项的配置，`x.second` 持有可调用对象。`grid_fn(x.first)` 根据当前配置决定启动范围，`(*current)(stream, ..., args)` 执行该候选。若先测 12、再测 9，则 `ret = (ts < best_ts) ? current : ret` 在第二次更新赢家，下一行把 `best_ts` 降到 9。仅一个已编译候选时，前面的快速路径直接返回。这里的 C++ `[&](...){ ... }` 创建一个可交给循环调用的函数，外层需要的名字用引用访问，具体值如 `params` 由每次调用传入。
