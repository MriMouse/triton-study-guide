# 第 3 章 历史路线（三）：布局、同步、后端与调优

[返回总目录](../README.md) · [上一章：旧 IR 与优化](02-legacy-ir-passes.md) · [下一章：向量加法与 JIT](04-vector-add-jit.md)

第 2 章结束时，旧 IR 已经过局部改写。这一章走完后半程：

- **3.1**：布局、连续性、共享存储存活期/分配和 RAW/WAR barrier；
- **3.2**：旧 IR → LLVM → PTX → CUDA driver → caller，以及选项空间、缓存和 autotune。

这两部分放在一起，是因为它们共享同一组分析对象：`layouts`、`liveness`、`allocation` 先由 3.1 建立，再由 3.2 的 `codegen::generator` 消费。如果 3.1 没读清，3.2 里“按布局把 tile 拆成线程值”就只是黑箱。

## 本章地图

```text
旧 ir::module（已局部改写）
  → 3.1：
      align（连续性）→ axes（轴关系）→ layouts（scanline/shared/mma884）
      → coalesce（把 load 改成 copy_to_shared）
      → liveness（共享布局的存活区间）
      → allocation（字节区间复用）
      → membar（RAW/WAR/WAW → barrier）
  → 3.2：
      codegen::generator → llvm::Module
      → NVPTX → PTX 文本
      → CUDA driver 装载 → function 句柄
      → caller（模块 + 入口 + 签名 + 选项）
      → precompile / autotune / stream enqueue
```

| 部分 | 核心问题 | 读完后能回答 |
|---|---|---|
| 3.1 | 共享 tile 放在哪、活多久、何时同步？ | 为什么 liveness 不重叠仍可能需要 barrier |
| 3.2 | 旧 IR 怎样变成可启动设备程序？ | PTX、driver 模块、caller、autotune 各自负责什么 |

**读法：** 把“逻辑 shape”“布局”“分配字节区间”“执行同步”四件事分开写。3.1 的表格和 3.2 的对象接力都依赖这个区分。

<a id="sec-3-1"></a>
## 3.1 共享 tile 的形状、存活期、地址与 barrier

用两个问题区分布局和分配。布局回答「值的第 i 项由谁持有」，分配回答「要暂存这块值时，用共享内存的哪些字节」。两个逻辑块即使形状相同，生命周期也可能不同，因此可以在不同时间使用同一段字节。

这一章把四件常被混为一谈的事分开：逻辑值的 shape、线程/存储 layout、共享内存分配、读写同步。
先沿向量加法理解布局和合并访存，再用一个显式共享复制的小 IR 追到字节区间与 barrier。
向量加法本身不能证明发生了共享中转；共享例子会另行标明。

历史版本固定 2020-05-04 v0.1、`b9e36331be854e37aa14c538861689533ec0dfb0`。
以下未构建、未运行，也没有实得 layout/IR dump。
纸笔区间只用于说明真实算法读什么信息。摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. 先把构造依赖恢复到脑中

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 210–217 行的关系：

```cpp
codegen::analysis::layouts layouts(&axes, &align, opt.num_warps);
codegen::analysis::liveness liveness(&layouts);
codegen::analysis::allocation allocation(&liveness);
codegen::transform::membar barriers(&liveness, &layouts, &allocation);
// 省略：其他对象
codegen::transform::coalesce coalesce(&align, &layouts);
```

axes 描述值之间哪些轴关联；align 提供连续性等分析结果。
layouts 将这些信息与 num_warps 配合，给 tile 值组选择布局。
liveness 只对共享布局求存活区间，allocation 把它们安排到字节地址。
barriers 同时需要布局与分配结果，判断某次读/写是否涉及同一共享存储。

2.2 已给真实运行顺序：初次 layouts→coalesce→清理与其他改写→重算 axes/layouts→liveness→allocation→membar。
因此后面讨论 liveness 时，使用的是改写后重新分析的布局。
不能将 coalesce 之前的 layout 直接当最终存储结果。

### 2. layout 不是 shape 的别名

shape=[4] 只说有四个逻辑位置，尚未说每个位置在哪个线程或存储区。
layout 需要表达分布或存储方式；相同 shape 可具有不同 layout。
[layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc) 第 331–350 行的 connect：

```cpp
if(!x->get_type()->is_tile_ty())
  return;
if(!y->get_type()->is_tile_ty())
  return;
std::vector<int> x_axes = axes_->get(x);
std::vector<int> y_axes = axes_->get(y);
// 省略：把轴列表放进集合，计算交集 common
graph_.add_edge(x, x);
graph_.add_edge(y, y);
if(!common.empty())
  graph_.add_edge(x, y);
```

x/y 是某条指令及其操作数等旧 IR value。
先排除标量，再查询它们的轴编号；这些编号表达轴关系，不等于内存字节地址。
有共同轴时连接值图；即使不连彼此，也先以自环把节点放入图。
第 352–357 行的 make_graph 对每条指令与操作数执行这些连接。
第 388–400 行再对图取连通分量，并为每一组 values 调用 create。
所以布局是根据一组关联值推断，不能只看某个变量在源码里的名字。

### 3. 一个值组会选择哪类布局

同文件第 372–385 行，省略前面的最大 shape 选择：

```cpp
auto it_cts = std::find_if(values.begin(), values.end(), [](ir::value* v) {
    return dynamic_cast<ir::copy_to_shared_inst*>(v);
});
if(it_hmma_c != values.end())
  layouts_[id] = new mma884_layout(num_warps_, axes, shapes, values, align_);
else if(it_cts != values.end()){
  ir::copy_to_shared_inst *cts = (ir::copy_to_shared_inst*)*it_cts;
  ir::value *arg = cts->get_operand(0);
  create(groups_.at(arg), values_.at(groups_.at(arg)));
  layouts_[id] = new shared_layout(get(arg), axes, shapes, values, largest->get_type()->get_scalar_ty(), align_);
}
else
  layouts_[id] = new scanline_layout(num_warps_, axes, shapes, values, align_);
```

it_hmma_c 是前面寻找半精度 dot 相关结果得到的迭代器。
迭代器等于 end 表示没找到，而非最后一个有效对象。
优先命中这类 dot 时选择 mma884，不能把本快照能力推广到所有现代 Tensor Core 指令。
若组内有 copy_to_shared，就取其来源 arg，先建立来源组布局，再建立共享布局。
其他普通值组走 scanline。

传给 shared_layout 的 `get(arg)` 是来源布局；axes/shapes 是当前组轴与形状。
最后两个实参提供元素类型和连续性分析。
这解释了共享布局为何依赖此前的普通布局，而不是凭空分配一块字节数组。

普通向量加法没有 dot，也没有由这个例子本身要求的 copy_to_shared。
在没有其他变换引入共享需求的前提下，可按 scanline 分支理解。
这是静态条件判断，不是本机打印出来的最终布局报告。

### 4. scanline 的两个数量怎样从输入算出

[layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc) 第 172–175、185–193 行，**非连续摘录**：

```cpp
unsigned size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
unsigned num_threads = num_warps * 32;
nts_.resize(shape_.size());
mts_.resize(shape_.size());
// 省略：查找store指针并取得连续性
unsigned i = order_[0];
int contiguous = 4;
if(ptr)
  contiguous = std::min<int>(align->contiguous(ptr)[i], 4);
nts_[i] = clamp(size / num_threads, 1, std::min<int>(contiguous, shape_[i]));
mts_[i] = clamp(num_threads, 1, shape_[i] / nts_[i]);
size /= shape_[i];
num_threads /= mts_[i];
```

accumulate 从初值 1 开始乘各维大小，所以 shape=[4] 时 size=4。
32 是本快照采用的 warp 线程数，num_warps 来自编译选项。
order_[0] 是优先分配的轴，受指针连续性影响。
contiguous 被限制到不超过 4，再限制 nts；nts/mts 是构造布局时按维安排的参数。
clamp 的定义在第 19–22 行，把数值夹到给定范围。

纸笔取一维 shape=[4]、num_warps=1、连续性至少 4：
size/num_threads 是整数除法 4/32=0，夹到至少 1 后 nts[0]=1。
mts[0] 则夹到不超过 shape[0]/nts[0]=4，所以得到 4。
这只能说明构造器算出的参数；不能由这个小算式宣称“已实测只启动四个线程”。
实际线程块仍由选项与 caller 启动，元素映射由后端消费 layout 完成。

> 回到论文：§5.2.1、Figure 5，PDF 第 6 页，讨论分层 tiling 与机器层次。2020 的 scanline/mma884 是具体实现，不能把本章两行参数公式说成原论文全部枚举算法。见[论文定位表](../appendices/reference.md#sec-a-1)。

### 5. coalesce 从哪个值找到 I/O 指令

[coalesce.cc](../../triton-legacy/lib/codegen/transform/coalesce.cc) 第 19–32 行：

```cpp
void coalesce::extract_io_use(ir::value *v, std::set<ir::io_inst*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(i);
  }
}
void coalesce::extract_ld(ir::io_inst* i, std::map<int, std::vector<ir::io_inst*>>& result) {
  ir::value *ptr = i->get_pointer_operand();
  auto contiguous = align_->contiguous(ptr);
  auto it = std::max_element(contiguous.begin(), contiguous.end());
  int axis = std::distance(contiguous.begin(), it);
  result[axis].push_back(i);
}
```

v 可取 2.1 的 px。users 中可能有 load，也可能有其他计算。
dynamic_cast 先挑 I/O 指令；第二个条件还要求 v 真是它的指针操作数。
如果 v 只是 store 的待写数值，不能把这个使用误算成“v 的地址访存”。
result 是引用形参，调用者的集合被填入这些 I/O 节点。

extract_ld 再取指针的连续性向量，选最大分量的轴，将 I/O 按轴分组。
例如**假定分析结果**为 `[1,4]`，得到 axis=1；为 `[4,1]` 时得到 axis=0。
这是给定分析输入后的纸笔分组，不是声称 align 对任意指针都恰好返回这两个数组。
同文件第 118–119 行从轴分组的反向迭代中保留一组、收集其他组待 rematerialize。
代码按 map 的轴键遍历；不能把它描述成“挑选性能实测最快的轴”。

第 35–61 行的 rematerialize 递归复制依赖，遇到 load 时插 copy_to_shared，遇到已有共享复制则直接返回。
run 的第 132–135 行对待处理 load 创建共享复制并改接 uses：

```cpp
ir::instruction *cts = builder.insert(ir::copy_to_shared_inst::create(r));
r->replace_all_uses_with(cts);
cts->replace_uses_of_with(cts, r);
```

r 是原 load。第一行让 cts 使用 r；第二行把 r 的用户改成 cts，连 cts 自己也可能被改接。
第三行将 cts 的自引用修回 r，恢复“load→copy→用户”的数据流。
所以最后一行不是多余操作，而是避免 copy 使用自己的结果。
这类变化正是后面重算 layouts 的原因。

> 回到论文：§5.2.2、Figure 6，PDF 第 6 页，帮助理解相邻线程访问附近地址的目标。这里讲的 users/轴分组/重物化来自 2020 源码，不保证任意 stride 都能变成理想合并访问。

### 6. 另设一个共享复制小例

接下来需要真实的共享值，不能硬说向量加法一定经过共享内存。
以下是**教学示意 IR**；copy_to_shared/copy_from_shared 对应旧指令种类，文本不是 dump：

```text
slot 1: a = load(px)                 # float[4]
slot 2: s = copy_to_shared(a)        # shared layout S
slot 3: b = copy_from_shared(s)
slot 4: c = add(b, 1)
slot 5: store(pz, c)
slot 6: t = copy_to_shared(d)        # shared layout T，d是另一已定义值
slot 7: e = copy_from_shared(t)
slot 8: store(pw, e)
```

假定 s/t 分属两个共享布局组，没有 dot 所需 padding，也没有 double buffering。
两者元素类型都是 float32，shape=[4]。
这里 d、pw 是第二段工作明确假设的输入值/输出指针，不是 1.1 的隐含新形参。
目的是展示两个共享 tile 的先后使用；它们不扩展向量加法接口。

[layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc) 第 314–319 行计算大小：

```cpp
size_ = ty_->get_primitive_size_in_bits() / 8;
for(auto s: shape_)
   size_ *= s;
if(double_buffer_)
  size_ *= 2;
```

ty_ 是共享 tile 的元素类型，32/8=4 字节；乘 shape 的 4 得 16 字节。
本例无双缓冲，不乘 2。
实际 dot 布局可能 padding，双缓冲可能加倍，因此不能普遍用“逻辑元素数×字节数”当最终分配大小。

### 7. liveness 把值的使用关系变成时间区间

[liveness.cc](../../triton-legacy/lib/codegen/analysis/liveness.cc) 第 18–25 行先按函数内块/指令列表给指令编号。
编号是编译分析的顺序槽位，不是 GPU 纳秒时间戳。
第 29–48 行只处理 to_shared 成功的布局，先收集布局所有值的 users，再计算：

```cpp
unsigned start = INT32_MAX;
for(ir::value *v: layout->get_values())
  if(indices.find(v) != indices.end())
    start = std::min(start, indices.at(v));
unsigned end = 0;
for(ir::user *u: users)
  if(indices.find(u) != indices.end())
    end = std::max(end, indices.at(u));
intervals_[layout] = segment{start, end};
```

layout 来自 layouts_->get_all 中当前共享布局；indices 来自前面的编号表。
start 是这组共享值最早定义的编号；end 是它们所有用户中最晚编号。
本例设 S 只含 s，用户为 slot3，便得到 segment{2,3}；T 得到 {6,7}。
没有沿 b/c 的普通寄存器用户一路延长 S，因为 copy_from_shared 之后值已离开该共享布局。

[liveness.h](../../triton-legacy/include/triton/codegen/analysis/liveness.h) 第 29–39 行使用 `start <= idx && idx < end` 的区间判定。
因此后续写作按半开区间 `[2,3)`、`[6,7)` 理解它的相交规则。
这份分析按所列指令序号建区间，不能未经论证推广为完整循环/并发时间的精确存活模型。

### 8. allocation 同时考虑时间和字节空间

[allocation.cc](../../triton-legacy/lib/codegen/analysis/allocation.cc) 第 13–51 行先维护候选布局集合 J 和带时间段的空间候选 H。
H 初始包含起点字节 0、时间范围 0 到 INT_MAX；选择布局后记录 starts，更新可用候选。
后面第 60–67 行的干涉判断是：

```cpp
unsigned X0 = starts[x], Y0 = starts[y];
unsigned NX = x->get_size();
unsigned NY = y->get_size();
segment XS = {X0, X0 + NX};
segment YS = {Y0, Y0 + NY};
if(liveness_->get(x).intersect(liveness_->get(y))
    && XS.intersect(YS))
  interferences[x].insert(y);
```

x/y 是两个 shared_layout 对象。
XS/YS 是字节范围；liveness_->get 返回时间槽位范围。
必须两种范围都重叠，才加入干涉边。
仅仅字节地址相同，但使用时间错开，可以复用；仅仅时间重叠但地址不同，也不冲突。
后续第 76–96 行使用 first-fit 着色调整最终 offsets_。

对本例，若 S/T 均安排到 offset=0，它们字节范围同为 `[0,16)`，时间范围不重叠，干涉条件为 false。
所以 16 字节复用是符合此检查的一个安排；我们不把它冒充本机运行 allocator 得到的唯一输出。
若另加 U，时间 `[2,4)`、大小16且也放 `[0,16)`，就与 S 相交，需换地址或由分配阶段调整。
本段只演算冲突判定，不声称验证了历史 allocator 全部候选选择和容量计算的正确性。

### 9. 从一个值找回它的共享字节范围

[membar.cc](../../triton-legacy/lib/codegen/transform/membar.cc) 第 32–44 行：

```cpp
auto *i = dynamic_cast<ir::instruction*>(v);
if(!i)
  return;
if(!i->get_type()->is_tile_ty())
  return;
analysis::shared_layout* layout = layouts_->get(v)->to_shared();
if(!layout)
  return;
if(alloc_->has_offset(layout)){
  unsigned offset = alloc_->offset(layout);
  res.push_back(interval_t(offset, offset + layout->get_size()));
}
```

v 来自当前指令自身或它的操作数。
先要求为指令、tile、共享布局，再要求有实际分配偏移。
对 s，在本例假定分配下取得 offset=0、size=16，追加 `[0,16)`。
普通 float 参数、寄存器结果、没有共享布局的指针不会因为名字像 s 就进入该集合。

第 47–54 行将指令操作数的共享引用视为读区间，将指令自身的共享结果视为写区间。
phi 与 trans 不作为这里的新写入处理，因为它们可改变组织/引用而非产生新的共享写。
这也说明本 pass 的读写判断依赖具体 IR/layout 约定，不能拿任意字符串伪码直接套分析。

### 10. RAW/WAR 是怎样触发 barrier 的

[membar.cc](../../triton-legacy/lib/codegen/transform/membar.cc) 第 94–112 行，省略 double-buffer 安全特判：

```cpp
interval_vec_t read, written;
get_read_intervals(i, read);
get_written_intervals(i, written);
bool read_after_write = intersect(new_written_to, read);
bool write_after_read = intersect(new_read_from, written);
// 省略：safe_war 中某些双缓冲值的特殊处理
if(read_after_write || write_after_read) {
  insert_loc.insert(i);
  new_written_to.clear();
  new_read_from.clear();
}
std::copy(written.begin(), written.end(), std::back_inserter(new_written_to));
std::copy(read.begin(), read.end(), std::back_inserter(new_read_from));
```

new_written_to 记自上一同步点以来写过的共享区间；new_read_from 记读过的共享区间。
当前 read 与旧 written 重叠是 RAW：别的线程的写可能尚未对读取者可见。
当前 written 与旧 read 重叠是 WAR：可能有人还没读完，下一批写就复用了存储。
遇到冲突先记录在 i 前插 barrier，清空旧历史，再把当前操作的读写记录为新阶段。

按本例假定 offset=0，纸笔追踪：

| 当前指令 | 共享动作 | 此前记录 | 判断 |
|---|---|---|---|
| slot2 s | 写[0,16) | 空 | 记录写 |
| slot3 b | 读[0,16) | 先前写[0,16) | RAW，在slot3前同步 |
| slot6 t | 写[0,16) | slot3后的读[0,16) | WAR，在slot6前同步 |
| slot7 e | 读[0,16) | slot6后的写[0,16) | RAW，在slot7前同步 |

共享复用节省地址空间，却仍需要正确的执行次序。
这正解释了“存活期不重叠”为什么没有自动消除线程间同步需求。
若 T 使用完全不同字节范围，中间那项 WAR 不会由这组区间触发。
表格只演算这段单块、无双缓冲的条件，不是实际 GPU trace。

还要留意第 18–23 行的实际 `intersect` 辅助函数：它检测待查区间的两个端点是否落在已记录区间中。
这不应被当作任意两区间相交的通用实现；一个区间完全包住另一个时，需要另外检查覆盖情形。
本表使用相同的 `[0,16)`，会命中其左端点检查，没有用不同大小的嵌套区间替这份实现声称完整正确性。

### 11. 记录位置与真的插入指令是两步

第 134–162 行遍历控制流反向后序，从前驱收集读写集合，多轮传递直到插入位置集合数量稳定，再统一插 barrier。
不是只线性看一块后立刻忽略其他前驱。
第 70–72 行的普通指令分支：

```cpp
builder.set_insert_point(instr);
builder.create_barrier();
```

instr 是已记录的冲突指令；在它之前插 barrier，随后后端将 barrier 降为目标同步操作。
phi 分支在第 58–67 行则到 incoming 值所在块的末尾插入，因为 phi 本身是汇合节点。
第 119–129 行还根据共享布局双缓冲信息建立 safe_war；这不等于所有循环写都能免同步。
本例明确没有该优化，避免把特殊条件省略成普遍规则。

> 回到论文：§5.2.3、Figure 7 与 §5.2.4，PDF 第 6 页，分别看“时间/空间复用”和“RAW/WAR 前向分析”。本章展示快照对应数据结构；没有验证其覆盖现代异步指令、descriptor 或新硬件 barrier 的能力。

自测：shape 相同就共用一个地址吗？不一定，布局与存活期还要决定。
换地址能消除所有同步吗？不能，同一共享值的跨线程生产/消费仍可有 RAW。
下一部分交出的对象是什么？已经做过布局、分配与 barrier 插入的旧 IR，以及供代码生成查询的分析对象。

**第一部分交出的边界。** 3.1 结束时，旧 IR 里已经能看到“哪个 tile 用哪种布局、共享存储在哪个字节区间、哪些位置有 barrier”。但这些仍是编译器内部对象。3.2 让 `codegen::generator` 消费这些分析结果，把每个逻辑操作展开成 LLVM 值/指令，再经过 PTX、CUDA driver 和 caller，最终走到 `cuLaunchKernel`。

<a id="sec-3-2"></a>
## 3.2 旧 IR 怎样成为可调优、可启动的设备程序

把这里的「发射」读成 host 向 GPU 提交工作。编译器先交出设备代码，驱动加载代码，从中找到要执行的函数，再带着 grid（启动范围）和实参把调用提交到 stream（有顺序的设备工作队列）。

3.1 留下了旧 IR，以及布局、对齐和共享分配分析。
本章追踪其中的加法/GEP 怎样生成 LLVM 对象，再到 PTX、CUDA driver 模块与 caller。
最后回到 host，读懂选项枚举、基准选择和最终启动如何闭合。

固定历史源码为 2020-05-04 v0.1、`b9e36331be854e37aa14c538861689533ec0dfb0`。
它晚于论文；本章没有执行历史环境、GPU、调优或计时。
代码许可见 [CODE-NOTICE](../CODE-NOTICE.md)。所有数值配置只作纸笔推演。

### 1. 两个 generator 在这里交接

2.1 的全局 `Generator` 把 AST 变成自有 ir::module。
本章的 `codegen::generator` 把该模块变成 llvm::Module。
它们名字相近，输入、输出与 builder 都不同。

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 219、247–249 行，**非连续摘录**：

```cpp
codegen::generator isel(&axes, &layouts, &align, &allocation, target.get(), opt.num_warps);
// 省略：run各项分析与变换
isel.visit(module, *llvm);
std::unique_ptr<driver::module> res(driver::module::create(context, std::move(llvm)));
return res;
```

module 是已处理的旧 IR；`*llvm` 是解引用 unique_ptr 后得到的 LLVM Module 引用。
isel 保存分析对象的借用指针，按布局决定怎样表示每个 tile 的元素。
visit 完成时得到的是 LLVM 层程序对象，还不是 CUDA 模块句柄。
随后 create 接收 context 和 LLVM Module 的所有权，建立 driver 层产物。
`std::move` 表示允许把 unique_ptr 的所有权转交出去，不是将 GPU 数组从一块显存搬到另一块。

### 2. LLVM module 入口先建立什么

[selection/generator.cc](../../triton-legacy/lib/codegen/selection/generator.cc) 第 1396–1414 行。
先摘出入口和函数循环，中间省略共享存储声明：

```cpp
void generator::visit(ir::module &src, llvm::Module &dst) {
  mod_ = &dst;
  ctx_ = &dst.getContext();
  builder_ = new Builder(*ctx_);
  // 省略：为GPU共享存储建立LLVM全局变量
  for(ir::function *fn: src.get_function_list())
    visit_function(fn);
}
```

src/dst 分别接收上一节旧 module 与 LLVM Module。
mod_ 保存目标模块指针；ctx_ 指向 LLVM 上下文，不能与前端 ir::context 混同。
Builder 是本文件使用的 LLVM builder 类型，后续 CreateGEP/CreateBinOp 都产生 LLVM 指令。
循环将每个旧 IR 函数交给 visit_function。

第 1402–1410 行仅在 GPU 且分配大小非零时建立 `__shared_ptr` 全局变量。
其元素存储以 allocation.allocated_size() 决定，使用 address space 3。
这把 3.1 的字节资源结果带到了目标程序声明，尚未在 host 为每个共享 tile 执行一次显存 malloc。
没有共享需求时，不必虚构这份数组存在。

### 3. 参数与基本块怎样一一对应

同文件第 1243–1287 行生成函数。
先用 llvm_type 把旧函数类型转换成 LLVM FunctionType，再创建同名 LLVM Function。
第 1273–1287 行连续摘录：

```cpp
for(unsigned i = 0; i < fn->args().size(); i++)
  vmap_[fn->args()[i]] = &*(ret->arg_begin() + i);
// create blocks
for(ir::basic_block *block: fn->blocks()) {
  BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
  vmap_[block] = dst_block;
}
builder_->SetInsertPoint((BasicBlock*)vmap_[fn->blocks()[0]]);
// initialize layouts
for(auto x: layouts_->get_all()){
  visit_layout(x.second);
}
// generate LLVM-IR code
for(ir::basic_block *block: fn->blocks())
  visit_basic_block(block);
```

fn 是旧函数；ret 是前面已创建的 LLVM Function 指针，不是一个 GPU 返回数值。
第一段用旧 argument 指针当键，保存对应 LLVM argument 地址。
因此旧 x 形参在 LLVM 中仍有对应参数，不需要把 x 内容嵌进编译结果。
第二段为每个旧基本块建立 LLVM 基本块，并用 vmap_ 保存对应关系。
接着设置首块插入点，再根据布局建立 machine layout 对象，最后访问指令。

vmap_ 负责标量及基本块等映射；tile 值另有 tmap_，因为一个逻辑 tile 未必只对应一个 LLVM 标量。
第 1330–1340 行 get_value/set_value 对 tile 走 tmap_、对标量走 vmap_。
这正是逻辑 tile 到线程所持元素的表示边界。

### 4. 读一条加法 visitor

第 224–230 行：

```cpp
void generator::visit_binary_operator(ir::binary_operator*binop) {
  for_each(binop, [&](indices_t idx){
    Value *lhs = get_value(binop->get_operand(0), idx);
    Value *rhs = get_value(binop->get_operand(1), idx);
    Value *ret = builder_->CreateBinOp(llvm_op(binop->get_op()), lhs, rhs);
    set_value(binop, idx, ret);
  });
}
```

binop 例如 2.1 的 fadd；它的两个旧操作数是 masked load 结果。
`[&]` 是 C++ lambda 捕获，允许函数体借用外层 binop/builder_ 等对象。
for_each 对标量只调用一次，对 distributed_tile 遍历该布局提供的元素索引。
idx 因而不是任意 Python 数组下标，也不能简单等同线程编号。

get_value 将旧操作数与当前 idx 对应到已有 LLVM Value。
llvm_op 把旧二元操作码转换为 LLVM 操作码。
CreateBinOp 在当前 LLVM 块里构造指令，返回 LLVM Value* ret。
set_value 再保存“旧 binop 的这个位置→新 LLVM 结果”的映射，供后面的 store 等用户查询。
这里的 lhs/rhs 仍是程序值对象；加法真正计算输入浮点数发生在设备执行时。

### 5. 指针 GEP 保留元素类型

第 233–242 行：

```cpp
void generator::visit_getelementptr_inst(ir::getelementptr_inst* gep) {
  for_each(gep, [&](indices_t idx){
    Value *ptr = get_value(gep->get_operand(0), idx);
    std::vector<Value*> idx_vals;
    std::transform(gep->idx_begin(), gep->idx_end(), std::back_inserter(idx_vals),
                   [&](ir::value* x){ return get_value(x, idx);});
    Type *source_ty = llvm_type(gep->get_source_elt_ty()->get_scalar_ty(), *ctx_);
    Value *ret = builder_->CreateGEP(source_ty, ptr, idx_vals);
    set_value(gep, idx, ret);
  });
}
```

gep 是旧 `x + offset` 生成的地址节点。
第一个 get_value 转换基指针；transform 逐个转换旧索引，并追加到 idx_vals。
`std::back_inserter` 让输出一个个放进 vector 尾部，不预先要求固定长度。
source_ty 把旧 GEP 的源元素类型转换到 LLVM，用于解释索引的元素步长。
CreateGEP 产生 LLVM 地址计算；set_value 让后续 LLVM load 能取得这个地址。
1.1 里 float 指针 offset=4 对应 16 字节的语义由类型与 GEP 保留下来，而不是把每个 `+` 一律当字节加法。

program id 的另一条小映射在第 540–543 行：

```cpp
Module *module = builder_->GetInsertBlock()->getModule();
Value *ret = tgt_->get_block_id(module, *builder_, pid->get_axis());
vmap_[pid] = ret;
```

pid 是旧 get_program_id_inst；axis 从旧指令取得。
tgt_ 选择目标实现，返回表示块编号的 LLVM Value，再存入 vmap_。
这是为运行时生成“取得编号”的代码，不是在编译时查询当前 CPU 的编号。

### 6. masked load 并没有丢掉条件

第 339–375 行读取布局、对齐、指针 tile、掩码 tile 和 false-value tile，再按合适 vector_size 组装 load。
其中第 358–375 行的关键控制流，省略 false vector splat 的分支：

```cpp
Value *mask = masks->get_value(idx);
BasicBlock *current_bb = builder_->GetInsertBlock();
Function *parent = builder_->GetInsertBlock()->getParent();
BasicBlock *mask_then_bb = BasicBlock::Create(*ctx_, "mask_then", parent);
BasicBlock *mask_done_bb = BasicBlock::Create(*ctx_, "mask_done", parent);
builder_->CreateCondBr(mask, mask_then_bb, mask_done_bb);
builder_->SetInsertPoint(mask_then_bb);
Value *result_then = builder_->CreateLoad(ptr);
builder_->CreateBr(mask_done_bb);
builder_->SetInsertPoint(mask_done_bb);
// 省略：声明current_result与if(false_values)
current_result = builder_->CreatePHI(result_then->getType(), 2);
((PHINode*)current_result)->addIncoming(result_then, mask_then_bb);
// 省略：取得并按需splat result_false
((PHINode*)current_result)->addIncoming(result_false, current_bb);
```

mask 是已经降到 LLVM 的条件；current_bb 是条件分支之前所在块。
true 时进入 mask_then，才执行 load；false 时直接去 mask_done。
done 中的 LLVM phi 根据到达前驱选择真实读取值或 false value。
这属于该快照后端具体 lowering，不表示旧 IR 原来就是 LLVM 分支形式。
2.1 的 undef 也没有在此凭空变成“保证为零”。

示例选择 vector_size 还依赖连续性和对齐分析。
本章没有运行验证任意尾部 mask 与向量化组合的全部正确性；不能仅凭几行控制流便推广到所有 shape。

### 7. 从 LLVM Module 转交 driver Module

[driver/module.cc](../../triton-legacy/lib/driver/module.cc) 第 79–85 行：

```cpp
module* module::create(driver::context* ctx, std::unique_ptr<llvm::Module> src) {
  switch(ctx->backend()){
    case CUDA: return new cu_module(ctx, std::move(src));
    case OpenCL: return new ocl_module(ctx, std::move(src));
    case Host: return new host_module(ctx, std::move(src));
    default: throw std::runtime_error("unknown backend");
  }
}
```

ctx 接收当前设备 context；src 接收 LLVM Module 所有权。
CUDA 分支把它继续移交 cu_module。
列表中出现 OpenCL/Host 分支，不证明这些路径在本快照均完整可用，例如 OpenCL 构造函数明确抛 not supported。
本章只继续 CUDA 路线。

### 8. LLVM 后端生成的是 PTX 文本

同文件第 243–249 行：

```cpp
auto cc = ((driver::cu_device*)device)->compute_capability();
std::string sm = "sm_" + std::to_string(cc.first) + std::to_string(cc.second);
llvm::SmallVector<char, 0> buffer;
module::compile_llvm_module(std::move(module), "nvptx64-nvidia-cuda", sm, "", buffer, "ptx63", Assembly);
std::string result(buffer.begin(), buffer.end());
```

device 给出 compute capability，两部分数字拼成 LLVM 目标处理器名称 sm。
compile_llvm_module 的 module 是 LLVM Module，buffer 接收输出字符。
triple 指定 NVPTX CUDA 目标，`Assembly` 要求汇编文本；这里对应 PTX。
`ptx63` 是这份历史代码写入的目标特性参数，不能照抄成现代环境建议。

通用发射函数第 100–128 行创建 TargetMachine、设置 data layout，并运行 LLVM pass manager 输出到 buffer。
第 127–128 行：

```cpp
machine->addPassesToEmitFile(pass, stream, nullptr, ll_file_type(ft));
pass.run(*module);
```

pass 此处是 LLVM 的 pass manager，与 2.2 手工顺序调用旧 Triton pass 属于不同层。
发射后 result 是 host 内存中的 PTX 字符串，尚未取得可 launch 的 CUDA function 句柄。
同文件第 250–259 行还有驱动版本检查与 PTX 文本修补，进一步说明它是有年代约束的具体实现。

### 9. PTX 交给 CUDA driver，再取得 kernel 句柄

同文件第 264 行委托构造：LLVM 输入先调用 compile_llvm_module，所得字符串进入另一个 cu_module 构造函数。
第 266–274 行摘录核心：

```cpp
cu_module::cu_module(driver::context * context, std::string const & source)
  : module(context, CUmodule(), true), source_(source){
  cu_context::context_switcher ctx(*context);
  // 省略：JIT错误日志选项和缓冲区
  dispatch::cuModuleLoadDataEx(&*cu_, source_.data(), 2, opt, optval);
```

source 就是 PTX 字符串；source_ 保存副本。
context_switcher 确保使用相应 CUDA context。
cuModuleLoadDataEx 让 driver 读取 PTX 并装载/JIT，输出 cu_ 持有的 CUDA module 句柄。
这条历史路径没有在这里显式调用外部 ptxas 产生 cubin 文件；不能套用现代 8.1 的阶段名称改写事实。
PTX 也不等于 GPU 最终执行的机器码，driver 仍承担后续目标处理。

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 148–155 行的 caller 构造保存 parent_ 模块、opt_、函数名，调用 driver::kernel::create 查找入口，并从旧 IR 函数类型提取签名。
所以 caller 不只是“一个函数名”：它把模块生命周期、入口句柄、形参类型和编译选项放在一起。

### 10. 选项空间怎样变成一批 caller

[function.h](../../triton-legacy/include/triton/runtime/function.h) 第 61–79 行定义两个不同结构：
options_space_t 保存每个宏的字符串候选 vector 和 num_warps 候选 vector；options_t 保存每个宏的单个字符串值与一个 num_warps。
纸笔设 TILE 候选为 `{"4","8"}`，num_warps 为 `{1,2}`，就有四个组合。
小 TILE 只是教学选项，未证明适合这个历史运行时或有性能意义。

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 304–317 行：

```cpp
std::vector<size_t> ranges;
ranges.push_back(space.num_warps.size());
for(const auto& x: space.defines)
  ranges.push_back(x.second.size());
auto do_make = [&](std::vector<size_t> params) {
  unsigned i = 0;
  options_t opt;
  opt.num_warps = space.num_warps[params[i++]];
  for(auto D: space.defines)
    opt.defines[D.first] = D.second[params[i++]];
  caller* call = make(stream, opt);
```

ranges 保存每个维度的候选数量，纸笔为 `[2,2]`。
params 是这组空间中的下标组合，例如 `[1,0]`。
首先用 params[0] 选 num_warps=2，然后用 params[1] 选 TILE="4"。
宏值仍是字符串，因为它接下来传给预处理器；options_t 的 D<T> 可按需将文本转整数。
make 为该组选项执行缓存或编译路径，返回 caller*；失败可返回 nullptr。
第 318–326 行跳过失败候选，初始化必要常量后以 opt 为键存入 callers_。
第 329 行 `_loop_nest(ranges, do_make)` 枚举这些组合；第 48–61 行的实现是普通计数循环，不能因旁边注释写 multithreaded 就断言它实际并行编译。

### 11. 编译缓存与调优缓存别混淆

make 第 256 行把源码缓存目录与 `opt.to_str()+".ptx"` 组成产物路径。
该目录前缀由源码哈希构造；options_t::to_str 在第 69–82 行编码 num_warps 和宏值。
caller::write 第 112–121 行写函数名、参数签名，再写 CUDA module 的 PTX source。
因此扩展名虽为 .ptx，文件开头还含 caller 的元数据，不能把整份缓存无条件当纯 PTX 传给工具。
read 第 124–139 行先读元数据，再用余下文本构造 cu_module 和 kernel。

另一份 cache_ 保存“运行参数条件→最佳 caller”，不是源码编译产物。
get_key 第 335–345 行把设备指针与整型实参值收集为键，忽略非整型参数。
向量加法里 N 进入这个键；x/y/z 的内容不进入键。
这是一份旧实现的策略，不能当成现代缓存键的完整规范或正确性保证。

### 12. autotune 什么时候真的执行 kernel

第 362–375 行，省略异常检查的摘录：

```cpp
if(callers_.size() == 1)
  return &*callers_.begin()->second;
double best_ts = INFINITY;
caller* ret = nullptr;
for(auto &x : callers_){
  // 省略：检查候选非空
  caller* current = &*x.second;
  double ts = tools::bench([&]() { (*current)(stream, grid_fn(x.first), args); },
                                   stream, true);
  ret = (ts < best_ts) ? current : ret;
  best_ts = std::min(ts, best_ts);
}
```

只有一个有效候选就直接返回，无需搜索。
多个候选时 x.first 是具体 options_t，传给 grid_fn 得到与该 TILE 匹配的网格。
current 是已编译 caller；bench 中的 lambda 真的调用它，因而调优会执行设备工作。
ts 是 bench 返回的计时值，不是编译时间，也不是本书提供的实测数。
ret 保存目前最快候选地址；最终返回它并缓存在 cache_。
第 351–361 行的输入缓冲复制代码处于注释状态，因此不能声称旧 autotune 自动保护所有会被原地改写的输入。

> 回到论文：§5.3，PDF 第 7 页，讨论从优化参数组成搜索空间。论文具体层级 tiling 空间与这里的宏/num_warps 选项接口应分别阅读；不声称这四个纸笔组合就是论文实验空间。见[论文定位表](../appendices/reference.md#sec-a-1)。

### 13. 最终一次启动怎样封装参数

function::operator() 第 455–469 行先 precompile、查 key、必要时 autotune，最后：

```cpp
(*it->second)(stream, grid_fn(it->second->opt()), args);
```

it->second 指向选出的 caller，grid_fn 用它的 opt 算网格，args 保持 z/x/y/N 顺序。
caller::operator() 第 159–180 行检查参数数量/类型，把 buffer 或标量传给 kernel::setArg，再把不足三维的 grid 补 1。
第 182 行真正 enqueue：

```cpp
stream->enqueue(&*bin_, grid, {opt_.num_warps * 32, 1, 1});
```

bin_ 是 kernel 句柄封装，不是 host 浮点数组。
grid 是块数，最后的三元组是线程块维度；num_warps=4 时 x 维为128线程。
TILE=4 的逻辑元素数与128个线程是两个独立数量，这正回到 1.1 的区分。

[stream.cc](../../triton-legacy/lib/driver/stream.cc) 第 152–157 行的 CUDA enqueue 最后调用：

```cpp
dispatch::cuLaunchKernel(*kernel->cu(), grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, *cu_,(void**)cu_kernel->cu_params(), NULL);
```

kernel->cu 提供设备函数句柄；cu_params 保存封装好的实参；cu_ 是 CUDA stream。
这一步是提交执行，和前面构造 LLVM load 是不同时间层。
提交成功也不等于 host 此刻已经可以无同步读取完整结果。

### 14. 用对象接力结束历史主线

```text
源码字符串 → tokens → AST
→ 自有ir::module（tile、value、basic_block）
→ 旧pass与布局/分配/同步
→ llvm::Module（目标程序对象）
→ PTX字符串 → CUDA driver模块句柄 → kernel句柄
→ caller（模块+入口+签名+选项） → stream提交
```

箭头两端不是同一种对象改了名字，而是每一步有明确函数输入、构造结果和所有权/引用关系。
预编译枚举“有哪些候选可调用”，调优选择“当前参数用谁”，最终 caller 才提交实际工作。
下一章回到现代路线，从向量加法和 JIT 开始；完整的两代对照放在第 11 章。

自测：PTX 是最终机器码吗？不是。
precompile 就测出了最快候选吗？没有，基准在 autotune。
缓存扩展名 .ptx 是否保证文件全为 PTX？不保证，本实现先序列化名称/签名。
本章有历史性能结果吗？没有，所有调用与对象关系只做静态核对。

## 第 3 章小结

| 对象 | 由谁创建 | 交给谁 | 不能混淆成 |
|---|---|---|---|
| `layouts` | `make_bin` 栈上构造 | coalesce、liveness、membar、isel | 逻辑 shape |
| `copy_to_shared` | coalesce 改写 | 后续 liveness/allocation | 已经执行的共享写 |
| `liveness` 区间 | `liveness.run` | allocation、membar | GPU 纳秒时间 |
| `allocation` | `allocation.run` | membar、isel | 设备显存分配 |
| `llvm::Module` | `codegen::generator::visit` | `driver::module::create` | 旧 `ir::module` |
| PTX 文本 | LLVM NVPTX 后端 | `cuModuleLoadDataEx` | 最终机器码 |
| `caller` | `function::make` | `function::operator()` | 单个函数名 |
| `cuLaunchKernel` | stream enqueue | 设备执行 | host 返回即完成 |

### 综合检查

1. 相同 shape 的两个值一定共用同一布局/地址吗？不一定；布局、存活期和分配都要看。
2. `liveness` 不重叠就能跳过 barrier 吗？不能；同一共享值的跨线程生产/消费仍可能是 RAW。
3. `PTX` 和 `cubin` 是一回事吗？不是；PTX 是虚拟指令文本，cubin 是目标二进制。
4. 旧 autotune 在什么时候执行 kernel？`tools::bench` 的 lambda 里真的调用 caller，因此会执行设备工作。
5. `precompile` 和 `autotune` 分别做什么？前者枚举候选并编译/缓存 caller，后者按运行参数测速选最快候选。
6. 历史版本能当作 2019 论文的精确实现吗？不能；它是 2020 v0.1，且存在已知映射差异。
