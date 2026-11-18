# 摘要、§1–§3：从问题到一段能读懂的 Triton-C

[首次读编译器的方法](00-how-to-read.md) · [路线目录](README.md) · [下一篇：IR](02-ir.md)

## 打开仓库前先认三段程序

想象你要求电脑把十个数两两相加。**算子**规定结果，例如每项都等于对应两项的和。**kernel**是设备执行的一段程序，例如规定「我这一份负责从第 4 项到第 7 项，读两边并写结果」。**host** 指这里负责组织工作的 CPU 一侧。host 准备数组、调用编译器获得设备程序，再把程序和参数交给 GPU。一个深度神经网络（DNN）由许多这样的运算组成；论文讲的编译工作是如何把其中的计算描述做成可以高效执行的设备程序。

打开[历史 dot.h](../../triton-legacy/tests/common/src/dot.h)。文件外层是 C++，`const char *dot` 里有一大段 `R"( ... )"`。它是 C++ 的原始字符串，用来直接保存中间的文字。字符串里的 `__global__ void dot(...)` 是旧 Triton-C kernel：

- `__global__` 标记设备程序入口。这里看它怎样声明入口，暂时不用查 C++ 的编译知识。
- `void` 表示函数没有通过 `return` 交还一个值；结果由最后的写回存到 C 指向的内存。
- `TYPE * A` 表示 A 是一个指针，值是输入存储的起点。TYPE 是构建时要换成具体类型的宏，例如 `float`。顾名思义，宏在本文这一例里就是编译源代码前的文字替换。
- `int M, int N, int K` 是三个整数。用这三个数描述矩阵的大小。
- `*` 在声明 `TYPE* A` 中说明类型是指针；表达式 `*pa` 按地址去访问值；`TK * stride` 中才是乘法。碰到同一个符号，先看它在声明里还是表达式里。

[旧运行时的 `cpp.Process(tokens)`](../../triton-legacy/lib/runtime/function.cc#L269) 在 CPU 上处理字符串，把字拆成更小的单位（token，例如标识符、数字、括号）。下一步 `Parser parser(tokens)` 用这些 token 创建 parser，`parser.Parse()` 才把它们组成 AST（用节点和子节点记录语法结构）。例如 `A + offa` 会有一个加法节点，左右各一个输入。GPU 真正取得 A 的数值要等编译完、host 启动设备程序之后。

先记住三个观察窗口：正在读的 **C++ 编译器程序**、被它读入的 **Triton-C 源码**、GPU 后来执行的 **设备程序**。同样一个 `+` 出现在三处时，输入对象和执行时刻可能都不同。

### 读完入口后，怎样选择下一处

先说这段代码想解决什么：矩阵 A 有两个逻辑坐标 `m` 和 `k`，但它在内存里是一长串数。计算地址前要知道「m 加 1，地址跨几个元素」和「k 加 1，地址跨几个元素」。这两个数叫步长。host 端的配置要告诉 kernel，两个方向各使用哪个步长。我们从一行源码算出选择结果，再代回 kernel。

<a id="read-stride-macro"></a>
#### 读懂 STRIDE_AK 的整行 C++

以下几行摘自 [测试驱动的 triton_dot](../../triton-legacy/tests/common/dot.h#L75)，中间省略其他参数设置：

```cpp
std::vector<std::string> sa = { "1", "lda" };
// ...
opt.defines.push_back({"STRIDE_AK", {AT ? sa[a_order[0]] : sa[a_order[1]]}});
opt.defines.push_back({"STRIDE_AM", {AT ? sa[a_order[1]] : sa[a_order[0]]}});
```

| 写法 | 念成一句话 | 在这行具体做什么 |
|---|---|---|
| `std::vector<std::string> sa` | 创建一个依次存放字符串的列表，名字叫 sa | 第 0 项是文字 `"1"`，第 1 项是文字 `"lda"`。引号里的 lda 是以后要写入 kernel 的名字 |
| `a_order[1]` | 取列表 a_order 的第二项，下标从 0 开始 | 如果 `a_order={1,0}`，第二项就是整数 0 |
| `sa[a_order[1]]` | 先算内层下标，再去外层列表查对应项 | 刚算出 0，所以取 `sa[0]`，得到文字 `"1"` |
| `AT ? x : y` | 如果 AT 为真，整条表达式的值是 x；如果为假，值是 y | `AT` 是调用者传下来的布尔值，这里用它在两种 A 布局下交换步长的分配 |
| `{"STRIDE_AK", {选出的文字}}` | 建立一条设置，第一项为宏名，第二项为候选列表 | 例如 `{"STRIDE_AK", {"1"}}` 这一条有一个候选。内层花括号建立只有一项的列表；外层把名字和列表配成一对 |
| `opt.defines.push_back(...)` | 从 opt 取出 defines 列表，把这条设置追加到末尾 | 原来的设置还在，新加一条 `STRIDE_AK`。声明里 `define_t` 是「字符串和字符串列表」的组合，所以同一个接口也能给一个宏多个候选 |

源码里的容器类型可对照 [options_space_t](../../triton-legacy/include/triton/runtime/function.h#L60)。第一次只要会把表达式从里向外算出来，就已经用到了这一行的 C++ 语法。

#### 用仓库已有的配置算出替换结果

在 [tests/bench/dot.cc](../../triton-legacy/tests/bench/dot.cc#L10)，外层循环给出 `ord={1,0}`，内层给出 `{false,false}`。调用 `bench_dot(..., ord, ord)` 时，同一份 ord 交给 A、B 两个参数，所以到上面这行时 `AT=false` 且 `a_order={1,0}`。这组实际配置的 M、N、K 都是 1024。

1. `AT` 是 false，`STRIDE_AK` 取冒号右侧。依次算 `a_order[1] → 0`、`sa[0] → "1"`，这条宏的候选就是文字 `"1"`。
2. `STRIDE_AM` 也取冒号右侧，这次是 `a_order[0] → 1`、`sa[1] → "lda"`。这里得到的是变量名。host 另算 `int32_t lda = (AT ^ (a_order[0] == 1)) ? K : M;`（补了表示求值顺序的括号）。对两个真假值，`^` 在两边不同时得到真。这里是 `false ^ true`，所以 host 算出 `lda=K=1024`。之后把这个整数放进 [kernel 实参列表 args](../../triton-legacy/tests/common/dot.h#L120)。
3. `BROADCAST_AK` 的两个候选分支拼写相同，都给 `"newaxis, :"`，`BROADCAST_AM` 都给 `":, newaxis"`。`SHAPE_A` 是 `"TM, TK"`。在当前 float 基准里，host 还给 `TM=128`、`TK=8`。因此形状宏对应 128 行、8 列。K 坐标放在一行，M 坐标放在一列，编译器再把两者广播到同一形状。

[precompile 的 do_make](../../triton-legacy/lib/runtime/function.cc#L310)把候选列表里的选中项存为 `opt.defines[名字] = 文字`。[function::make](../../triton-legacy/lib/runtime/function.cc#L269) 再把每一对交给 `cpp.AddMacro`。左边的 C++ 已经完成选择，parser 随后读出的 Triton-C 对应下面的教学展开：

```c
int offa[128, 8] = rk[newaxis, :] * 1 + rm[:, newaxis] * lda;
float* pa[128, 8] = A + offa;
```

现在取其中一个位置，让这一项的行坐标是 `m=2`、归约坐标是 `k=3`，偏移就是 `3 + 2*1024 = 2051` 个 float，`pa` 对应 `A+2051`。编译时替换出来的仍是名字 `lda`，执行时函数参数提供它的值。这样就能分清三个动作：CPU 选择替换文字，编译器用替换后的文字造程序，GPU 执行程序时才算地址和取数。

如果第一次读到 `^`、`push_back` 或两层方括号，先停在需要的一个表达式，把变量换成上面的具体值。算到 `A+2051` 就可以回到 Listing 1 对照。下一段的小地址表换了一组较小的步长来说明广播。

## 摘要与 §1 Introduction（PDF 第 1–2 页）

| 原文位置 | 读这一句时问什么 | 到仓库看什么 |
|---|---|---|
| 摘要开头，从新想法受 kernel 限制到专家实现的代价 | kernel 是某个算子的设备程序。新的索引或计算方式需要一份实际实现，才能在 GPU 上做性能比较 | [旧 dot 测试](../../triton-legacy/tests/common/dot.h#L147)先把输入传给 GPU，用 `function(args, grid, stream)` 提交 kernel，再读回结果交给 `testing::diff(hc, rc)` 与 CPU 结果比较。这里 `hc` 是读回的值，`rc` 是 CPU 算出的值。作者对开发成本的判断属于动机 |
| 摘要介绍 tile 的句子 | tile 是一份程序操作的一块数组，形状可以在编译时确定。比如矩阵中一块有 16 行、32 列 | 先读[旧 type.h 的前两步](02-ir.md#read-tile-type)，找到保存 shape 的 `shapes_` 和保存元素类型的 `contained_tys_`；§4 再顺着创建者与使用者走完 |
| 摘要列两项设计的句子 | 谁表达整块计算，谁决定怎样优化这块计算 | [旧 function.cc](../../triton-legacy/lib/runtime/function.cc#L191)依次读 `make_ir` 和 `make_bin`。前者从语法树产生旧 IR，后者让多个编译步骤处理该 IR |
| 摘要末句 | 证据来自哪些算子和比较对象 | 在[实验篇](04-experiments.md)追矩阵乘、普通卷积和 shift 卷积的入口，各自对照论文图 |
| §1 第 1 段，GPU 普及、供应商库、覆盖限制 | 三句分别谈计算硬件、可调用的现成实现、现成实现缺失时的工作量 | 用[旧 dot 基准驱动](../../triton-legacy/tests/bench/dot.cc)理解为什么自己写的实现和库函数能够并列测量 |
| §1 第 2 段，DSL 与索引表达能力 | DSL 是为特定领域提供的语言；这里讨论的是当时表达式能否写出依数据变化的地址 | [旧卷积示例](../../triton-legacy/python/examples/einsum.py#L174)取输入行的表达式是 `h + sh[c]`。例如 `sh=[0,1]`、输出行 `h=2`，则通道 0 取输入第 2 行、通道 1 取第 3 行。两通道的值及权重怎样相乘见[卷积走读](04-experiments.md#listing-8) |
| Figure 1 | 横轴是每单位传输数据的运算量，纵轴是计算吞吐量。沿横轴变化的是矩阵形状带来的运算强度 | 这张图是 GTX1070 实验。读图应先记下固定的两维和变化的一维，源码从[旧 dot 基准驱动](../../triton-legacy/tests/bench/dot.cc)理解测试对象 |
| §1 第 3 段，手写 micro-kernel、高层 tiling、提出 Triton | micro-kernel 是针对小块数据手写的局部实现。作者把块交给编译器处理，想减少每次重新指定硬件细节的工作 | 在[旧 layout.cc](../../triton-legacy/lib/codegen/analysis/layout.cc)看 block 内的分布由编译器对象记录。现代对照见[第 7 章](../chapters/07-layout-passes-memory.md#sec-7-1) |
| Figure 2 与贡献列表 | 从输入语言，经 IR 和优化，到设备产物。benchmark 的反馈用于挑选配置 | 顺着[旧 function.cc](../../triton-legacy/lib/runtime/function.cc)的 `make`、`make_ir`、`make_bin`、`precompile` 阅读。图左侧对接其他 DSL 的箭头是论文注明的范围外接口。贡献列表最后一项是实验，证据在 §6 |
| §1 末段 | 作者交代相关工作与结论的位置 | 这里跟着目录分别进入 §2 与 §7 |

<a id="listing-1"></a>
### Listing 1 怎样从第一行读到写回

论文的 `A` 和 `B` 分别有 `M×K` 和 `N×K` 个元素，输出 `C[m,n]` 是 A 第 m 行和 B 第 n 行沿 K 的乘加。把[历史 dot kernel](../../triton-legacy/tests/common/src/dot.h)与[现代矩阵乘](../../triton/python/tutorials/03-matrix-multiplication.py)并排看。两份源码都有可独立追踪的指针计算。

| Listing 1 的操作顺序 | 每句怎样改变程序状态 | v0.1 中找什么 |
|---|---|---|
| 定义 TM、TN、TK 和函数签名 | 前三行列编译候选；签名传基地址与矩阵大小。TM/TN 决定一份程序产出的行列数，TK 决定每轮计算多少个乘积项 | `dot.h` 的三组宏由[测试配置](../../triton-legacy/tests/common/dot.h)提供，函数还显式接收 stride 信息 |
| 得到 rm、rn、rk，清零累加块 | rm、rn 是输出行列坐标，rk 是一轮内从零开始的 K 坐标。每个输出位置先放 0 | `ridx/ridy` 保存块坐标，`rm = ridx * TM + 0 ... TM` 先算这块的起点，再加块内坐标。例如重排后的 ridx=1、TM=4 时得到 4–7。随后 `float c[TM, TN] = 0` 给输出块每个位置准备初始和 0 |
| 构造 pa、pb | 一维行索引增一个轴，再与 K 方向的偏移相加。结果的每项是一个元素地址 | `offa/offb` 是相对 A/B 基地址的元素数；把它们分别加到 A/B 后得到指针 tile。比如行步长 1、K 步长 5 时，行 2、K 坐标 3 的偏移是 `2 + 3*5 = 17` 个元素 |
| 循环、三组检查、两次载入 | 剩余 K 少于 TK 时，后几项越界；行列尾块也可能越界。用于 dot 的被遮住项取零 | 首轮 `checka ? *pa : 0` 按每项的真假决定读地址还是填 0。历史基准只启用 M=N=K=1024 的整除配置；首轮 mask 用 `rk < K`，循环底部用 `k > TK` 判断是否准备下一轮。两处检查的范围不同 |
| dot 与指针推进 | 当前块做乘加；下一轮需把两个源指针各沿 K 推进 TK 步 | 找 `USEA @ USEB` 和 `pa/pb += TK * STRIDE_*`，转置/操作数写法由配置宏和布局共同决定。论文清单中 pb 的初始 K 系数与推进系数不同，以基准展开为例，B 的地址每沿 K 增一格就加 ldb，推进 TK 格就是 `TK*ldb` |
| 重新构造输出地址、合并行列条件、写回 | 输出地址只由 rm、rn 与输出 stride 决定；只有行列都有效的元素才会写 | `offc` 逐项计算 `rxm*ldc+rxn`，`checkc` 逐项检查行小于 M 且列小于 N。若某项是第 2 行第 3 列且两个条件为真，`*?(checkc)pc = ...` 将相应结果写到 `C+2*ldc+3`；列越界时跳过该项 |

### Figure 1 的斜线和横线怎样读

Roofline 是一种上限模型。假设设备每秒能搬 100 GB，某计算每搬 1 GB 能做 0.02 万亿次浮点运算。如果一直以这个速度搬数据，最多能供给每秒 2 万亿次运算；这一段上限随横轴上的运算强度一起升高。假如设备本身最高只能算到 4 TFLOPS，越过交点以后，上限就停在 4。这是图中先斜后平的原因。读图时，先确认一个点代表某个输入形状在某个实现上的结果，再问该点距离上限有多远。

[历史基准的计时段](../../triton-legacy/tests/common/dot.h#L123)先测同一次 `function(args, grid, stream)` 的耗时，再用 `2.*M*N*K / nanosec * 1e-3` 折算吞吐。M×N 个输出各有 K 次乘加，一次乘加按两次浮点操作算。按纳秒相除得到每纳秒操作数，再乘 `1e-3` 得到每秒万亿次浮点操作。图里的 Roofline 和历史基准驱动分别提供硬件上限模型和实际测量入口，读它们时关注的变量不同。

<a id="read-operand-address"></a>
### 把论文的 pa 对照到 v0.1 的 offa

把[旧 dot 源码](../../triton-legacy/tests/common/src/dot.h)滚到 `// pointers to operands`。先只读 A 一边。源代码的核心是：

```c
int offa[SHAPE_A] = rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
TYPE* pa[SHAPE_A] = A + offa;
```

按[上面的宏展开](#read-stride-macro)，A 这边的 K 坐标用 `[newaxis, :]`，M 坐标用 `[:, newaxis]`。第一个把长度 TK 的索引排成一行，第二个把长度 TM 的索引排成一列。`:` 在这两处表示保留这一轴的全部位置，`newaxis` 新加长度为 1 的轴。两个结果相加时自动扩到同一个二维形状。

这里换一个容易画在纸上的布局，让 M 方向 stride 为 1、K 方向为 5，和上面实际基准的布局不同。令 TM=2、TK=3、`rm=[2,3]`、`rk=[0,1,2]`，对应 stride 分别为 1 和 5。stride 是沿一个逻辑轴前进一步时，地址要跨过几个元素。

```text
rm[:, newaxis] * 1     rk[newaxis, :] * 5     offa（逐位置相加）
[[2],                 [[0, 5, 10]]          [[2, 7, 12],
 [3]]                                         [3, 8, 13]]
```

因此 `pa` 的第二行第三列存的是 `A+13`，也就是距基址 13 个元素。如果元素是 4 字节，字节地址相差 52。这里还没有取数。到下一段 `TYPE a[SHAPE_A] = checka ? *pa : 0` 才读内存。`条件 ? 值1 : 值2` 的意思是条件为真取值 1，为假取值 2；此处条件按 tile 位置给出，真实前端如何保护读操作见[下一篇](02-ir.md#mask-to-builder)。

现在回到论文同一行看 `a + rm[:, newaxis] + rk * M`：两个式子都在组成二维地址表。v0.1 从 host 接收布局相关的 stride 再选中其值。沿同一个基准配置继续算 B：`BT=false`、`b_order={1,0}`、`sb={"1","ldb"}`，所以 `STRIDE_BK=sb[b_order[0]]="ldb"`、`STRIDE_BN=sb[b_order[1]]="1"`。B 的 `rk[:,newaxis]` 成一列，`rn[newaxis,:]` 成一行，于是 `offb[k,n]=k*ldb+n`。下一轮分别将 A 和 B 的指针推进 `TK*1` 和 `TK*ldb` 个元素。这样即使论文拼写和仓库不同，也可以沿着同一个地址问题继续读。

## §2 Related Work 与 Listing 2（PDF 第 2–3 页）

- 段首先从框架依赖线性代数库，转到当时三类实现路径。读三个条目时依次问「匹配已有运算模板」「用循环代数来变换计算」「合成循环并指定 schedule」分别在哪一层表达优化。
- 接下来的比较与 Listing 2 使用同一个矩阵乘数学关系。看 TF 的运算调用，再看 PlaidML、TC、TVM 写出的下标和归约；回看 Listing 1，Triton 程序员要写每块的地址和尾部处理。用[旧 dot kernel](../../triton-legacy/tests/common/src/dot.h)找这些额外决定。对外部 DSL 的评价按 2019 年论述阅读。

## §3 The Triton-C Language（PDF 第 3–4 页）

| 原文次序 | 操作怎样读 | 取证位置 |
|---|---|---|
| §3 开头，接口、语法、语义、执行模型 | 语法决定哪些文字能被接受；语义决定这些文字代表什么操作；执行模型决定数据怎样分给多个实例 | [parser.cc](../../triton-legacy/lib/lang/parser.cc)处理语法，[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc)构造 IR，[旧 CUDA 对照](../../triton-legacy/docs/tutorials/triton-vs-cuda.rst)解释 program 的分工 |
| §3.1 开头及 tile 声明项，Listing 3 | `[16,16]` 是一个二维 tile，range 的右端不包含在内。形状候选先选成具体整数再生成 IR | 见下方[按 token 读两个语法分支](#read-parser-expressions)。例如预处理先把 `0 ... TK` 里的 TK 换成 8，parser 收到的是数字与符号 |
| §3.1 内建操作项 | `+` 可以逐位置相加；dot 跨 K 归约；program 编号决定算哪块 | [VisitBinaryOp 的 @ 分支](../../triton-legacy/lib/lang/code_gen.cc#L47)为 dot 准备与结果同形状的 0 块并交给 `create_dot(lhs,rhs,_0)`。[VisitFuncCall](../../triton-legacy/lib/lang/code_gen.cc#L246) 遇到 `get_program_id(0)` 时，把常量轴 0 交给 builder 创建取编号的 IR。设备执行时，编号 1、块长 4 给全局索引 4–7 |
| §3.1 广播项 | `a[:, newaxis]` 给原来的一维值增加一列轴。比如 `[7, 9]` 变成 `[[7], [9]]`，随后可以沿列重复 | 这里的方括号跟在一个已有值后面，所以由后缀表达式 parser 处理。二维结果在[地址表例子](#read-operand-address)中直接算出来 |
| §3.1 谓词项 | 谓词是逐位置的真假条件。`mask=[true,false]` 可以只访问第一个地址 | 论文画出 `@predicate`；快照读[ParseUnaryOp](../../triton-legacy/lib/lang/parser.cc#L644)的 masked dereference 和[赋值分支](../../triton-legacy/lib/lang/code_gen.cc#L728)的 `create_masked_store` |
| §3.2.1 两项收益 | 第一项说源代码可以先写逻辑块；第二项说编译器掌握整块坐标后能决定合并访存或布局 | 从 `tile_type` 看保留的形状，再到[旧 Coalesce](../../triton-legacy/lib/codegen/transform/coalesce.cc#L64)看形状与访问模式怎样用于优化 |
| §3.2.2 形状约束、规则 1、规则 2 | 先从末轴对齐，秩较少的形状左补 1，再沿大小为 1 的轴补成目标尺寸。如果某轴两侧都大于 1 又不同，广播失败 | [GenBroadcastOp](../../triton-legacy/lib/lang/code_gen.cc#L518)对标量直接造 splat；对缺轴的值可以先造 reshape，再造 broadcast。比如 `[7,9]` 变成三行前，先得到一行两列，最后三行各为 `[7,9]`。现代对照是[broadcast_impl_value](../../triton/python/triton/language/semantic.py#L768) |
| Listing 4 三个加法 | 第一行显式把 `[16]` 写作 `[1,16]`，第二行隐式补轴，两者结果都是 `[32,16]`。第三行让 `[1,16]` 与 `[16,1]` 各沿自己的长度 1 轴扩展，合成 `[16,16]` | 在同一个 `GenBroadcastOp` 里检查补轴和扩轴的输入/输出类型。第 4 节 Figure 4 把两个方向分开画 |
| §3.3 第 1 段与 Figure 3 | SPMD 就是同一段程序运行许多份，各份用不同坐标。Triton 在源语言里给每份程序一块工作，编译器将它分给物理线程 | [旧 CUDA 对照](../../triton-legacy/docs/tutorials/triton-vs-cuda.rst)看两种写法各从哪里拿编号，再看[现代 program/CTA/warp](../chapters/00-foundations.md#sec-0-4) |
| §3.3 第 2 段 | 取全局坐标是为了把实例编号转换成具体的地址。例如两块每块 4 项，编号 1 的坐标为 4、5、6、7 | 旧[复制示例源码](../../triton-legacy/tests/common/src/copy.h)可追 `get_program_id` 到地址，现代[向量加法](../chapters/04-vector-add-jit.md#sec-4-1)手算最后一块 |

读完这篇，从 Listing 1 任取 `pa` 或 `check_c`，先说出它的形状和单项内容，再去看编译器怎样保存这些信息。下一篇从论文里打印出来的 IR 起步。

<a id="read-parser-expressions"></a>
### 按 token 读两个语法分支

预处理后的 `0 ... 8` 可以先写成三个记号：数字 0、运算符 `...`、数字 8。[ParseRangeExpr](../../triton-legacy/lib/lang/parser.cc#L699) 中，`lhs = ParseCastExpr()` 先读左操作数 0；`ts_.Next()` 从 token 序列拿下一个记号。`tok->tag_ == Token::ELLIPSIS` 检查它是不是三个点，命中后读右操作数 8。`BinaryOp::New(tok, lhs, rhs)` 做出带左右子节点的 range 节点。`ts_.PutBack()` 把循环多读的非 range 记号放回去，例如声明末尾的分号会由外层处理。这里只形成语法树。在 [VisitBinaryOp 的 ELLIPSIS 分支](../../triton-legacy/lib/lang/code_gen.cc#L65)，前端要求左右两个 IR 值是整数常量，才用 `make_range::create` 创建 IR 操作。

而 `float a[2,3]` 是在声明一个名字。进 [ParseArrayFuncDeclarator](../../triton-legacy/lib/lang/parser.cc#L1777)，`ts_.Try('[')` 在当前位置尝试消费左方括号，成功后 `ParseTileShape()` 读取两个尺寸，`ts_.Expect(']')` 要求下一段用右方括号收尾。最后的 `TileType::New(shape, base)` 把 `[2,3]` 和元素的前端类型 float 关联起来，交回声明解析。到这里你已经知道 shape 从哪里出现，接下来用[类型走读](02-ir.md#read-tile-type)看同样的数字如何进入 IR。
