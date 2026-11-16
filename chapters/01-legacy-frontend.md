# 第 1 章 历史路线（一）：Triton-C 与 parser

[返回总目录](../README.md) · [上一章：基础](00-foundations.md) · [下一章：旧 IR 与优化](02-legacy-ir-passes.md)

从这一章开始回到 2019 论文时代。固定的历史快照是官方 v0.1（2020-05-04），它晚于论文，保留 Triton-C 和自有 IR，但**不是已证实的论文实验精确版本**。

本章包含两部分：

- **1.1**：用同一个 N=10、TILE=4 的加法读 Triton-C 语言、masked dereference 和 host 选项；
- **1.2**：追源码字符串怎样经过预处理器、scanner、parser 变成 AST，再交给 IR 生成器。

历史路线的完整对象链如下，本章只走到 AST：

```text
Triton-C 源码字符串 + TILE/num_warps 选项
  → Preprocessor → TokenSequence
  → Parser → AST（TranslationUnit / FuncDef / BinaryOp）
  → make_ir → 旧 Triton-IR
  → make_bin 的 pass → LLVM → PTX → driver → caller
```

**读法：** 不要用现代 Python 语法去“翻译”旧 C 代码。先把旧语言当作一种独立的 tile DSL：`0 ... TILE` 是 range，`*?(check)` 是 masked dereference，TILE 是编译宏。遇到与论文的差异，保留差异，不要强行对齐。

**运行边界：** 本机没有构建 2020 v0.1，它需要当时的 LLVM 8 和 CUDA 工具链。第 1–3 章以固定源码的静态核对与纸笔推演为主；第 4 章起的现代实现有从固定提交构建后的实际运行记录。

## 本章地图

| 部分 | 输入 | 输出 | 读完后能回答 |
|---|---|---|---|
| 1.1 | 一份 Triton-C 源码字符串、TILE/num_warps 选项 | 对每个 program 的地址/掩码/读写语义的理解 | `float* px[TILE]` 和 `*?(check)` 到底在说什么 |
| 1.2 | 字符串 + 宏 + runtime::function | token 序列 → AST（TranslationUnit/FuncDef） | TILE 在哪一步展开、AST 节点怎样接住参数和函数体 |

<a id="sec-1-1"></a>
## 1.1 把现代向量加法换成 Triton-C 来读

这一部分只闭合一件事：给定 x、y、输出地址和 N，旧 Triton-C 怎样描述每个 program 的加法，host 又怎样决定启动范围。
沿用全书统一的纸笔输入：N=10，x 为 0 到 9，y 为 1 到 10；期望输出为 1、3、5、…、19。
为了列出每个元素，纸笔推演令 TILE=4。原教程用 1024；我们没有运行这两个配置。

历史源码固定为官方 v0.1、提交 `b9e36331be854e37aa14c538861689533ec0dfb0`（2020-05-04）。
它晚于 2019 论文，不能当作论文实验精确版本。
摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。本章只核对静态源码，没有安装、构建或 GPU 实测。

### 1. 先看完整 kernel，再拆每行

来源：[custom-operation.rst](../../triton-legacy/docs/tutorials/custom-operation.rst)，第 15–27 行。
下面是删除注释、整理缩进后的上游语句：

```c
__global__ void add(float* z, float* x, float* y, int N){
    int pid = get_program_id(0);
    int offset[TILE] = pid * TILE + 0 ... TILE;
    float* pz[TILE] = z + offset;
    float* px[TILE] = x + offset;
    float* py[TILE] = y + offset;
    bool check[TILE] = offset < N;
    *?(check)pz = *?(check)px + *?(check)py;
}
```

这段是交给 Triton 编译器的语言，不是交给系统 C/C++ 编译器的普通 C 程序。
`0 ... TILE` 和 `*?(check)` 都是旧 Triton 扩展。
现代 `@triton.jit` 的 Python 函数不能直接粘贴这段语法。

甚至论文和这份历史快照也要分开：2019 论文 Listing 1/3 使用 `kernel`、`tunable`、`get_global_range` 和 `@predicate` 等表达。
本教程快照使用 `__global__`、host 提供的 TILE 宏、`get_program_id` 与 `*?(check)`。
例如论文的语句谓词前缀 `@check_c`，不能未经 parser 核对就原样当成本快照的 masked store 写法。
这项语法差异来自[作者论文](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) PDF 第 2–3 页；后文固定按本地源码解释。

`__global__` 标记设备入口；`void` 表示函数不通过返回值交出计算结果。
结果经 z 指向的存储写回。z、x、y 是三个指针参数，N 是整数参数。
`float* x` 的意思是“x 保存一个 float 元素的地址”，不是“把所有输入元素复制进形参”。
函数名 add 与 host 中叫 add 的包装器可同名，但承担不同角色。

约定三个数组连续、dtype 为 float32、设备一致，输出另行分配。
旧示例只显式检查 x 的 dtype；这不足以替代完整输入合同。
本章纸笔计算依赖上述合同，不声称示例已经检查了所有条件。

**确认：** `void` 是否表示没有输出？不是。函数通过 `pz` 写内存，返回类型只说明没有一个直接返回的数值。

### 2. pid 是一份工作的编号

第 17 行：

```c
int pid = get_program_id(0);
```

右边返回第 0 个网格维度中的 program 编号，保存在标量整数 pid 中。
我们的 grid 会有三份工作，所以 pid 分别是 0、1、2。
三个 program 各执行同一段源码，pid 的不同让它们负责不同地址。

program 是语言层的一份块级工作。
它不是一个线程编号，也不意味着 GPU 只用一个物理线程执行它。
后续布局分析才决定 tile 中的元素由哪些线程、寄存器持有。
此刻无需猜测 TILE=4 会选择怎样的真实线程布局。

> 回到论文：§3.3、Figure 3，PDF 第 4 页，解释程序实例与并行执行模型。这里使用 2020 源码帮助理解该抽象，不证明两版线程映射算法一致。定位见[论文表](../appendices/reference.md#sec-a-1)。

### 3. 一次构造四个偏移

第 19 行：

```c
int offset[TILE] = pid * TILE + 0 ... TILE;
```

先按这份语言的语义读右边：`0 ... TILE` 表示 0 到 TILE−1，右端不包含。
TILE 不是 GPU 上临时读取的数组长度，而是 host 编译选项定义的宏常量。
`pid * TILE` 是标量起点；标量与长度为 TILE 的 tile 相加时，标量广播到每个位置。
广播不是把输入数组复制 TILE 遍，而是把同一个标量参与每个位置的运算。

左边 `offset[TILE]` 是带形状的逻辑 tile。
它在语法上像 C 数组，但这里的加法、比较和指针运算按 tile 整体表达。
后端可能把它拆成多个标量寄存器；不能据方括号推断一定分配线程私有数组。

纸笔结果如下，列号 j 表示 tile 内第几个位置：

| pid | 起点 pid×4 | j=0 | j=1 | j=2 | j=3 |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 | 2 | 3 |
| 1 | 4 | 4 | 5 | 6 | 7 |
| 2 | 8 | 8 | 9 | 10 | 11 |

注意第三行仍然生成 10、11。range 的长度由 TILE 决定，不会因为 N=10 自动变短。
下一步产生掩码，负责保护这两个位置。

> 回到论文：§3.1、Listing 3 与 §3.2.1，PDF 第 3 页；§3.2.2、Listing 4，PDF 第 3–4 页。关注 tile、range 和广播的表达能力，具体语法以本章固定快照为准。

### 4. 类型化指针怎样按元素前进

第 20–22 行连续创建三个指针 tile：

```c
float* pz[TILE] = z + offset;
float* px[TILE] = x + offset;
float* py[TILE] = y + offset;
```

z、x、y 各是一个基地址；offset 是刚得到的四个整数。
以 x 为例，`x + offset` 广播 x，再分别计算四个元素的地址。
结果不是四个输入数值，而是四个“到哪里读”的地址。
`float* px[TILE]` 也应读成“由 float 指针组成的 tile”，不能把它误认成一个 float 值 tile。

假设 x 的基地址为 1000（纯纸笔地址），float32 占 4 字节：

```text
pid=1：offset = [4, 5, 6, 7]
px 对应字节地址 = [1016, 1020, 1024, 1028]
```

指针的 `+4` 是前进四个 float，即 16 字节。
编译器通过指针元素类型知道这个倍率。
如果改成 half 指针，元素字节数变了，不能仍套用 4 字节。
这就是“类型化指针”的用途之一。

即使纸笔列出越界位置的候选地址，也不能真的去解引用。
地址计算与读取内存是两类操作，下面的掩码控制后者。

**确认：** 执行 `float* px[TILE] = x + offset` 已经读到 x 的数值了吗？没有，表达的只是地址计算。

### 5. 布尔 tile 控制每个位置是否访问内存

第 24 行：

```c
bool check[TILE] = offset < N;
```

N 是标量 10；比较按位置进行，结果是四个真假值。
pid=0 与 pid=1 得到 `[true,true,true,true]`。
pid=2 得到 `[true,true,false,false]`。
这里的 bool 在 runtime 添加的 preheader 中宏定义为 `_Bool`，不是现代 Python 的 bool 对象。
preheader 的连接在 1.2 解释。

最后第 26 行：

```c
*?(check)pz = *?(check)px + *?(check)py;
```

从右往左读数据依赖：`*?(check)px` 按 check 读取 px 中有效地址；py 同理。
两个读取结果是 float tile，`+` 逐元素求和。
左边是带相同 check 的写入，只有 true 的位置写到 pz 指向的地址。
`?` 在这里属于旧 masked dereference 语法，不是完整的 C 三目表达式 `条件 ? a : b`。

pid=2 的纸笔推演为：

```text
offset      [8, 9, 10, 11]
check       [T, T, F, F]
有效 x       8  9
有效 y       9 10
有效 sum    17 19
有效写地址   z[8], z[9]
```

不要给后两个位置补写“读到 0”。
真实前端在 [code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) 第 58–63 行给 masked load 构造的 false value 是 `undef_value`，tile 时再 splat。
它没有现代 `tl.load(..., other=0)` 的显式零填充保证。
本例中无效位置的结果不写回，所以无需赋予它们确定数值。
若后面改成归约，无效值就可能影响有效结果，必须重新设计边界处理。

**确认：** 只给 store 加 check 足够吗？不足够；输入 load 也必须避免访问 x[10]、x[11]。

### 6. 把 host 的选项与 kernel 形参对上

来源仍为 [custom-operation.rst](../../triton-legacy/docs/tutorials/custom-operation.rst)，第 59–78 行。
原文第 61 行在 Python 实参中写了分号，是语法笔误。
下面是**教学整理并修正该标点**，不是本机执行成功的日志：

```python
kernel = triton.kernel(src, defines={'TILE': 1024}, num_warps=[4])
z = torch.empty_like(x).cuda()
N = x.numel()
grid = lambda opt: (triton.cdiv(N, opt.d('TILE')), )
_add.kernel(z, x, y, N, grid=grid)
return z
```

`src` 来自第 44–58 行的三引号字符串，内容就是前面的 Triton-C 函数。
Python 创建字符串时不会执行其中的加法，Triton 编译器以后解析它。
`defines` 将 TILE 交给预处理器；`num_warps=[4]` 表达可选 warp 数集合。
TILE 影响逻辑 tile 长度，num_warps 影响线程资源与布局；二者不是同一个数量。

`torch.empty_like(x)` 准备与 x 形状、类型相同的输出存储；empty 的原值不可依赖。
`.cuda()` 是历史示例中的设备转移写法。
N 来自 x 的元素数，所以这里 N=10。
`lambda opt: ...` 创建一个稍后接收选项对象 opt 的函数。
`opt.d('TILE')` 取得这组选项中的 TILE，因为不同候选 TILE 需要不同的 grid。
`(数值, )` 是单元素 tuple，表示一维网格。

本章纸笔改成 TILE=4 时，grid 返回 `(ceil(10/4),)=(3,)`。
若保持教程 TILE=1024，则 grid 是 `(1,)`，一个逻辑 tile 只有前十个位置有效。
这两个配置都不是本章实测推荐值。

调用 `_add.kernel(z, x, y, N, grid=grid)` 的四个位置实参，依次对应：

| host 实参 | Triton-C 形参 | 含义 |
|---|---|---|
| z | `float* z` | 输出地址 |
| x | `float* x` | 第一个输入地址 |
| y | `float* y` | 第二个输入地址 |
| N | `int N` | 有效元素数 |

`grid` 是 host 的启动参数，不是 kernel 的第五个形参。
TILE 是编译宏，也没有占据第五个运行时位置参数。
host `return z` 返回张量对象；设备函数 `void add` 则通过地址写数据。
host 返回对象与 GPU 完成所有写入也不是同一事件。

### 7. 找到 runtime 里真正的调用分支

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 455–469 行，以下摘录省略了调优缓存中间部分：

```cpp
if(callers_.empty())
  precompile(stream, opt_);
// 省略：根据参数查找或调优，取得 it
(*it->second)(stream, grid_fn(it->second->opt()), args);
```

`callers_` 保存可调用的已编译候选；空集合需要先编译。
`stream` 是提交工作的流，`opt_` 是 function 构造时保存的选项空间。
最后一行先从候选取得它的 opt，传给 grid_fn；返回的 grid 与 args 一起交给 caller。
这里正解释了 Python 示例为何让 grid 是“选项的函数”。
具体包装到 C++ 的绑定不在本章闭环内；3.2 会接着读 caller、候选空间与设备提交。

旧 [triton-vs-cuda.rst](../../triton-legacy/docs/tutorials/triton-vs-cuda.rst) 的另一个向量加法片段还存在多余星号的笔误。
历史文档适合核对语言思想，不能因代码印在教程里就声称可直接运行。
本章选择 custom-operation 的 kernel，并明确披露 host 标点修正。

### 8. 带到下一部分的对象与自测

现在已有两种不同对象：host 上的 Triton-C 字符串 src，以及 GPU 上 x、y、z 的存储。
编译器读取前者生成程序；运行程序时才访问后者。
本章第二部分（1.2）从这个 src 开始，追踪它怎样变成 token 和 AST。

1. pid=1、TILE=4 时 px 指向哪些元素？答案：x[4] 到 x[7]。
2. pid=2 有几个有效写入？答案：两个；分别写 z[8]=17、z[9]=19。
3. TILE 与 N 是否都作为 int 形参传入？答案：只有 N；TILE 由编译选项预处理。
4. 为什么不把 false load 位置写成零？答案：本快照构造 undef；现代 other=0 不能倒推为旧语义。
5. 本章是否证明了历史版本可以运行？答案：没有，只静态核对语句与纸笔索引，未构建或执行。

**第一部分交出的边界。** 1.1 读的是“用户写的 Triton-C 源码”和“host 交给运行时的选项”。这份源码此刻还只是一个字符串，里面的 TILE 宏、`0 ... TILE`、`*?(check)` 都没有被解析。下一部分从 `function::make` 开始，看预处理器怎样展开宏、scanner 怎样切 token、parser 怎样构造 AST，最后把 AST 根交给 `make_ir`。

<a id="sec-1-2"></a>
## 1.2 源码字符串怎样长成一棵 AST

1.1 留下了 `src` 字符串、TILE 宏选项，以及运行时才使用的 x/y/z 地址。
本章跟踪字符串进入 `function::make`，直到函数体成为 AST，再把 AST 交给旧 IR 生成器。
这段路都在 host 编译阶段发生，没有读取 GPU 输入元素。
仍使用 N=10、TILE=4 的纸笔配置；它只是帮助辨认常量，未运行旧编译器。

固定历史版本为 `b9e36331be854e37aa14c538861689533ec0dfb0`，2020-05-04 v0.1。
它不是已证实的 2019 论文实验版本。代码许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. function 保存的是什么

[function.cc](../../triton-legacy/lib/runtime/function.cc)，构造函数第 438–452 行。
下面是**非连续摘录**，省略计算源码哈希与创建缓存目录的语句：

```cpp
function::function(const std::string &src,
                   const options_space_t& opt,
                   const std::string &cache_ref):
    src_(src), opt_(opt), cache_ref_(cache_ref) {
  // 省略：哈希与缓存路径
  src_ = preheader() + src_;
}
```

`const std::string &src` 是只读字符串引用形参；调用者的字符串通过它交进来。
冒号之后的 `src_(src)` 是 C++ 成员初始化，给成员 src_ 保存一份字符串内容。
`opt_(opt)` 保存候选选项空间，不是已经选好的某一个 TILE 配置。
`cache_ref_` 保存判断缓存有效性所需的引用文件名。

最后一行把 `preheader()` 的返回字符串拼到用户源码前面。
同文件第 386–417 行的 preheader 包含 `#define bool _Bool`、整数别名和 `get_program_id` 的 extern 声明。
因此 1.1 源码里的 bool 和内建函数并非全部靠 parser 特判：一部分由这段前置文本提供声明。
拼接后成员仍是字符串，尚未成为 AST，更不是设备机器码。

### 2. 为什么先说明缓存分支

`function::make(stream, opt)` 的 opt 是某一个具体选项，而非整个候选空间。
stream 提供当前设备 context；3.2 解释 precompile 怎样枚举具体 opt。
[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 254–268 行先检查 PTX 缓存。
其中第 266–267 行：

```cpp
if(ifs)
  return new caller(stream->context(), ifs, opt);
```

ifs 是刚打开的缓存输入流。它有效时可以从已有产物建立 caller 并提前返回。
所以并非每次启动都会重新扫描和解析源码。
本章继续的是没有可复用产物、需要编译的路径。
不要把下面的解析流程当成每次 GPU launch 的固定运行开销。

### 3. 宏值进入 token 序列

同文件第 269–277 行，连续摘录：

```cpp
TokenSequence tokens;
Preprocessor cpp(&src_, true);
for(auto it: opt.defines)
  cpp.AddMacro(it.first, &it.second);
cpp.Process(tokens);
Parser parser(tokens);
parser.Parse();
```

第一行在栈上创建 tokens，它承接预处理输出。
token 是带种类、文字和源码位置的信息，例如标识符 `offset`、整数 `4`、加号、左括号。
tokens 不是字符串数组 x，也不保存输入浮点数。

`&src_` 取源码字符串对象地址，让 cpp 能读取它。
循环中的 it 是 defines 中的一对键值；first 是宏名，second 是宏文本。
在纸笔配置中，可以把它理解为名字 TILE、文本 `"4"`。
`AddMacro` 并不是对 GPU 写入一个叫 TILE 的变量，而是在预处理器注册替换规则。

`cpp.Process(tokens)` 把输出写进 tokens，函数通过引用形参修改这个对象。
它不是通过 `return` 返回 token 集合，所以前一行不能改写成“tokens 接收返回值”。
`Parser parser(tokens)` 随后构造 parser，并让它访问同一个 token 序列。
最后 `parser.Parse()` 改变 parser 持有的 AST 结构，不返回 GPU 数值。

### 4. Process 里面还发生了什么

[cpp.cc](../../triton-legacy/lib/lang/cpp.cc) 第 248–258 行，原样摘录：

```cpp
void Preprocessor::Process(TokenSequence& os) {
  TokenSequence is;
  // Add source file
  if(fName_)
    IncludeFile(is, fName_);
  else
    IncludeSrc(is, fSrc_, nullptr);
  // Expand
  Expand(os, is);
  Finalize(os);
}
```

os 是前面 tokens 的另一个名字，因为它是引用形参。
is 是预处理内部的初始 token 序列。
当前走源字符串输入，`IncludeSrc` 以 fSrc_ 所指的文本产生初始 token。
若是文件输入则走 IncludeFile，两种入口在这里合流。
`Expand(os, is)` 从 is 处理宏展开，把结果写入 os。
`Finalize(os)` 再识别关键词等；同文件第 227–238 行把 identifier 文本与关键词表对照。

扫描器的基本循环在 [scanner.cc](../../triton-legacy/lib/lang/scanner.cc) 第 7–23 行。
下面摘出连续的核心循环开头：

```cpp
void Scanner::Tokenize(TokenSequence& ts) {
  while (true) {
    auto tok = Scan();
    if (tok->tag_ == Token::END) {
```

`Scan()` 每次认出一个 token，`tag_` 是其种类。
非 END 的分支在第 18–21 行将 token 追加到 ts；END 分支补齐必要换行后退出。
因此扫描解决“这几个字符是什么符号”，parser 接下来解决“这些符号怎样组成声明和表达式”。
例如 `0 ... TILE` 经宏展开后可概念化为整数 0、ELLIPSIS、整数 4。
这是 token 示意，不是实际打印的 dump，也不包括全部位置信息。

### 5. ts_ 与 unit_ 从哪里来

[parser.h](../../triton-legacy/include/triton/lang/parser.h) 第 41–52 行，删去其他成员初始化的摘录：

```cpp
explicit Parser(TokenSequence& ts)
  : unit_(TranslationUnit::New()),
    ts_(ts),
    // 省略：作用域与控制流状态成员
    defaultLabel_(nullptr) {
      ts_.SetParser(this);
    }
```

形参 ts 对应刚才传入的 tokens。
成员声明在第 240–242 行：unit_ 是 `TranslationUnit*`，ts_ 是 `TokenSequence&`。
星号表示指针，保存对象地址；引用表示绑定已有对象，不另拷贝一份 token 序列。
`ts_(ts)` 因而让成员 ts_ 引用 tokens。
`this` 是当前 parser 自己的地址；SetParser 把解析器地址交给 token 序列。

`TranslationUnit::New()` 建立 AST 根节点，unit_ 保存根地址。
TranslationUnit 是“翻译单元”，这里可以理解为本次源码中的顶层声明列表。
它不是一个 GPU program 实例，也不是 LLVM module。

第 226 行的访问函数 `TranslationUnit* Unit() { return unit_; }` 返回同一根指针。
后来 Generator 通过它读树；这次调用没有复制一棵新树。

### 6. Parse 怎样找到函数定义

[parser.cc](../../triton-legacy/lib/lang/parser.cc) 第 58–60 行：

```cpp
void Parser::Parse() {
  DefineBuiltins();
  ParseTranslationUnit();
}
```

先补充内建符号，再从顶层读完整 token 流。
`ParseTranslationUnit` 第 73–90 行的关键部分，省略检查空标识符分支：

```cpp
int storageSpec, funcSpec, align;
auto declType = ParseDeclSpec(&storageSpec, &funcSpec, &align);
auto declInfo = ParseDeclarator(declType);
// 省略：取 tok/type/attrs，处理无名字的声明
auto ident = ProcessDeclarator(tok, type, attrs, storageSpec, funcSpec, align);
type = ident->Type();
if (tok && type->ToFunc() && ts_.Try('{')) {
  unit_->Add(ParseFuncDef(ident));
```

`ParseDeclSpec` 读取 void、float 等声明说明，输出参数还承接存储/函数说明和对齐信息。
`ParseDeclarator` 在基础类型上接名字、指针、参数列表等结构。
`ProcessDeclarator` 将名字与类型放入符号处理流程，返回 ident。
ident 是 AST 层标识符对象，不是函数真正的 GPU 地址。

`type->ToFunc()` 检查类型能否当作函数类型；`Try('{')` 尝试消费函数体左花括号。
两项都满足时调用 ParseFuncDef；其返回的 FuncDef 指针由 Add 追加到 unit_。
preheader 的 extern 声明也会被解析，但没有函数体，不走这个分支。

### 7. FuncDef 怎样接住参数与函数体

同文件第 109–126 行，删去报错分支后的**非连续摘录**：

```cpp
FuncDef* Parser::ParseFuncDef(Identifier* ident) {
  auto funcDef = EnterFunc(ident);
  // 省略：重复定义检查
  auto funcType = ident->Type()->ToFunc();
  funcType->SetComplete(true);
  // 省略：形参必须有名字的检查
  funcDef->SetBody(ParseCompoundStmt(funcType));
  ExitFunc();
  return funcDef;
}
```

ident 来自上一节 ProcessDeclarator，携带 add 名字和函数类型。
EnterFunc 创建/进入该函数的解析状态；函数类型保存 z、x、y、N 的形参声明。
SetComplete 标记现在读到的是定义，不能再允许另一个同名函数体重复定义。
ParseCompoundStmt 接收 funcType，使函数参数能进入函数体作用域。
它返回一个 CompoundStmt 指针，再由 SetBody 保存到 FuncDef 的 body_ 成员。
ExitFunc 结束解析状态；return 交回的是编译期树节点。

同文件第 2322–2342 行的函数体循环，省略 EOF 报错：

```cpp
EnterBlock(funcType);
std::list<Stmt*> stmts;
while (!ts_.Try('}')) {
  // 省略：EOF 检查
  if (IsType(ts_.Peek()))
    stmts.push_back(ParseDecl());
  else
    stmts.push_back(ParseStmt());
}
auto scope = curScope_;
ExitBlock();
return CompoundStmt::New(stmts, scope);
```

EnterBlock 在第 48–54 行把函数参数插入当前块作用域。
所以解析 `x + offset` 时，x 可对应到函数形参声明，offset 可对应到前面的局部声明。
Peek 只看下一个 token；Try 匹配并前进。
读到类型开头时走 ParseDecl；其他情况走 ParseStmt。
stmts 依次保存 pid、offset、px 等声明以及最后的赋值表达式。
ExitBlock 恢复外层作用域；CompoundStmt 保存这组语句与本块作用域信息。

### 8. 一个 range 与一个 masked load 的树形状

[parser.cc](../../triton-legacy/lib/lang/parser.cc) 第 699–708 行构造 range 的核心：

```cpp
auto lhs = ParseCastExpr();
auto tok = ts_.Next();
while (tok->tag_ == Token::ELLIPSIS) {
  auto rhs = ParseCastExpr();
  lhs = BinaryOp::New(tok, lhs, rhs);
  tok = ts_.Next();
}
ts_.PutBack();
return lhs;
```

解析 `0 ... 4` 时，lhs 先是常量 0，rhs 是常量 4。
BinaryOp::New 收到操作 token 与两个子表达式，返回 range 节点地址。
lhs 随后改为指向整个新节点；返回值交给更外层加法等解析函数。
PutBack 把不属于 range 的下一个 token 放回，留给外层语法处理。

同文件第 649–660 行的 ParseDerefOp 先解析可选的 `?(pred)`，再读地址 addr。
第 658 行明确构造 `BinaryOp::New(tok, Token::MASKED_DEREF, pred, addr)`。
所以旧 masked dereference 的左孩子是 predicate，右孩子是 address。
这项顺序会在 2.1 的 `create_masked_load(rhs, lhs, false_value)` 再出现。

下列是 AST **教学示意**，不是打印输出：

```text
TranslationUnit
  FuncDef(add)
    CompoundStmt
      Declaration(offset, ... BinaryOp(ELLIPSIS, 0, 4) ...)
      ...
      BinaryOp(=, masked_pz, BinaryOp(+, masked_px, masked_py))
```

### 9. “保存指针”不等于“负责自动释放”

[ast.h](../../triton-legacy/include/triton/lang/ast.h) 第 772–819 行：FuncDef 保存 ident_、body_ 等裸指针；TranslationUnit 用 list 保存 ExtDecl*。
这能说明树的引用关系，不能据此画成 unique_ptr 所有权树。
Parser 的析构函数在 [parser.h](../../triton-legacy/include/triton/lang/parser.h) 第 55 行是空函数。

真实节点构造可看 [ast.cc](../../triton-legacy/lib/lang/ast.cc) 第 1089–1092 行：

```cpp
auto ret = new (funcDefPool.Alloc()) FuncDef(ident, retLabel);
ret->pool_ = &funcDefPool;
return ret;
```

这是 placement new：先从静态内存池取得空间，再在该空间构造对象。
它与普通 `new FuncDef(...)` 的分配方式不同。
该文件第 9–27 行为多种节点设静态池；TranslationUnit::New 则直接 new 根节点。
因此本章只确认 parser/AST 的访问与构造关系，不声称 Parser 离开作用域会递归释放所有节点。
这一历史实现不能作为现代 C++ 自动所有权管理的示范。

### 10. 把树交给下一章

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 279 行调用 `auto ir = make_ir(parser)`。
parser 以引用传入；make_ir 在它仍存活时创建 Generator 并同步读取 AST。
Generator 的 `parser_` 是借用指针，`Unit()` 返回根地址；它不接收 GPU 数组内容。
下一章解释真正创建旧 IR 的那几行。

> 回到论文：§3.1–§3.3，PDF 第 3–4 页，说明语言表达与程序模型。论文没有替本章这份 parser 提供逐行所有权证明；这里的 ts_/unit_/FuncDef 关系来自固定 2020 源码。

自测：TILE 展开在哪一步？在预处理 Expand 中，不在 GPU 执行时。
ParseFuncDef 返回什么？FuncDef*，由根节点列表保存。
tokens 与 ts_ 是两个独立序列吗？不是，ts_ 引用 tokens。
AST 已经决定哪些元素进哪个寄存器吗？没有，布局在旧 IR 后续分析阶段处理。

## 第 1 章小结

| 阶段 | 输入 | 输出 | 谁创建/持有 |
|---|---|---|---|
| 源码字符串 | 用户写的 Triton-C + `src_` | `preheader()` 拼接后的字符串 | `runtime::function` |
| 预处理 | 字符串 + `opt.defines` 宏 | `TokenSequence` | `Preprocessor::Process` |
| 扫描 | 字符流 | token（种类、文字、位置） | `Scanner::Tokenize` |
| 解析 | token 序列 | AST（`TranslationUnit`、`FuncDef`、`BinaryOp`） | `Parser::Parse` |
| 交给下一步 | AST 根 | `Parser&` / `unit_` | `function::make_ir` 的实参 |

### 综合检查

1. `TILE=4` 是在 GPU 运行时读取的吗？不是；它是编译选项/宏，预处理阶段就展开进 token。
2. `tokens` 和 `ts_` 是两份拷贝吗？不是；`ts_` 是 `TokenSequence&`，引用同一份序列。
3. `Parser::Parse()` 返回 AST 吗？它修改 parser 持有的 `unit_`；后续通过 `Unit()` 取得根指针。
4. `*?(check)pz` 中 `?` 是三目运算符吗？不是；它是旧 masked dereference 语法，`check` 是逐元素谓词。
5. 旧 parser 的 AST 已经决定线程布局吗？没有；布局在旧 IR 的分析阶段处理。
6. 为什么不能把旧 Triton-C 直接贴进现代 `@triton.jit`？语法、预处理、类型系统和 IR 都不同；两者只是解决相似问题。
