# 第 4 章 向量加法：从 host 调用到 GPU 启动

[返回总目录](../README.md) · [上一章：旧布局与后端](03-legacy-layout-backend.md) · [下一章：前端与 IR](05-frontend-ir.md)

这一章只追一次调用：`add(x, y)`。x 为 0 到 9，y 为 1 到 10，N=10，教学配置 B=4。读完后，你应该能指出：

- host 在什么时候分配输出、计算 grid、把哪些对象交给 Triton；
- kernel 的每个 program 怎样选中自己的元素、构造 mask、load/add/store；
- `add_kernel[grid](...)` 的方括号和圆括号各自做了什么；
- JIT 怎样绑定参数、查缓存、构造 `ASTSource`、编译、装载和发射；
- 返回的 Python 张量为什么可能还没算完。

本章分两部分：第一部分先走计算语义，第二部分从同一行调用继续走运行时。两部分首尾相接，中间没有新的数学，只有同一次调用的后半段。

## 本章地图

| 部分 | 你跟踪的对象 | 读完能回答 |
|---|---|---|
| 4.1 | 十个输入数、host 的 `add`、kernel 的三个 program | 十个输出是怎样被算出来并写回的 |
| 4.2 | `add_kernel[grid]` 之后的 binder、缓存、`ASTSource`、`CompiledKernel`、launcher | 同一次调用怎样变成一次设备发射 |
| 4.3 | 从固定提交构建的 Triton 与 A800 | 实际 grid、输出、编译阶段和资源计数 |

**读法：** 第一遍不要试图记住 `jit.py` 的每个函数名。跟着同一组 x/y/output/N/B 走，每到一个新对象就问：它由谁创建、保存什么、交给谁。4.2 会反复使用 4.1 已经建立的这组对象。

<a id="sec-4-1"></a>
## 4.1 从十个数出发，逐行走完一个 Triton 向量加法

本章只解决一个问题：**两个各含十个数的输入，怎样经过 Python 的 `add(x, y)`，变成 GPU 内存中的十个相加结果？** 你不需要先读懂整个仓库。本章会把必要代码贴出来，逐段接起来；源码链接供你同时打开核对。

先走完整个算子的使用与计算过程。`@triton.jit` 内部怎样编译、怎样调用驱动，交给 4.2 继续；本章不会用一串未解释的编译器名词跳过它，也不假装已经讲完它。

### 阅读导航与代码约定

| 阅读段落 | 跟踪的对象 | 可以停下来确认什么 |
|---|---|---|
| 1–3 | 十个输入数、host 与 kernel 两段代码 | 数学目标和两段代码分别负责什么 |
| 4–7 | `add` 中的输出、N、grid 与调用实参 | 参数怎样从调用者到达 kernel |
| 8–12 | 一个 program 内的索引、指针、数值 | 每一行到底改变了什么 |
| 13–15 | 三个 program 与返回值 | 边界是否安全，结果什么时候可用 |
| 16–18 | 论文回读、CPU 练习、自测 | 能否自己走完另一个小输入 |

本章固定源码 SHA：`46eccb199bfd3f3559b512c4a15870bd45e6a74d`，主要读 [01-vector-add.py](../../triton/python/tutorials/01-vector-add.py)。下面“上游第几行”均指这个文件在该提交中的实际行号；跨文件时会重新注明。

代码块有三类：**上游摘录**保留原语句；**教学整理**只删除注释、调整排版或明确改写；**纸笔推演**是为理解语义算出的值，不是运行日志、IR dump 或线程分布。代码摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. 先知道我们希望算出什么

#### 1.1 输入和输出都是一组数

先不考虑 GPU，给出两个一维输入。下面是数值示意，方括号只是把元素列出来：

```text
索引 i       0  1  2  3  4  5  6  7  8   9
输入 x       0  1  2  3  4  5  6  7  8   9
输入 y       1  2  3  4  5  6  7  8  9  10
期望 out     1  3  5  7  9 11 13 15 17  19
```

“向量加法”在这里就是相同位置相加。例如索引 3 的输出是 `x[3] + y[3] = 3 + 4 = 7`。不是把 x 的所有元素先加成一个数，也不是把两组数拼接起来。

用普通 Python 表达数学目标，可以写成下面的**CPU 列表示意**：

```python
x_values = list(range(10))
y_values = list(range(1, 11))
expected = [x_values[i] + y_values[i] for i in range(10)]
```

第一行产生整数 0 到 9；第二行产生整数 1 到 10。`range` 的右端不包含在内。第三行依次取相同索引的两个元素相加，收集成新列表，所以得到 `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]`。

这三行只说明数学目标。真正的 Triton 输入是 GPU 上的张量，不是这两个 Python list。

#### 1.2 把“张量”落实到本例

张量在这里可以先理解为“带有形状、元素类型和存储位置的一组数”。本例约定：

| 属性 | x 和 y 的约定 | 为什么要知道 |
|---|---|---|
| shape，形状 | `(10,)`，一维、十个元素 | 决定有效索引是 0 到 9 |
| dtype，元素类型 | `torch.float32`，每个元素 4 字节 | 决定指针跨一个元素时前进多少字节 |
| stride，步长 | `(1,)`，相邻逻辑元素在内存中也相邻 | 才能直接用 `基指针 + 索引` 找元素 |
| device，设备 | 同一张活动 GPU | kernel 要在能访问这些输入的设备上执行 |

`(10,)` 不是坐标，而是形状；逗号让它成为 Python 的单元素元组。稍后 grid 的 `(3,)` 也用这种语法，但表示的是 program 数量，不能把这两个对象混为一谈。

本例还约定输出单独分配，不覆盖输入；N 为正且很小。空张量、超大索引和任意非连续视图不在本章接口范围内。

**停下来确认：** x 和 y 各十个数，为什么输出不是二十个？因为我们做十次对应位置相加，每次只产生一个输出。

### 2. 为什么要把程序分成 host 和 kernel

先看职责，再读代码：

```text
CPU 上的 Python：准备 x/y → 调用 add → 分配 out → 确定启动多少份工作
                                               ↓ 提交 kernel
GPU 执行的程序：                               读取 x/y → 相加 → 写 out
```

CPU 上这部分常称 **host，主机端**。GPU 上完成计算的设备程序称 **kernel，内核**。GPU 不是在每个线程中启动一个 Python 解释器：Triton 会先把受支持的 kernel 源码编译成设备程序，再提交执行。

因此有两种阅读视角。读 `add` 时，我们跟踪主机端 Python 对象；读 `add_kernel` 时，我们推演它所表达的设备计算。编译器内部怎样把后者变成机器指令，留给后续章节。

初学时只要先抓住：**host 决定数据在哪里、准备哪些参数、启动多少个 program；kernel 定义每个 program 负责哪些元素，以及怎样读、算、写。**

### 3. 先把本章要读的代码完整放在一起

下面是上游第 21–78 行的**教学整理**：删除空行和解释性注释，整理函数形参排版，保留所有执行语句与原来的 `BLOCK_SIZE=1024`。这就是本章的代码主体，不需要你另去拼出函数。

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

前四行 import/设备选择、`add` 函数体由 host Python 执行。`add_kernel` 的函数体描述 GPU 计算；装饰器把它接到 Triton 的编译与启动接口。文件中写得靠前，不表示它会在定义时马上处理输入。

**唯一的纸笔配置改动：** 后面为了能逐个列出元素，把上游第 75 行的 `BLOCK_SIZE=1024` 换成 `BLOCK_SIZE=4`，记作 B=4。于是 N=10 分成三个 program。这个改动没有写回上游源码，也没有在本机运行 GPU；B=4 是教学配置，不是性能建议。

若保持上游 B=1024，N=10 只需一个 program，逻辑 tile 有 1024 个位置，其中 10 个有效；算法没有变，只是纸上列出它不方便。后文每张三 program 表都基于明确改成 B=4 的版本。

#### 3.1 三个 import 到底给出了什么名字

上游第 21、23、24 行：

```python
import torch
import triton
import triton.language as tl
```

`import torch` 让本文件可以用 `torch.empty_like`、`torch.Tensor` 等名字。它们服务于张量对象、分配与参考运算。

`import triton` 提供 `triton.jit` 和 `triton.cdiv` 等接口。这里的点表示从模块中取一个名字，并不是“把数据传到 GPU”的运算符。

`import triton.language as tl` 为 kernel 语言模块起别名 `tl`。所以 `tl.load` 实际指向该模块提供的 load 接口；`tl` 不是什么变量张量。

#### 3.2 DEVICE 是从哪里来的

上游第 26 行原样摘录：

```python
DEVICE = triton.runtime.driver.active.get_active_torch_device()
```

从右向左理解结果：调用 `get_active_torch_device()` 返回活动设备对应的 PyTorch device 对象，保存为本文件的 `DEVICE`。例如在一个 NVIDIA 环境中可以是某张 CUDA 设备；这里没有实际查询本机得到某个编号。

这个对象描述设备，不包含 x 的十个数，也不负责创建 x。后面的输入创建与 device 断言会使用它。驱动对象如何选出活动后端属于后续运行时内容，本章只需要它的这个输入/输出边界。

### 4. 进入 add：输入在哪里，函数怎样收到它们

为了把第一张表变成 GPU 输入，可以**设想**下面的调用代码。它是本章新增的输入构造示例，不是上游随机输入的原样摘录，也没有在本机执行：

```python
x = torch.arange(0, 10, dtype=torch.float32, device=DEVICE)
y = torch.arange(1, 11, dtype=torch.float32, device=DEVICE)
result = add(x, y)
```

第一行创建一个 PyTorch 张量对象，元素是 0 到 9，dtype 明确指定 FP32，数据存放在 DEVICE 对应设备。第二行同理创建 1 到 10。`torch.arange` 与刚才 Python 的 `range` 都是不含右端，但这里返回的是张量对象，不是 range 对象。

第三行开始调用我们自己定义的 `add`。括号里的两个对象分别交给函数形参 x、y。**此时没有把十个元素复制进十个 Python 参数；传入的是两个张量对象。**

看函数定义，上游第 62 行：

```python
def add(x: torch.Tensor, y: torch.Tensor):
```

`x`、`y` 是函数内的局部名字。冒号后面的 `torch.Tensor` 是类型注解，告诉读者和工具预期是什么类型；普通 Python 不会因写了这个注解就自动检查设备、长度和 dtype，更不会自动把 CPU 数据搬到 GPU。

现在这一步的输入明确了：`add` 持有两个 host 能操作的张量对象，它们描述的元素数据在 GPU。下一行才创建输出。

### 5. 第 64–66 行：准备输出和 N

上游第 64–66 行，连续原样摘录：

```python
output = torch.empty_like(x)
assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
n_elements = output.numel()
```

#### 5.1 第 64 行：empty_like 没有做加法

`torch.empty_like(x)` 返回一个新的张量对象，基本属性参考 x。在本例连续一维合同下，output 的形状也是 `(10,)`、dtype 也是 FP32、device 也相同，并拥有另外的存储。

`empty` 的意思是内容未初始化，不是“没有任何存储”，也不是“全部为零”。可以把此刻的设备存储画成：

```text
x 数据：       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y 数据：       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output 数据：  [?, ?, ?, ?, ?, ?, ?, ?, ?, ?]
```

这里 `?` 表示不能依赖其初始值，不是张量中真的存了问号。必须保证后面的 kernel 写到十个有效位置，否则可能把未初始化内容当结果返回。

`output` 这个局部变量保存的是张量对象；它不等于指向一个浮点数的裸地址。等启动 kernel 时，Triton 才会从对象取出所需设备参数。

#### 5.2 第 65 行：assert 只检查写出的条件

`x.device` 是 x 的设备属性，`==` 比较它是否等于 DEVICE。三个比较用 `and` 连接，只有全部成立才通过；不成立会触发断言失败。它不会替我们移动数据。

这一行只写了设备条件，没有写“x/y长度相同”“dtype一致”“一维连续”等条件。因此本章一开始才显式约定输入合同。未来把教程变成通用接口，需要自行设计输入校验；本章不把这份短 wrapper 冒充通用张量加法库。

例如 y 只有 8 个元素，断言仍可能通过，但我们稍后会尝试读取索引 8、9，不能指望只检查设备就安全。

#### 5.3 第 66 行：numel 返回元素个数

`output.numel()` 读取输出总共有多少个元素，返回主机端整数 10，赋给局部变量 `n_elements`。它不求和、不读取十个元素的数值，也不是字节数。

本例 `n_elements=10`；若算存储字节量才是 `10 × 4 = 40` 字节。后面 mask 使用的是元素索引，所以要跟 10 比较，不跟 40 比较。

**本步结果：** `add` 有了 x/y/output 三个张量对象和整数 N=10。下一步要决定启动多少个 program，才能把十个输出位置都写到。

**停下来确认：** 如果把 output 未初始化误解为全零，会出什么问题？漏写的位置可能看起来像某个合理结果；实际上程序并没有计算它，不能依赖其值。

### 6. 第 70 行：grid 怎样从一个函数变成启动数量

#### 6.1 先约定一个 program 负责 B 个位置

我们让每个 program 按相同规则处理 B=4 个逻辑位置。N=10，所以可以这样分配：

```text
program 0：索引 0、1、2、3
program 1：索引 4、5、6、7
program 2：索引 8、9，以及两个需要屏蔽的位置 10、11
```

三份工作使用同一段 kernel 程序，只是 program 编号不同。这里的“4”是**逻辑元素数**，不是“启动4个硬件线程”。怎样把元素分给GPU线程，后续布局章节再讲。

grid 描述要启动多少个 program。本例是一维 grid，因此需要包含整数3的元组 `(3,)`。不是 `(4,)`，也不是输入形状 `(10,)`。

#### 6.2 原语句先保存一个函数，并没有立即除法

上游第 70 行原样摘录：

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```

把它拆开读：

| 语法 | 在这行中的作用 |
|---|---|
| `grid =` | 把右边产生的函数对象保存到局部名字grid |
| `lambda meta:` | 创建一个只有一个参数meta的匿名函数 |
| `n_elements` | 使用外层add刚才得到的N，本例为10 |
| `meta['BLOCK_SIZE']` | 从传入的参数映射中读取本次B |
| `triton.cdiv(...)` | 向上取整，算需要多少个program |
| `(结果, )` | 返回一维grid元组，尾部逗号很重要 |

如果 lambda 不熟悉，在 add 内把这一行写成下面的**等价教学改写**就容易读了：

```python
def grid(meta):
    block_size = meta['BLOCK_SIZE']
    count = triton.cdiv(n_elements, block_size)
    return (count,)
```

调用这个函数时，形参 `meta` 才会接到实际对象。它不是预先存在的全局变量，也不是某个 GPU 张量。

函数可以访问定义它时所在的 add 中的 `n_elements`；这种保留外层名字的方式叫闭包。当前 add 调用里该名字一直是10，所以以后求grid时可以使用它。

#### 6.3 cdiv 的三和十是怎么算出来的

切到 [python/triton/__init__.py](../../triton/python/triton/__init__.py)，第 69–70 行，原样摘录函数定义和主体：

```python
def cdiv(x: int, y: int):
    return (x + (y - 1)) // y
```

原函数第68行还有 `@constexpr_function` 装饰器，这里只摘主体。在本例host调用中它根据整数计算结果；它也可服务编译时计算，具体机制不在本章展开。

注意这里的 x、y 是 cdiv 自己的形参，接到的是 `10` 和 `4`，**不是** add 中装着数据的两个张量。不同函数可以使用相同名字，含义由实际传入参数决定。

代入过程：

```text
cdiv 的 x = n_elements = 10
cdiv 的 y = BLOCK_SIZE = 4
(10 + (4 - 1)) // 4 = 13 // 4 = 3
```

`//` 是整数整除。加上 `y-1` 后再整除，使正长度遇到尾部不足一块时仍然多分配一块。这里10需要两整块加一尾块；若误写成 `10//4=2`，最后两个有效元素永远不会被处理。

函数返回整数3给grid，grid再返回 `(3,)` 给Triton调用方。这是两个函数返回值的连续传递，不是GPU做了十次除法。

#### 6.4 meta 是谁传进来的

看一眼后面第75行的教学配置调用：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
```

这次调用将 `BLOCK_SIZE=4` 交给Triton。Triton整理参数后，在正常发射前求grid；在 [runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第790行能看到实际调用语句，位于“grid可调用”的分支中：

```python
grid = grid(bound_args)
```

右边的grid仍然是我们定义的函数，`bound_args` 是整理好的调用参数映射。这个映射作为实参进入grid函数，所以函数内形参meta指向它；其中可以查到 `meta['BLOCK_SIZE'] == 4`。不能把meta写成“只有BLOCK_SIZE的字典”，实际还包含其他绑定参数。

函数返回 `(3,)` 后，左边的grid重新绑定为这个元组。这里“同一个变量名先保存函数、再保存元组”是运行时内部的写法，不是输入数据发生变化。JIT可能在求这个启动grid之前准备编译产物；本章只追清这个参数边界，具体顺序由4.2带读。

**本步结果：** 我们知道了本次提交的grid将是 `(3,)`，对应program id 0、1、2。还没有因此把输出加好。

**停下来确认：** B=4、N=8时grid是什么？`(2,)`。不能固定“有尾块”就永远多加一块，整除时不需要额外块。

### 7. 第 75 行与第 29–36 行：实参怎样到达 kernel

上游调用的**教学配置版**和kernel签名放在一起。调用只把原来的1024改成4；签名仅调整排版并移除注释：

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
```

第二段只展示签名，不是独立可运行的完整函数；完整函数体已经放在第3节。

#### 7.1 装饰器为什么出现在 def 上面

普通Python里，`@某个装饰器` 会让定义好的函数经过装饰器处理，之后名字指向处理后的对象。这里 `@triton.jit` 把函数接到Triton接口，支持后续的 `add_kernel[grid](...)` 形式。

它没有在定义时得到x/y，没有定义时便计算十个数，也不是让GPU逐行执行Python解释器。你可以先把它理解为“这段函数体将由Triton用于生成kernel，并通过特殊接口启动”。生成过程在4.2/5.1再拆开，不需要现在背下AST和IR。

#### 7.2 方括号和圆括号是两步

`add_kernel[grid]` 先把启动网格交给Triton的接口，得到可调用对象；后面的圆括号再传入这一次计算的实参。这里方括号不是从一个数值列表里取元素，也不是取输出的第grid个位置。

为便于观察，可以理解成下面的**两步教学改写**：

```python
launch_add = add_kernel[grid]
launch_add(x, y, output, n_elements, BLOCK_SIZE=4)
```

第一行选择启动方式，第二行才交出数据参数。接口内部怎样转到 `run`，正是4.2的起点。本章保留这一边界，不再把整个编译器压到这里。

#### 7.3 每个参数的接收关系

| add调用处提供什么 | kernel中叫什么 | 在本例中表示什么 |
|---|---|---|
| 第1个位置参数，host张量x | `x_ptr` | 指向x第一个元素的设备指针，元素类型为FP32 |
| 第2个位置参数，host张量y | `y_ptr` | 指向y第一个元素的设备指针 |
| 第3个位置参数，host张量output | `output_ptr` | 指向host已经分配好的输出第一个元素 |
| 第4个位置参数，整数10 | `n_elements` | 有效元素数N，kernel执行时用它比较索引 |
| 关键字参数`BLOCK_SIZE=4` | `BLOCK_SIZE: tl.constexpr` | 编译时已知的逻辑tile长度B |

前四个按位置对应，最后一个按关键字名字对应。`x_ptr` 是我们给形参取的名字；并不是只要变量名带ptr，普通Python就会自动生成指针。是Triton的张量参数处理，让设备程序得到地址与相应类型信息。

`tl.constexpr` 是Triton认识的特殊注解：该参数用于编译时决策，例如下面arange的形状。若以后改B，可能需要另一个编译版本；`n_elements` 则保留为运行时输入。不要把两者都理解成每个GPU线程从内存里读到的普通Python整数。

**本步结果：** kernel需要的三个基地址、N和B都能追溯到调用者。接下来假设设备程序已经编译并开始执行，先跟踪program 1的计算语义。

### 8. 第 39、44 行：program 1 怎样选中第二组位置

上游第39行和第44行，非连续摘录，中间仅有注释：

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
```

#### 8.1 第 39 行：pid 的来源

grid为 `(3,)`，所以有三个program实例执行同一段程序，它们在第0维的编号分别为0、1、2。`axis=0` 选择grid的第0维；不是“取输入x的第0个元素”。本章一维grid只需要这个轴。

`tl.program_id` 返回当前实例的编号，赋给pid。在我们这次跟踪的program 1里，pid就是1。其他program拥有各自的pid与局部逻辑值，没有共享一个会被彼此覆盖的Python变量。

#### 8.2 第 44 行：从编号到起点

`block_start = pid * BLOCK_SIZE` 代入 `1*4=4`。得到的4是**元素索引起点**，不是地址、不是数值x[4]、不是字节数。

```text
program 0：block_start = 0 * 4 = 0
program 1：block_start = 1 * 4 = 4  ← 当前跟踪
program 2：block_start = 2 * 4 = 8
```

每个program把自己的编号乘上B，便选中不同的一段数据。数据本身此时还没加载，计算起点不需要读取x/y。

**停下来确认：** program 2的起点为什么不是2？因为pid是块的编号，而我们要找它负责的第一个元素；每块隔B个元素。

### 9. 第 45 行：从一个起点生成四个整数索引

上游第45行原样摘录：

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

先算右边的 `tl.arange(0,4)`。它描述一个长度为4的逻辑整数张量，值为 `[0,1,2,3]`，不含右端4。它不是host创建的torch张量，也不是Python list；我们只是用方括号把逻辑元素画出来。

然后把标量起点4加到每个位置：

```text
block_start                         4
tl.arange(0, BLOCK_SIZE)             [0, 1, 2, 3]
逐元素相加                          [4, 5, 6, 7]
offsets                             [4, 5, 6, 7]
```

把标量自动用于每个元素，叫广播。这里没有先在显存中复制四份4，也不需要你手写四次加法；源码表达的是一组元素的相加语义。

这一步的结果offsets是形状 `(4,)` 的**整数索引张量**。它不是四个浮点输入值，也不是四个指针；只有后面的 `x_ptr + offsets` 才构造指针张量。上游第43行的“list of pointers”注释容易让人混淆，按实际表达式类型区分即可。

为什么不直接写 `range(N)`？每个program只负责自己的B个位置，而不是让每个program重复整份输入。B也需要编译时已知，才能用于本例的逻辑张量形状；取B为4这个二次幂符合本例arange的范围要求，N可以是10。

**本步结果：** 当前program知道要处理索引4、5、6、7，尚未读取这四个位置的输入值。

### 10. 第 47 行：mask 是四个布尔条件

上游第47行原样摘录：

```python
mask = offsets < n_elements
```

左边是四个索引，右边是有效元素数10。这是逐元素比较：

```text
offsets       [4,    5,    6,    7   ]
小于10？      [True, True, True, True]
mask          [True, True, True, True]
```

结果mask是形状 `(4,)` 的布尔逻辑张量，不是单个Python的True/False，也不是一个笼统地“整个块有效”的判断。

当前四个位置都有效；它真正重要的作用会在program 2体现，因为那时offsets是 `[8,9,10,11]`，mask会变成 `[True,True,False,False]`。

**mask这一行本身只得到条件。** 它不会自动保护后面所有代码；必须把它交给实际的load和store。下一步会看到具体交给哪里。

### 11. 第 50–51 行：从索引到地址，再从地址到输入值

上游第50–51行连续原样摘录：

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
```

#### 11.1 先拆开 x_ptr + offsets

`x_ptr` 来自host传入的第一个张量，指向x的第一个元素。offsets是刚才算出的 `[4,5,6,7]`。类型化指针加整数按**元素**前进：FP32每个元素4字节，所以 `x_ptr+4` 前进16字节。

为了手算，假设x首元素的字节地址记作1000。**这是解释用的假地址，不是真实指针，也不要把它用来访问内存。**

```text
元素索引                           4     5     6     7
相对首地址的字节偏移               16    20    24    28
假设x_ptr对应字节地址1000
x_ptr + offsets对应的字节地址      1016  1020  1024  1028
这些地址上原有的x元素值              4     5     6     7
```

这里不能再把offsets乘以4后写进 `tl.load(x_ptr+...)`，因为指针加法已按其元素类型处理缩放。若写 `x_ptr + offsets*4`，就会跳到第16、20等元素，改变含义甚至越界。

`x_ptr + offsets` 产生四个逻辑指针，**只计算要去哪里，还没有得到那里存的浮点数**。这就是地址计算与加载的区别。

#### 11.2 再读 tl.load 的两个参数

第一个参数是那组指针，第二个 `mask=mask` 的左边mask是API形参名字，右边mask是当前kernel的布尔张量变量。含义是“每个逻辑位置按对应条件决定是否读对应地址”。

program 1的mask全True，所以四个位置都会读取，返回形状 `(4,)` 的FP32逻辑张量，赋给局部名字x：

```text
x_ptr + offsets    指向原始输入x的第4、5、6、7个元素
tl.load(...)       得到这四个位置的值
kernel中的x        [4.0, 5.0, 6.0, 7.0]
```

同理，y_ptr指向第二个输入的起点，而第二个输入是1到10，所以第4、5、6、7个元素分别为5、6、7、8：

```text
kernel中的y        [5.0, 6.0, 7.0, 8.0]
```

这两次load访问的是设备全局地址，实际取数可以经过缓存；不能因教程注释用了DRAM一词就断言每次都直接到显存芯片。`tl.load` 也不等于显式声明一块共享内存。

#### 11.3 为什么两处都叫 x，却不是同一个对象

| 名字出现在哪里 | 是什么 | 本例包含/描述什么 |
|---|---|---|
| `add`中的x | host的`torch.Tensor`对象 | GPU上完整的十个输入元素及其元信息 |
| `add_kernel`形参x_ptr | 设备程序中的类型化基指针 | 完整输入第一个元素的位置 |
| `add_kernel`中load结果x | 当前program的逻辑张量值 | 当前这四个输入数，program 1里为4到7 |

它们之间有联系，但没有变成同一个Python变量。函数作用域不同，x这个简短名字被复用了。kernel里的 `x = tl.load(...)` 不会把host张量变量x替换成四个元素，也不修改原始输入。

**本步结果：** program 1已经有了四个x值、四个y值。现在才能执行数值加法。

**停下来确认：** `x_ptr+5` 是不是浮点数5？不是。它是指向第5个元素的地址表达式；用load才得到那里存的数。本例恰好x[5]=5，容易掩盖两者的区别。

### 12. 第 52、54 行：相加不会自动写进输出

上游第52行与54行，非连续摘录，第53行只有注释：

```python
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

#### 12.1 第 52 行：先产生一个逻辑计算结果

当前两边形状都是 `(4,)`，所以逐元素相加：

```text
kernel里的x       [4.0, 5.0,  6.0,  7.0]
kernel里的y       [5.0, 6.0,  7.0,  8.0]
kernel里的output  [9.0, 11.0, 13.0, 15.0]
```

这只是把结果赋给kernel局部名字output，尚未把它写入host预分配的输出存储。与x一样，**kernel里的output逻辑值和host里的output张量对象不是同一个对象**。

可以用 `output_ptr` 判断该往哪里写，但光有一个叫output的变量，编译器不会凭这个名字猜出你要保存到哪个地址。

#### 12.2 第 54 行：用输出基地址和相同索引写回

把store三个参数分开：

| 参数表达式 | 从哪里来 | 当前意义 |
|---|---|---|
| `output_ptr + offsets` | host分配给出的基地址，加第45行的索引 | 输出的第4、5、6、7个元素位置 |
| `output` | 第52行相加结果 | 要保存的9、11、13、15 |
| `mask=mask` | 第47行比较结果 | 当前四个位置都允许写 |

例如输出首地址假设为3000，那么四个写入地址是3016、3020、3024、3028。它们与x的地址不同，即使索引同为4到7。**offsets相同只表示逻辑位置一致，基指针决定是哪一份张量。**

写完后，可以把program 1的贡献单独画出来：

```text
输出索引      0  1  2  3    4   5   6   7    8  9
本program写入 -  -  -  -    9  11  13  15    -  -
```

横线只表示这个program不写这些位置，不表示最终输出里存了横线，也不表示另外两个program此刻一定没执行。实际执行顺序可以与纸笔顺序不同。

kernel不靠Python的 `return output` 把这四个数交还host；它靠store写入调用者提供的输出存储。host最后返回的是描述同一份存储的张量对象，这个联系会在第14节闭合。

### 13. 换到 program 2：两个不存在的位置怎样处理

#### 13.1 不是把输入实际扩容到十二个数

保持同一段kernel，只将当前pid改成2。完整推演：

| 语句对应结果 | 第0个逻辑位置 | 第1个 | 第2个 | 第3个 |
|---|---|---|---|---|
| pid | 2 | 2 | 2 | 2 |
| block_start | 8 | 8 | 8 | 8 |
| arange | 0 | 1 | 2 | 3 |
| offsets | 8 | 9 | 10 | 11 |
| mask，是否小于10 | True | True | False | False |
| load得到的x | 8 | 9 | 未定义 | 未定义 |
| load得到的y | 9 | 10 | 未定义 | 未定义 |
| 加法的有效结果 | 17 | 19 | 不依赖其值 | 不依赖其值 |
| store行为 | 写out[8] | 写out[9] | 不写 | 不写 |

前两行把标量按列重复，只为方便对照；pid和block_start本身仍是标量，不因此变成四元素张量。

逻辑tile保持四个位置，不意味着输入真的有十二个元素。mask传给load后，索引10、11的对应地址不读取；传给store后，它们也不写入。不能去访问 `x[10]` 来“看看会得到什么”。

#### 13.2 未定义不是默认零

上游第50–51行没有指定 `other`。在 [core.py](../../triton/python/triton/language/core.py) 第2451–2453行的load参数文档中，明确写了两点：mask为False时不加载对应地址；`other=None` 时，该位置结果未定义。

因此上表用“未定义”，不能填0。本例只做逐元素加法，无效位置不影响前两个有效位置，最终又被同一个store mask丢弃，所以有效输出不依赖无效位置的数值。

如果为了教学观察希望补零，可以把两行改成下面的**明确改写**：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0)
y = tl.load(y_ptr + offsets, mask=mask, other=0)
```

这时program 2的逻辑加载结果变成 `[8,9,0,0]` 与 `[9,10,0,0]`，加法结果为 `[17,19,0,0]`，仍然只写出前两个位置。`other=0` 给的是被屏蔽位置的替代**值**，不会将越界输入存储补出两个零。

之后学习softmax和求和归约时，无效位置可能参与多个值的合并，补齐值必须按算法选择。本章不把“所有算法都补0”当规则。

#### 13.3 两个常见错误为什么错

**只保护store：** 如果load没有mask，程序已经表达了越界读取，即使最后不写这两个结果，也不能让前面的读取合法。

**先load，再选值：** `tl.where(condition, tl.load(ptrs), 0)` 不是Python短路if，不能用后续结果选择保证其中的load不发生。要保护读取，就把mask交给load本身。

这两点都来自同一个顺序：条件必须影响真正的内存操作，不能只影响最后使用哪个结果。

### 14. 三个 program 的结果怎样合成十个输出

现在把三个program都完成后的写入位置合起来。这是最终状态推演，不要求程序按0、1、2的顺序串行执行。

| program id | offsets | 有效输入x | 有效输入y | 写回位置与数值 |
|---|---|---|---|---|
| 0 | 0、1、2、3 | 0、1、2、3 | 1、2、3、4 | out[0:4] = 1、3、5、7 |
| 1 | 4、5、6、7 | 4、5、6、7 | 5、6、7、8 | out[4:8] = 9、11、13、15 |
| 2 | 8、9、10、11 | 8、9 | 9、10 | out[8:10] = 17、19 |

这里的切片 `[a:b]` 表示从a到b−1。合并后：

```text
out = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```

为什么不会漏写？每个有效索引i都能拆成 `pid=i//4` 与 `j=i%4`，于是 `i=pid*4+j`，且j在0到3。索引9对应pid=2、j=1；启动了program 2，它的这个位置也通过mask。

为什么不会重复写？不同pid选的是互不重叠的区间，同一个i不可能有两种这样的商和余数。输出又是独立分配，三个program也不会通过输出地址覆盖输入。

因此本例不需要program间一起累加同一位置，也不需要原子加法或跨program等待。**这来自本例地址分工，不是所有Triton kernel都天然没有写冲突。**

#### 14.1 host 第 78 行返回了什么

上游第78行原样摘录：

```python
return output
```

这行回到CPU上的add函数；其中output仍是第64行分配的完整张量对象，描述十个输出的存储。最外层 `result = add(x,y)` 接到它，所以result描述的就是kernel store写入的那份输出。

把两个同名output再放在一起：

```text
add的output：       分配完整输出 → 把其地址传为output_ptr → 返回给result
                                      ↑
kernel的output：    四个逻辑相加结果 → store写入output_ptr + offsets
```

上面箭头指存储关联，不是kernel直接返回一个Python对象。

#### 14.2 返回不等于GPU已完成

host通常向设备stream提交工作后继续执行，因此走到 `return output` 时，GPU可能仍在计算，也可能已经完成。不能据此作完成性假设。第14节列出的整数组是**工作完成后的期望状态**。

stream可以先理解为设备工作的有序队列。同一stream里的后续依赖可以按先后关系消费输出；跨stream使用要另建正确依赖。CPU若要读取设备结果，需要相应的同步/传输机制；不能仅看Python调用已经返回。

这也解释了为什么 `t0=CPU时钟; add(...); t1=CPU时钟` 不能直接当成纯GPU计算时间：它可能包含编译、提交和分配，却没有等待设备完成。测量工具与同步口径放在后续章节。

**停下来确认：** out有十个有效位置，kernel里的output只有四个位置，矛盾吗？不矛盾。前者是完整输出存储，后者是一个program的一块逻辑计算结果，三份program分段写入同一完整输出。

### 15. 现在再回读完整代码，会看到哪些连接

这张表只复查已经逐段讲过的对象，不引入新的隐藏实现：

| 对象 | 由谁创建 | 谁接着使用 |
|---|---|---|
| x/y的host张量对象 | 输入构造示例 | add → Triton调用参数处理 |
| host output | `torch.empty_like(x)` | 作为kernel的output_ptr来源，最后返回给result |
| N=10 | `output.numel()` | grid闭包和kernel的n_elements |
| B=4 | 本章修改后的调用关键字参数 | grid参数映射和kernel编译时常量 |
| grid函数 | lambda表达式 | JIT在发射前调用，得到 `(3,)` |
| pid | 当前program的编号操作 | 计算当前块起点 |
| offsets | 起点 + arange | 同时为x/y/out构造对应位置的地址 |
| mask | offsets与N比较 | 同时保护两次load和一次store |
| kernel x/y | masked load | 逐元素加法 |
| kernel output | 加法 | store写入完整输出 |

本章实际跨了三处源码：教程定义算子、顶层工具实现cdiv、JIT里一行回调解释grid由谁求值。后者只确认调用边界，还没展开JIT内部对象与编译过程。下一部分（4.2）从第75行调用的这组实参继续，不要求你现在自己去补完整个jit.py。

### 16. 回到原论文：现在读哪些位置会更有感觉

按 [论文定位表](../appendices/reference.md#sec-a-1) 打开MAPL 2019原文。下面只对应问题和思想，不把现代Python接口说成原论文语法。

| 本章已经理解的代码 | 论文准确位置 | 回读时想的问题 |
|---|---|---|
| arange形成一组逻辑索引，B编译时已知 | §3.1 Syntax，PDF第3页 | 当时怎样声明tile和整数range？现代constexpr与旧参数化语法有什么区别？ |
| 一个program算四个位置，不写逐线程索引 | §3.2.1 Tile semantics，第3页；§3.3/Figure 3，第4页 | 源程序表达整块工作后，哪些线程分工交给编译器？ |
| 标量起点加到四个位置 | §3.2.2，PDF第3–4页，Listing 4在第4页 | 广播为何让块级表达更简洁？ |
| mask决定每个逻辑位置是否访问 | §3.1 Predication，第3页；§4.3/Listing 6，第5页 | 为什么tile内部需要逐元素条件？为什么单个标量if不够？ |

论文原先使用Triton-C与旧Triton-IR。现代masked load不等于论文的psi指令；我们固定的2020历史快照也还未核实完整PSSA/prefetch对应。可沿[论文 IR 篇](../paper-reading/02-ir.md)和[编译篇](../paper-reading/03-compiler.md)对照真实代码。

### 17. 不装 Triton，先用 CPU 检查这套索引推演

本章的 [trace_vector_add.py](../tools/trace_vector_add.py) 只使用Python标准库，输入与本章一致：x为0到9，y为1到10，N=10、B=4。采用显式补零来观察无效位置，相当于第13.2节的教学改写；不是原教程未定义位置的数值预测。

在两个仓库和手册的共同父目录执行：

```bash
python3 triton-study-guide/tools/trace_vector_add.py
```

这是一段CPU列表程序，枚举program和逻辑位置，核对每个有效索引恰好写一次、有效加法结果一致。它不导入torch/triton、不启动GPU，也不模拟线程/寄存器布局或编译器。

其中最关键的短段是下面两行。它们是辅助脚本的CPU写法，用来表达“先判断有效性，再读取Python列表”：

```python
loaded_x = [x[i] if valid else 0 for i, valid in zip(offsets, mask)]
loaded_y = [y[i] if valid else 0 for i, valid in zip(offsets, mask)]
```

`zip` 把相同位置的索引和布尔条件配成对；例如尾块得到 `(8,True)`、`(9,True)`、`(10,False)`、`(11,False)`。Python条件表达式只有valid为真才取 `x[i]`，所以不会去索引不存在的x[10]。

这和第13.3节提到的Triton `tl.where` 语义不同。这里使用的是Python条件表达式，不能把它当作两种语言任何情况下都一样执行的证据；CPU脚本只核对本章的整数与数学关系。

### 18. 自己走一遍：自测与参考答案

可以先盖住右侧答案，画表再核对。回答时尽量说“这个变量由哪行得到”，而不只背定义。

| 问题 | 参考答案 |
|---|---|
| N=6、B=4，要几个program？尾块offsets与mask是什么？ | 2个；尾块 `[4,5,6,7]`，mask为 `[True,True,False,False]` |
| 对同样x[i]=i、y[i]=i+1，这个N=6尾块写出什么？ | 只写out[4]=9、out[5]=11 |
| 第70行执行后，grid立即是整数3吗？ | 不是，它先是函数；以绑定参数求值后返回元组 `(3,)` |
| cdiv里的形参x接到十个浮点数吗？ | 不会，接到N=10这个主机整数；同名不代表同一对象 |
| x的FP32基指针加3，前进几字节？ | 12字节，不再手乘4写入类型化指针偏移 |
| offsets `[4,5,6,7]` 本身是四个输入数吗？ | 不是，是四个整数索引；加基指针后是地址，load后才是输入数 |
| 原教程尾块被mask掉的load值一定为0吗？ | 不一定，未指定other时未定义；显式other=0是改写 |
| 只有mask比较，没有给load传mask，安全了吗？ | 没有，条件必须用于内存操作 |
| B=1024是否要求1024个硬件线程？ | 否，B是逻辑位置数，不能单凭B推断线程数和layout |
| kernel里的output为什么没有return也能得到结果？ | store写入host事先分配且通过output_ptr传入的存储 |
| Python函数返回时，CPU能直接当普通内存读取GPU地址吗？ | 不能；对象返回、设备工作完成和跨设备读取是不同事情 |

#### 本章交出的边界

你现在应该可以把 `add(x,y)` 到十个有效输出的每一步说清，并手算另一组N/B。还没展开的是：方括号接口怎样进入JIT、如何整理参数、何时编译、怎样加载并发射设备程序。**[4.2将接着同一次调用的x/y/output/N/B继续讲](#sec-4-2)。** 不要求你先自行补读完整运行时源码。

4.1 的验证范围：固定快照源码静态核对，CPU 索引与教学数值检查；这部分没有实际 GPU 执行或性能结论。4.3 节给出同一 kernel 在 A800 上的一次实际运行。

**第一部分交出的边界。** 4.1 已经把计算语义闭合：三个 program 覆盖十个输出，mask 保护尾块，store 写回 host 分配的同一份存储。现在把镜头拉回 host，从 `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)` 这一行开始，看方括号返回了什么、参数怎样绑定、缓存何时命中、编译产物怎样变成设备发射。

<a id="sec-4-2"></a>
## 4.2 方括号后面发生了什么：host 到 JIT 与启动

本章继续追同一次调用：x 为 0 到 9，y 为 1 到 10，output 是 host 新分配的十元素 FP32 GPU 张量，N=10，教学配置 B=4。我们要解释这些对象怎样被按名字绑定、哪些信息用于编译、哪些对象留到启动时传下去。以下先做固定提交 `46eccb199bfd3f3559b512c4a15870bd45e6a74d` 的静态带读与纸笔推演；4.3 节给出从该提交构建后的实际运行。摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. 同一行里有两次不同的调用

[01-vector-add.py](../../triton/python/tutorials/01-vector-add.py) 第70–75行定义 grid 并调用 kernel。下面是教学配置改写，只有 B 从上游1024改为4：

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
```

不要把整个表达式当成 Python 内置 GPU 语法。先求 `add_kernel[grid]`，得到一个可调用对象；再用后面的圆括号调用它。两步均在 host Python 中发生。grid 现在仍是函数，不是已经算好的 `(3,)`。

为什么 `add_kernel` 支持方括号？在通常的 JIT 模式下，`@triton.jit` 把原函数包装成 `JITFunction` 对象。装饰器返回对象的真实分支见 [runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第1025–1044行，其中第1031–1033行：

```python
else:
    return JITFunction(
        fn,
```

这里只摘构造调用开头，后续版本、debug 等参数省略。`fn` 是刚定义的普通 Python 函数；包装后名字 `add_kernel` 指向的对象保存了它的源码和参数信息。若显式开启解释器模式会走另一分支，本章限定正常 JIT 路径。

### 2. __getitem__ 返回一个记住 grid 的函数

[runtime/jit.py](../../triton/python/triton/runtime/jit.py) 的 `KernelInterface.__getitem__`，第371–377行，删除文档字符串：

```python
def __getitem__(self, grid) -> T:
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

`self` 是“这次方法所属的对象”，此处就是 `add_kernel` 对应的 JITFunction；不是 x，也不是某个 program。Python 会把 `add_kernel[grid]` 转到它的 `__getitem__` 方法。继承让 JITFunction 可以使用 KernelInterface 提供的接口。

先写成 `launch = add_kernel[grid]` 和 `launch(x, y, output, 10, BLOCK_SIZE=4)`。第一行创建函数，第二行才执行函数体。函数从外层的 `__getitem__` 取到 `self` 和 `grid`，从第二行得到 `args=(x, y, output, 10)` 与 `kwargs={"BLOCK_SIZE": 4}`。后者是一张按名字查值的字典，键 `"BLOCK_SIZE"` 对应值 `4`。

把星号展开读，这次调用等价于 `self.run(x, y, output, 10, grid=grid, warmup=False, BLOCK_SIZE=4)`。这里的 `grid` 还是可在稍后调用的函数。需要从普通 Python 调用开始拆解时，回看[形参、字典、星号与闭包](00-foundations.md#python-launch-syntax)。

**确认题：** 方括号执行后有没有读取 x[0]？没有；得到的是 host 上的代理函数。现在连 grid 函数都还没有求值。

### 3. 参数表和 binder 是在哪里来的

`JITFunction.__init__` 在 [runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第817–840行建立参数记录。下面是不连续摘录，第817–823、828–829、839–840行：

```python
self.params = []
for i, param in enumerate(self.signature.parameters.values()):
    if param.kind == inspect.Parameter.VAR_KEYWORD:
        raise TypeError(f"JIT functions do not support **{param.name} parameters")
    dns = i in do_not_specialize or param.name in do_not_specialize
    dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
    self.params.append(KernelParam(i, param, dns, dns_oa))

self.device_caches = defaultdict(self.create_binder)

self.arg_names = [p.name for p in self.params]
self.constexprs = [p.num for p in self.params if p.is_constexpr]
```

`signature` 描述函数的形参声明，顺序来自 `def add_kernel(...)`；它不是输入张量的 shape。enumerate 同时给出编号 i 与参数对象 param。KernelParam 保存名字、注解和是否允许特化等元信息。

本例 `arg_names` 是五个名字：`x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE`。最后一个有 `tl.constexpr` 注解；KernelParam 的 `is_constexpr` 在同文件第337–339行根据注解识别，因此其编号4出现在 constexprs 列表。

这里的 constexprs 列表是**参数位置记录**，后面 `_pack_args` 生成的同名局部字典才是“位置到实际常量值”的映射。相同单词不代表同一个对象。

`defaultdict(self.create_binder)` 是有默认创建函数的字典：首次用某个 device 查找而尚无条目时，调用 create_binder 生成该设备的缓存和辅助对象。不是每次启动都重新创建全部对象。

第707–718行的 create_binder，去掉文档字符串：

```python
def create_binder(self):
    from ..compiler import CompiledKernel, compile, ASTSource, make_backend
    target = driver.active.get_current_target()
    backend = make_backend(target)
    self.CompiledKernel = CompiledKernel
    self.compile = compile
    self.ASTSource = ASTSource
    binder = create_function_from_signature(self.signature, self.params, backend)
    return {}, {}, target, backend, binder
```

import 把 compiler 包中的函数/类拿到当前函数内；随后保存为 self 成员，所以 `_do_compile` 里能写 `self.compile` 和 `self.ASTSource`。这并不是 JITFunction 自己实现了另一个编译器。

target 描述当前目标设备类别及架构；backend 是为这个目标选出的编译后端对象。两个空字典分别供编译内核缓存与键计算辅助缓存使用，后面 run 会按同一顺序接收返回的五项。binder 是 host 函数，专门把位置/命名参数整理成形参名字。

### 4. binder 实际返回什么

`create_function_from_signature` 按 kernel 的签名生成 Python 函数。第451–456行构造函数文本，第470–474行用 exec 创建并返回函数，均在 [runtime/jit.py](../../triton/python/triton/runtime/jit.py)。关键模板原样摘录：

```python
func_body = f"""
def dynamic_func({", ".join(arg_list + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""
```

f字符串把参数名等插入文本；双花括号让最终文本里出现普通字典花括号。这里使用 exec 的目的，是让 Python 自己按生成出的函数签名处理位置、命名参数与默认值。它不是执行 GPU kernel 源码。

本例 binder 生成的函数可以**概念性改写**为：

```python
def bind_example(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE, **options):
    params = dict(x_ptr=x_ptr, y_ptr=y_ptr, output_ptr=output_ptr,
                  n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return params, specialization_descriptions, options
```

`specialization_descriptions` 只是本段的说明性名字，真实生成语句调用后端特化实现，见第420–436行；这段不能独立运行。

于是四个位置参数加 `BLOCK_SIZE=4` 被绑定为：

| bound_args 的键 | 值的来源与对象 |
|---|---|
| x_ptr | host 的 x 张量对象 |
| y_ptr | host 的 y 张量对象 |
| output_ptr | empty_like 返回的 output 张量对象 |
| n_elements | output.numel() 返回的整数10 |
| BLOCK_SIZE | 调用处显式给出的整数4 |

参数名字叫 x_ptr，不表示 binder 已把十个值变成 Python 列表。它还保存着传入对象。类型、对齐或整数值等特化描述另外放入 specialization；剩余编译选项放入 options。不同目标与实参可能产生不同描述，本章不捏造实际内存地址或对齐值。

### 5. run 先问缓存，再决定是否进入编译

[runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第750–751、757–760、768–778行，不连续摘录：

```python
device = driver.active.get_current_device()
stream = driver.active.get_current_stream(device)

kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
bound_args, specialization, options = binder(*args, **kwargs)

key = compute_cache_key(kernel_key_cache, specialization, options)
kernel = kernel_cache.get(key, None)
if kernel is None:
    options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                          options)
    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
    if kernel is None:
        return None
```

device 用来选择该设备的缓存；stream 描述本次设备工作提交到哪条执行流。它们都不是 kernel 在 GPU 上计算出的数。

第一行解包与 create_binder 的五项返回一一对应。binder 刚才返回的 params 在这里改叫 bound_args；传递对象时可以改局部变量名，内容仍是那个映射。

key 由特化描述和选项计算，`get(key, None)` 表示查找已编译对象，找不到就返回 None。缓存命中时无需再次走下面的编译分支；更换 B 这类 constexpr 会改变编译条件。不能因此反推“每个不同 N 必定重新编译”或“只看 shape”，实际特化规则还涉及参数类别和值/对齐等。

此处是进程内、按设备组织的内核缓存。进入 `compile` 后还有磁盘缓存检查，所以“这里未命中”不等于“必定重新执行所有编译 pass”。

### 6. _pack_args 把编译所需的四类东西分开

同文件第722–726、735–742行，不连续摘录：

```python
options = backend.parse_options(kwargs)
sigkeys = [x.name for x in self.params]
sigvals = [x[0] for x in specialization]
signature = {k: v for (k, v) in zip(sigkeys, sigvals)}

constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
attrvals = ['' if x[0] == 'constexpr' else x[1] for x in specialization]
attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
return options, signature, constexprs, attrs
```

`x[0]` 取每条特化描述的类型字段；zip 把参数名字与类型按顺序配对。FP32 张量通常对应指向 FP32 的参数类型；这里记录的是类型字符串，而非十个元素。constexpr 参数在 signature 中有专门标记。

find_paths_if 支持嵌套参数，所以位置用 tuple 表示。本例简单参数的 B 路径为 `(4,)`，对应值4；即局部 constexprs 中有 `{(4,): 4}`。这个4会参与编译，可以决定 arange 的长度。

attrs 保存后端可理解的参数性质，例如已推导出的可整除性；它不是 tensor 内容。options 则是 num_warps 等编译配置的解析结果。未提供这些选项时使用后端默认值，不能从 B=4 推断物理线程数为4。

**确认题：** B 与 N 都是 host 整数，为什么不完全一样？因为 B 的形参带 constexpr 注解，会作为编译时常量使用；N 的普通形参仍可作为设备函数的运行时标量参数，另有特化规则可能附加性质。

### 7. _do_compile 包装的是源码任务，不是输入数据

同文件第910–916、939–944行，选取通常的同步路径，省略钩子及异步编译分支：

```python
def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    kernel_cache, _, target, backend, _ = self.device_caches[device]
    src = self.ASTSource(self, signature, constexprs, attrs)
    kernel = self.compile(src, target=target, options=options.__dict__)
    kernel_cache[key] = kernel
    return kernel
```

这是**分支整理摘录**，插入注释明确标出了省略处。`self.ASTSource` 来自 create_binder 中的赋值；第一个实参 self 是 JITFunction，而不是普通输入张量。因此 ASTSource 可以找到原 kernel 函数、源码、形参表。

切到 [compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) 第54–66行，主体摘录：

```python
self.fn = fn
self.language = Language.TRITON
self.ext = "ttir"
self.name = fn.__name__
self.signature = signature
self.constants = dict()
if constexprs is not None:
    for k, v in constexprs.items():
        k = (fn.arg_names.index(k), ) if isinstance(k, str) else k
        assert isinstance(k, tuple)
        self.constants[k] = v
self.attrs = attrs or dict()
```

此处 self 已换成新建的 ASTSource 对象；形参 fn 接到刚才的 JITFunction。成员 constants 把参数位置映射到编译时值。本例保留 `(4,) → 4`。ASTSource 是一个“待编译来源”的包装，不是已经生成的 AST 树，也不是 GPU 二进制。

compile 返回的 kernel 是 CompiledKernel 这一类产物对象，随后写入缓存。它如何调用 make_ir 生成最初的 IR，由 [5.1](05-frontend-ir.md#sec-5-1) 接着带读。

### 8. 回到 run：grid 终于得到参数

缓存命中和新编译完成最终汇合到同一个启动段。回到 [runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第787–799行摘录，删除注释、保留执行语句：

```python
if not warmup:
    if callable(grid):
        grid = grid(bound_args)
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
               knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
return kernel
```

代理给的 warmup=False，所以进入此段。callable 判断 grid 是否可调用；我们的 lambda 是函数，因此 `grid(bound_args)` 把上表映射作为 meta 传入。读取 `meta['BLOCK_SIZE']=4`，闭包里的 N=10，算出 `(3,)`。

补齐三维后，grid_0=3、grid_1=1、grid_2=1。它们是 program 网格维度，不是元素数量。`*bound_args.values()` 按绑定顺序把五个值展开给启动层。看到 B 也在这个接口中，不要误认为设备机器函数一定多一个 B 寄存器参数；launcher 会根据 signature/constants 处理它。

launch_metadata 是供启动钩子等使用的元数据。kernel.function 是已装载函数句柄，packed_metadata 是后端打包的启动信息。它们都与 GPU 上 x 的十个浮点数不同。

### 9. 编译、装载、发射与完成分清楚

[compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) 第494–498行：

```python
@property
def run(self):
    if self._run is None:
        self._init_handles()
    return self._run
```

property 让读取 `kernel.run` 时先执行这段 host 代码。首次需要时初始化句柄；第468行创建 launcher，第485–486行通过驱动 load_binary 得到 module/function 与资源信息。之后返回可调用 launcher。完整装载与驱动参数转换留给 [8.1](08-backend-debug.md#sec-8-1)。

所以三种对象要区分：JITFunction 保存待编译函数；CompiledKernel 保存产物和元数据；设备函数句柄标识已装载设备程序。返回的 Python kernel 对象不是计算结果张量。

4.1 的 wrapper 不接收这一返回值，而是 `return output`。output 是之前分配、之后由设备写入的同一个张量对象。提交工作不等于所有 GPU 计算已完成；是否等待完成由流与后续同步决定。

warmup 的接口在 jit.py 第365–366行把 warmup=True 交给 run，因此可以准备编译产物而跳过这里的实际发射分支；6.1 会用它获取资源信息。这也不表示 warmup 什么设备初始化都不可能做。

### 10. 用五句话重新走一遍

1. add 分配 output，并把 x/y/output/N 与 B 交给方括号返回的代理。
2. 代理保存 grid，把 args/kwargs 交给 JITFunction.run。
3. binder 按形参名字形成 bound_args，另给出特化描述和选项。
4. 缓存未命中时构造 ASTSource，compile 返回产物；命中时复用已有产物。
5. grid 接收 bound_args 得到三份工作，launcher 接收网格、流、句柄与实参，提交设备执行。

**自测与答案：** 第二次换 x/y 数值、保留同样类型和特化条件，需要为了每个新元素编译一次吗？不需要；缓存依据编译条件，kernel 的职责正是启动时读取新数据。把 B 改为8后 grid 多少？`cdiv(10,8)=2`；同时 B 是不同编译常量，不应只改 grid 而复用不匹配的设备程序。

> 回到论文：原论文 §3（PDF第3–4页，见[定位表](../appendices/reference.md#sec-a-1)）帮助理解“tile 参数决定一份程序处理多少逻辑元素”。本章的 Python JIT/binder/ASTSource 是现代实现，不是论文 Triton-C 的原样接口；旧 host 路线在历史篇独立阅读。

4.1–4.2 静态核对了对象与调用流；4.3 只记录一次实际启动结果，没有测量编译时间、缓存命中次数或启动延迟。下一章接住的是 ASTSource 和编译选项，解释怎样生成 IR；不把纸笔数值冒充实际编译输出。

## 4.3 实际运行：同一个 kernel 在 A800 上

4.1、4.2 追清了对象流。这一节把同一份 kernel 交给从固定提交构建的 Triton，观察实际 grid、输出、编译阶段和资源计数。

**环境。** GPU 为 NVIDIA A800 80GB PCIe；PyTorch 2.10.0+cu129；Triton 3.9.0，源码固定 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`，在 `/research` 的隔离副本中构建，`triton/` checkout 未修改。

**命令。**

```bash
python examples/actual_vector_add.py
```

**输出。**

```text
triton_version 3.9.0
torch_version 2.10.0+cu129
cuda_version 12.9
device NVIDIA A800 80GB PCIe
n 10 block 4 grid (3,)
x [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
out [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
reference [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
asm_stages ['cubin', 'llir', 'ptx', 'source', 'ttgir', 'ttir']
n_regs 12
n_spills 0
shared_bytes 0
second_call_out [21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0]
```

**怎样读。**

- `grid=(3,)` 与 4.1 的 `cdiv(10, 4)` 一致。三个 program 分别覆盖 0–3、4–7、8–11；最后一块的两个无效位置由 mask 截断。
- `out` 是实际的十个结果。`second_call_out` 说明换一组 x/y 数值后，同一设备程序读取新的参数，而不是把输入值编译进二进制。
- `asm_stages` 是这次编译保留的阶段；`n_regs=12`、`n_spills=0`、`shared_bytes=0` 是实际资源计数，不是从 B=4 推导出来的。
- 这些数字只对应这次小输入和后端配置，不能推广为其他 shape 的固定布局或性能结论。

**边界。** 本节只验证十个数、B=4 的一次运行；没有测编译时间、缓存命中次数或带宽。

## 第 4 章小结

把这一章压缩成一条对象链：

```text
add(x, y)
  → output = empty_like(x)，N = numel()
  → grid 函数（尚未求值）
  → add_kernel[grid]（__getitem__ 返回代理）
  → 圆括号调用 → JITFunction.run
  → binder 按形参名得到 bound_args、specialization、options
  → 缓存查找
      ├─ 命中：已有 CompiledKernel
      └─ 未命中：ASTSource → compile → make_ir → TTIR → TTGIR → LLIR → PTX → cubin
  → grid(bound_args) 得到 (3,)
  → CompiledKernel._init_handles：装载二进制、取得 function 句柄、构造 launcher
  → launcher 把 grid、stream、句柄和实参提交给驱动
  → GPU 执行三个 program；host 返回 output 对象
```

### 综合检查

1. `add_kernel[grid]` 已经启动 GPU 了吗？没有，它返回一个保存了 grid 的代理；圆括号调用才进入 `run`。
2. 第二次调用只换 x/y 的数值，为什么要复用同一个编译版本？因为缓存键依据类型、特化条件和编译选项，普通输入元素值不属于编译条件。
3. B 从 4 改成 8，grid 和编译产物会怎样？grid 重新计算；B 是 `constexpr`，可能产生另一个编译版本，不能只改 grid 而复用旧产物。
4. `warmup=True` 和性能测试里的“先跑几次”是一回事吗？不是；前者跳过实际发射，只准备编译产物和资源信息。
5. `return output` 时 GPU 一定算完了吗？不一定；提交与完成是不同事件，跨 stream 或 CPU 读取需要额外同步。
