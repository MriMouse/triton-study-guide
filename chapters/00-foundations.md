# 第 0 章 基础：从一次 Python 加法到 GPU 执行

[返回总目录](../README.md) · [下一章：历史路线（一）](01-legacy-frontend.md) · [附录](../appendices/reference.md)

这一章是全书的地基。你不需要先懂 GPU、编译器或 PyTorch 的内部实现；只要会变量、函数、列表、循环和基本的 Python 语法，就可以从“两个数组逐元素相加”开始。

本章不追求把 Triton 的全部概念一次讲完，而是先把四个最容易混淆的层次分开：

1. **职责层**：模型、算子、kernel、编译器、运行时、GPU 分别负责什么。
2. **语法层**：读真实源码时会遇到哪些 Python/C++ 写法。
3. **地址层**：shape、stride、类型化指针和广播怎样决定一个元素在哪里。
4. **执行层**：program、CTA、warp、SM、寄存器和共享内存怎样对应。

最后两部分补上仓库地图和五个反复使用的追问。读完后，你不必记住所有术语，但应该能在看到一段 kernel 时先问出正确的问题。

## 本章地图

| 部分 | 回答的问题 | 读完后可以做什么 |
|---|---|---|
| 0.1 | 一个加法为什么牵涉框架、编译器和 GPU？ | 说清模型、算子、kernel、编译器、运行时的分工 |
| 0.2 | 这些 Python/C++ 语法在真实代码里是什么意思？ | 看懂 `self`、闭包、`*args`、引用、指针、visitor |
| 0.3 | 逻辑坐标怎样变成内存地址？ | 手算 shape/stride/转置/广播和二维指针 |
| 0.4 | program、CTA、warp、SM 和存储是什么关系？ | 分清逻辑 tile 与物理线程，理解同步和访存直觉 |
| 0.5 | 以后去哪里找代码？ | 按职责打开仓库，而不是从目录名猜 |
| 0.6 | 读代码时先问什么？ | 用五个问题检查自己是否真的跟上了对象流 |

**阅读建议：** 第一次读不必背下每个 C++ 语法点。后面章节出现 `self`、`->`、`[]`、`_semantic` 时，再回到 0.2 查对应解释即可。基础篇的目标是建立“够用的地图”，不是替代 C++ 或 GPU 教材。

<a id="sec-0-1"></a>
## 0.1 一个加法为什么牵涉框架、编译器和 GPU

本章从你会写的函数调用出发，定位整个学习工程。源码基于[现代快照](../appendices/source-version.json)，以下数字推演均为教学例子，不是性能实测。

### 1. 先区分“想算什么”和“怎样执行”

给出两个输入：`x=[0,1,2,3]`、`y=[1,2,3,4]`，目标是逐元素得到`[1,3,5,7]`。数学关系是`out[i]=x[i]+y[i]`。

当你在PyTorch里写`x+y`，框架知道张量的类型、设备和形状，可以选择已有实现。手写Triton则是自己描述其中的设备计算，再让Triton生成实现它的GPU程序。它不会自动替你决定模型层应该采用什么数学公式。

| 层次 | 回答的问题 | 本例中是什么 |
|---|---|---|
| 数学运算/算子 | 对输入应产生什么结果？ | 对应位置相加 |
| 算法与分工 | 每份工作计算哪些输出？ | 每个program负责B个位置 |
| 编译器 | 怎样把这个表达转换为合法设备程序？ | 类型检查、IR、布局、指令生成 |
| 运行时/驱动 | 怎样提交这次执行？ | 绑定参数、加载二进制、向stream发射 |
| GPU | 执行哪些指令、读写哪些地址？ | 读取两组数据、相加、写回 |

同一个算子可以有多种kernel实现；一个kernel也可以融合多个算子。不要把层与kernel数目绑在一起。

### 2. 直接打开一段能认出来的代码

在[官方向量加法](../../triton/python/tutorials/01-vector-add.py)第62–78行，函数`add`的关键执行语句如下。这里删除了注释、设备断言，属于**非完整摘录**：

```python
output = torch.empty_like(x)
n_elements = output.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
return output
```

第一行在host端分配输出张量；此时没有做加法，内容尚未初始化。第二行读出元素数量作为N。第三行创建一个稍后用于计算启动数量的函数，方括号中的 `grid` 就指向它。等第 0.2 节再拆解[两处 lambda 和两对括号](#python-launch-syntax)。

第四行是边界：它把输入、输出、N、B交给Triton接口。运行时可能先准备编译版本，再提交设备执行。第五行返回的是张量对象；设备结果可能还在生成中。

这五行串起的不是“五个GPU kernel”，而是一次算子调用的host准备与发射过程。

#### 对照真正的设备计算

同一文件第50–54行，删除其中一行注释：

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

`offsets`、`mask`由本函数前面的语句构造，4.1会逐行追到它们。这里先看输入输出：load表达从地址获得逻辑值，`x+y`产生计算结果，store表达写回指定地址。kernel中的output是当前program的一块逻辑值，host的output是完整输出张量，二者通过output_ptr所指存储连接。

先把 `tl.load(address, mask=mask)` 读成「从 address 取值，由 mask 决定哪些位置允许读取」。这些地址与真假条件可以各有多项，`store` 则把有效位置写回。编译时，Triton 读取这段源码并构造操作；执行时，编译好的设备程序才真正读取数据。后面每章都会明确当前在哪一层。

### 3. 大模型层与这些小算子怎样联系

一个线性层可写成`Y=XW+bias`。若X是`(M,K)`，W是`(K,N)`，输出是`(M,N)`；每个输出需沿K维累加。随后还可能有激活、归一化等计算。

在[矩阵乘教程](../../triton/python/tutorials/03-matrix-multiplication.py)第238–255行，kernel输入包含：

```python
a_ptr, b_ptr, c_ptr,
M, N, K,
```

前三者是输入/输出存储的基地址，后三者描述问题大小。签名后还接有stride和分块参数。它并没有传入“这是Transformer第几层”，因为这个kernel只需要完成局部矩阵乘语义。

模型层把算子组合成网络；算子实现关注shape、dtype、地址和算法。学习Triton的入口是后者，随后再理解这些实现如何服务Attention、MLP或Norm。

### 4. 融合为什么有吸引力，又为什么不是无条件更快

设`tmp=x+y`之后紧接着`out=max(tmp,0)`。两个分离kernel可能先将tmp写入全局存储，再由下一个kernel读取。如果同一kernel在加法后直接做max，只写最终out，有机会省掉tmp的全局读写。

下面是**教学公式，不是仓库里的新实现**：

```text
分离：load x/y → add → store tmp → load tmp → max → store out
融合：load x/y → add → max → store out
```

收益来自数据流改变，不来自函数名字变短。融合也可能增加寄存器压力、改变并行度或限制指令选择，所以必须明确输入和环境再测；本书不把这个例子写成实测加速结论。

### 5. 编译器为什么要保留多种表示

源码方便人表达`x+y`。编译器还需要明确它是指针加偏移、整数加法还是浮点加法，并知道每个值的shape和数据依赖。再往下需要决定值分给哪些线程、使用哪些资源、如何生成目标代码。

因此现代路线依次引入AST、TTIR、TTGIR、LLVM、PTX/cubin。它们不是同一段计算重复执行多遍，而是同一程序在编译中被逐步细化。5.1/5.2会读实际短代码和测试IR，8.1进入目标代码。

历史路线则读Triton-C、自有IR和旧LLVM后端。两份树固定不同年代，是为了比较表示和机制，不是让你把两套构建环境混装。

> 回到论文：§1、Figure 2（PDF第2页）解释旧系统分工；§3.2.1解释tile表达的动机。现代Python/MLIR实现变化见[论文映射](../appendices/reference.md#sec-a-1)，不能把旧Figure 2当当前调用图。

### 6. 学完之后具体能做什么

先从4.1学会为一个小kernel追清输入与地址；4.2–5.2追清程序怎样变成IR；softmax/GEMM/Attention将算法和数据复用连接起来；布局、pass、分配、同步与后端帮助你读懂为什么生成代码会不同。

读完不是等于掌握仓库中所有架构的每个文件，而是可以独立追一条新算子路径、读懂关键变换、构造正确性验证，并识别尚未证实的性能判断。高级稀有分支仍应按具体问题继续查源。

### 自测

1. **`torch.empty_like`已经完成加法了吗？** 没有，它准备输出存储。
2. **融合两个算子一定只产生一个更快kernel吗？** 不保证，实际实现与资源代价仍需检查。
3. **TTIR→PTX是否表示数据在这两层各计算一次？** 不是，它们是编译过程的程序表示。
4. **为什么LLM学习手册从加法开始？** 它能用最少数学，先建立host/设备、地址、边界和启动的完整闭环。

**从职责地图到可读代码。** 0.1 说明了模型、算子、kernel、编译器和运行时分别负责什么。下一步会遇到大量 Python 与 C++ 写法：`self`、闭包、`*args`、引用、指针、visitor。0.2 不教完整语言，而是把这些语法直接连到后面会读到的真实接口上。

<a id="sec-0-2"></a>
## 0.2 用真实调用补齐 Python 与 C++ 阅读语法

本章不列完整语言教程，而是把后面反复出现的语法连到实际对象。现代/历史源码版本分别见[版本](../appendices/source-version.json)和[历史版本](../appendices/legacy-version.json)。教学例子使用普通 Python；Triton 摘录后会标出可以继续追的源码。

### 1. 点号先从名字空间看起

[向量加法](../../triton/python/tutorials/01-vector-add.py)第23–24行：

```python
import triton
import triton.language as tl
```

第一行让本文件获得模块名triton；第二行为其语言模块起别名tl。因此`triton.jit`和`tl.load`来自不同接口层。点号只表示访问对象上的属性，并不自动改变执行设备。

如果一个文件写`from .compiler import compile`，相对import以当前Python包为起点；不是以你执行shell时的目录为起点。追一个名字应先看本文件import与赋值，不能因为都叫compile就认定是同一个函数。

<a id="python-launch-syntax"></a>
### 2. 拆开 `add_kernel[grid](...)` 里的 Python 语法

读这节是为了回答一个实际的源码问题：**方括号里给了 grid，为什么后面还能写一对圆括号传张量？最后是谁拿到了这些值？** 先学会把这行拆成两次操作，再追进 JIT。

#### 2.1 调用时按位置给值，或写出名字

看普通的 `def show(size, count): ...`。`size` 和 `count` 是函数定义里的名字（**形参**）。调用 `show(4, 10)` 时，Python 按顺序让 `size=4`、`count=10`，因此这两个值叫**位置实参**。调用 `show(4, count=10)` 时，`4` 仍按位置交给 `size`，`10` 则按名字交给 `count`。后一种实参叫**关键字实参**，也常说「命名参数」。调用里的 `count=10` 意思是用这个名字传值。

#### 2.2 tuple 和 dict 中各装了什么

| 写法 | 类型与含义 | 如何取出一个值 |
|---|---|---|
| `(4, 10)` | `tuple`（元组），有顺序的一组值，建好后各位置不能换成别的值 | `values[0]` 取第一个值 `4` |
| `{"BLOCK_SIZE": 4}` | `dict`（dictionary，字典），用键查对应的值 | `options["BLOCK_SIZE"]` 按字符串键取出 `4` |
| `()`、`{}` | 分别是空元组和空字典 | 目前没有任何项 |
| `(3,)` | 只有一项的元组，逗号区分它与表示数字 3 的 `(3)` | `grid_shape[0]` 得到第一维的启动数量 `3` |

字典里的冒号把一个键与一个值配成一项。元组里的项目可以是数字，也可以是整个张量对象。比如 `(x, y, 10)` 有三项，无论 `x`、`y` 各自包含多少个元素。

#### 2.3 星号在定义和调用里各做一件事

`args` 是 arguments（实参）的惯用缩写，`kwargs` 是 keyword arguments（关键字实参）的惯用缩写。这两个变量名可以换成 `positional` 和 `named`。让 Python 执行下面行为的是星号所在的位置。

```python
def show(size, count):
    return size, count

def relay(*args, **kwargs):
    print(args)
    print(kwargs)
    return show(*args, **kwargs)

print(relay(4, count=10))
# (4,)
# {'count': 10}
# (4, 10)
```

1. 调用 `relay(4, count=10)` 时，定义里的 `*args` 让 Python 把未被其他形参接走的位置实参装进一个元组，再把这个元组交给局部变量 `args`。这里是 `(4,)`。
2. 定义里的 `**kwargs` 让 Python 把未被其他形参接走的关键字实参装进一个字典，再把字典交给 `kwargs`。这里有一项，键为字符串 `"count"`，值为 `10`。这就是有些资料里「收成字典」所指的具体动作。
3. 调用 `show(*args, **kwargs)` 时，星号要求 Python 拿出元组中的每项，按顺序传递，再拿出字典中的每项，按键名传递。这次等价于 `show(4, count=10)`。因此写成 `show(args, kwargs)` 会得到另一个结果：两个容器会各自作为一个完整的位置实参传进去。

#### 2.4 lambda 创建一个可以稍后调用的函数

`lambda value: value + 1` 读作「创建一个接收 `value` 并返回 `value + 1` 的函数」。冒号左边写形参，右边是调用时计算并返回的表达式。写 `plus_one = lambda value: value + 1` 时，变量 `plus_one` 指向刚创建的函数；写 `plus_one(4)` 时才做加法。

再看一个只差一行的版本：

```python
def make_adder(amount):
    return lambda value: value + amount

add_four = make_adder(4)
add_ten = make_adder(10)

print(add_four(3))  # 7
print(add_ten(3))   # 13
```

执行 `make_adder(4)` 时，`amount` 是这次外层调用的局部名字，指向 `4`。返回的是内层函数。稍后执行 `add_four(3)`，这个内层函数从当次圆括号获得 `value=3`，再从创建它的那次外层调用找到 `amount=4`。`add_ten` 由另一次外层调用创建，所以找到的是另一个 `amount`。`return` 之后内层函数还能访问这个名字，这个机制叫**闭包**。它保留的是访问外层变量的能力。把这个例子的 `lambda` 写成 `make_adder` 内层的 `def add(value): return value + amount` 再返回 `add`，效果相同。

#### 2.5 应用到 Triton 的两对括号

[jit.py](../../triton/python/triton/runtime/jit.py#L371) 的 `KernelInterface.__getitem__`（省略文档字符串）：

```python
def __getitem__(self, grid) -> T:
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

把原调用写成两行。这里的 `x`、`y`、`out` 是三个张量对象，`10` 是元素总数，`BLOCK_SIZE=4` 表示每份程序处理 4 个逻辑位置。

```python
launch = add_kernel[grid]
launch(x, y, out, 10, BLOCK_SIZE=4)
```

| 时刻 | 谁接到哪些值 | 接下来做什么 |
|---|---|---|
| 求 `add_kernel[grid]` | Python 调用对象的 `__getitem__`。`self` 是 `add_kernel` 包装对象，`grid` 是方括号里的值 | 创建并返回 `lambda`。它以后能找到这次 `__getitem__` 的 `self` 和 `grid` |
| 求 `launch(...)` | 内层函数的 `args` 是 `(x, y, out, 10)`，`kwargs` 是 `{"BLOCK_SIZE": 4}` | 开始执行冒号右边的 `self.run(...)` |
| 进入 `run` | 转交四个位置实参、`BLOCK_SIZE=4`，再加上接口显式给出的 `grid=grid`、`warmup=False` | 运行时按 kernel 形参表绑定参数，后面再求 grid 并发射 |

`grid=grid` 左边是传给 `run` 的参数名，右边取外层方括号留下的值。在向量加法里这个值本身也是函数：`lambda meta: (...,)`。那一处函数记住的是 host 函数里的 `n_elements`，调用时按 `meta["BLOCK_SIZE"]` 查块大小。以 `N=10`、`B=4` 为例，向上取整得到三个 program，返回值是 `(3,)`。因此源码里的两个 lambda 有不同工作：一个计算启动数量，一个把启动所需参数转交给 `run`。从这一步继续读[第 4 章 JIT 路径](04-vector-add-jit.md#sec-4-2)。

### 3. self不是隐藏的全局变量

[jit.py](../../triton/python/triton/runtime/jit.py)第362–369行：

```python
class KernelInterface(Generic[T]):
    run: T

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        raise NotImplementedError("run not implemented")
```

class定义一类对象的接口。`Generic[T]`服务于类型表达，第一次可以先只看KernelInterface。`run: T`是注解，不等于已经创建一个具体编译器。

调用`obj.warmup(...)`时，Python会把obj作为self交给方法。于是`self.run`取的是这个对象上实际可用的run。基类里的run抛异常表示“这个默认实现不能做事”，子类提供实际实现后才有完整行为。

`JITFunction`沿用该接口，所以你不能读完基类的`raise`就断言所有kernel启动都会失败。4.2会跟到真正覆盖的方法；这里先学会分清接口与具体实现。

warmup还用`map(MockTensor.wrap_dtype,args)`处理参数，并把warmup设True。它的语义不是性能测试里的“先跑十次”，二者名字相同、用途不同，后面按真实分支解释。

### 4. 字典查找与参数绑定不等于编译

下面是**教学小例**，不冒充Triton完整缓存键算法：

```python
cache = {}
key = ('fp32', 4)
kernel = cache.get(key)
if kernel is None:
    kernel = '假设这里得到一个编译产物对象'
    cache[key] = kernel
```

第一行创建容器；第二行用元组 `('fp32', 4)` 作为键，表示数据类型和块大小的组合；第三行找这个条件对应的值。字典没有该键时get返回None，再执行分支。cache赋值存的是对象引用，不是重新运行设备计算。

真实JIT还要考虑类型、常量、后端与选项，不能把上面两个字段当真实缓存键。此例只解释“查找→未命中→准备对象→保存”的控制流。

### 5. C++里的引用、指针与所有权怎样读

切到[旧function.cc](../../triton-legacy/lib/runtime/function.cc)第191–196行：

```cpp
std::unique_ptr<ir::module> function::make_ir(Parser& parser) {
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  return std::unique_ptr<ir::module>(module);
}
```

逐句解释：

| 语句 | 读取或创建的对象 | 下一步怎样使用 |
|---|---|---|
| `function::make_ir` | function类的方法；`::`说明作用域 | 调用者请求从parser产生旧IR |
| `Parser& parser` | parser对象的引用，非复制一整个parser | 读取已解析AST |
| `ir::module* module = new ...` | 创建旧IR模块，局部变量保存裸指针 | 给Generator作为输出容器 |
| `Generator gen(&parser)` | 创建前端生成器；`&parser`取得地址 | gen能访问parser持有的语法树 |
| `gen.Gen(module)` | 将AST转换成IR并填入module | module内容发生变化，不是GPU输入变化 |
| 返回unique_ptr | 将模块交给独占所有权包装 | 调用方持有返回对象，生命周期结束时释放模块 |

同一个`&`在声明`Parser&`中表示引用，在表达式`&parser`中表示取地址。`*`在`ir::module*`中表示指针类型，不是乘法。

unique_ptr可理解为“明确只有一个拥有者负责释放资源”的对象。它与一个GPU设备地址不是同一层东西；这里拥有的是编译器内存中的IR模块。

### 6. 箭头、继承、动态类型与visitor

[旧coalesce.cc](../../triton-legacy/lib/codegen/transform/coalesce.cc)第19–25行：

```cpp
void coalesce::extract_io_use(ir::value *v, std::set<ir::io_inst*>& result) {
  for(ir::user* u: v->get_users()){
    auto i = dynamic_cast<ir::io_inst*>(u);
    if(i && i->get_pointer_operand() == v)
      result.insert(i);
  }
}
```

输入v是一个IR值指针；result是引用，函数会直接填充调用方的集合。返回类型void表示不靠return交结果。

`v->get_users()`通过指针访问成员，取得使用v的对象。for依次取出每个user作为u。`dynamic_cast`判断这个user是不是I/O指令；失败时i为空指针。

`i && ...`先检查i存在，再安全访问成员。即使i是I/O，还要确认它使用v的方式是“把v当指针操作数”，不是当待存数值。最后把符合条件的指令加入result。

这里每一层判断都在缩小匹配范围，不是多余样板。后面读pass时，要把“它只处理哪些对象”与“它会怎样改”分开。

visitor则是让不同节点进入对应处理函数的常见模式。旧`Generator::VisitBinaryOp`与现代`CodeGenerator.visit_BinOp`都属于这类组织方式；方法名字相近，不证明内部IR或类型体系相同。

### 7. C++ lambda里的 [&] 怎样连接当前分析

[现代Coalesce.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)第87–90行的开头：

```cpp
moduleOp.walk([&](Operation *curr) {
  Value ptr = getMemAccessPtr(curr);
  if (!ptr)
    return;
```

这是有意截断的函数体片段，闭括号和其余处理在源码后续。walk遍历module中的operation，每次把当前操作交给回调形参curr。`[&]`表示回调可以按引用使用外层需要的变量，类似Python闭包但语言与生命周期规则不同。

取出内存访问指针，找不到就return退出**本次回调**，不是终止所有module遍历。看见return时先找函数边界，才能判断它到底返回到哪里。

### 自测

1. **`*args`会展开一个张量的十个元素吗？** 不会，它收集/展开调用的位置参数对象。
2. **`gen.Gen(module)`执行了GPU加法吗？** 没有，它处理AST与IR。
3. **看到`class ...Base`中的抽象方法即可判断实际行为吗？** 不行，需追到实际子类方法和对象来源。
4. **`result.insert(i)`为什么没有return仍能改变输出？** result按引用传入，集合本身被修改。

遇到不认识语法，先问它在当前对象流中的作用；不必为了读一段pass先学完整C++标准库。

**从语法到数据在哪里。** 能读懂 `x_ptr + offsets` 的写法，不等于能算出 `offsets` 指向哪个元素。0.3 把 shape、stride、类型化指针、转置视图和广播放在同一张地址表上，让“逻辑坐标 → 元素偏移 → 字节地址”这条链可以手算。

<a id="sec-0-3"></a>
## 0.3 从坐标走到地址：shape、stride、指针与广播

本章的闭环是：给定二维张量的坐标，手算它的元素偏移与字节地址，再读懂官方GEMM里的指针广播。所有地址都是纸笔假地址，不应执行访问；源码版本见[记录](../appendices/source-version.json)。

### 1. shape不能单独决定物理位置

设张量A的形状是`(2,3)`，值为：

```text
A = [ [0, 1, 2],
      [3, 4, 5] ]
```

shape告诉我们有2行3列，有效坐标为i=0或1、j=0到2。如果按连续行主序存放，内存元素顺序是`[0,1,2,3,4,5]`。

此时stride为`(3,1)`：行坐标增加1，跨3个元素；列坐标增加1，跨1个元素。若基指针已指向视图的第一个元素：

```text
element_offset(i,j) = i*stride_0 + j*stride_1
byte_address(i,j)   = base_address + element_offset*element_size
```

FP32的每元素4字节，假设首地址1000，则A[1,2]偏移`1*3+2*1=5`个元素，地址`1000+5*4=1020`。数学坐标(1,2)、元素偏移5、字节地址1020，是三个不同对象。

### 2. 一个转置视图为什么能不复制数据

将A转置，逻辑值是：

```text
T = [ [0, 3],
      [1, 4],
      [2, 5] ]
```

它可以仍使用同一份底层存储，只把shape改为`(3,2)`，stride改为`(1,3)`。此时T[2,1]的偏移是`2*1+1*3=5`，仍指向原A[1,2]。

如果错误地把新shape当连续行主序，用stride`(2,1)`计算T[1,0]，会得到偏移2；正确是`1*1+0*3=1`。这可能仍落在分配内、没有越界，却读错元素。**地址合法不等于数学索引正确。**

视图也可能从原存储中间开始。本章base已指向视图首元素，因此不重复加storage offset；若你从底层storage首地址计算，才需要另行加入视图起点。

### 3. 对应到真实kernel参数

[矩阵乘教程](../../triton/python/tutorials/03-matrix-multiplication.py)第246–248行是签名中的连续片段：

```python
stride_am, stride_ak,  #
stride_bk, stride_bn,  #
stride_cm, stride_cn,
```

它们不是六个张量，而是每一维的元素步长。A形状(M,K)，所以`stride_am`沿M维走一步，`stride_ak`沿K维走一步。B形状(K,N)，对应bk与bn；C形状(M,N)，对应cm与cn。

名字后缀不是语言强制约定，而是作者提示。真实含义还要看调用者传入的stride与地址公式是否一致。6.2会追完整调用，这里先把接收端的公式算清。

### 4. 类型化指针为什么不能再手乘字节数

4.1中`x_ptr+offsets`使用的是带元素类型的指针。下面是**两种纸笔写法的对应关系**：

```text
Triton表达：FP32指针p + 5
等价字节位置：byte_base + 5*4
```

所以不要改成`p + 5*4`，否则变成前进20个FP32元素。字节公式服务你理解地址，不能未经类型分析就原样代入指针运算。

在[semantic.py](../../triton/python/triton/language/semantic.py)第273–282行，原样摘录：

```python
if input_scalar_ty.is_ptr():
    other_handle = other.handle
    if other.dtype.is_int_unsigned() and other.dtype.int_bitwidth < 64:
        i64_ty = other.type.with_element_ty(tl.int64).to_ir(self.builder)
        other_handle = self.builder.create_int_cast(other.handle, i64_ty, False)
    return self.make_tensor(self.builder.create_addptr(input.handle, other_handle), input.type)
elif input_scalar_ty.is_floating():
    return self.make_tensor(self.builder.create_fadd(input.handle, other.handle), input.type)
```

上文已经完成二元操作类型整理；本段input与other是编译期的逻辑值包装，handle引用IR值。

第一分支确认左边是指针。先取得偏移的IR handle；对某些小于64位的无符号偏移做零扩展，是为了保持其正数含义。然后调用builder生成addptr操作，返回带原结果类型的包装。

第二分支处理浮点加法，调用的是create_fadd。**同一个源码加号因类型不同，生成的操作不同。** 此时不是在CPU上解引用GPU地址，也没有从x_ptr读到数据。

### 5. None维度怎样产生二维指针表

在[官方GEMM](../../triton/python/tutorials/03-matrix-multiplication.py)第293–295行：

```python
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

这里offs_am/offs_bn由前两行构造，分别选择当前输出块对应的行/列。为单独学地址，纸笔取offs_am=[0,1]、offs_k=[0,1,2,3]，A行主序stride=(3,1)。B K取4是逻辑补齐，不把A实际扩容。

#### 5.1 先改变逻辑形状

`offs_am[:,None]`把`(2,)`看成`(2,1)`：

```text
[0,1] → [ [0],
           [1] ]
```

`offs_k[None,:]`把`(4,)`看成`(1,4)`，即`[[0,1,2,3]]`。None增加长度为1的轴；不是“取不到所以返回None”。

#### 5.2 分别乘上维度步长，再广播相加

```text
行部分 ×3：       列部分 ×1：
[[0],             [[0,1,2,3]]
 [3]]

广播相加后的元素偏移：
[[0,1,2,3],
 [3,4,5,6]]
```

广播使两个维度分别扩展到(2,4)，不要求先在全局内存复制出两个矩阵。然后加上a_ptr，就成为(2,4)的指针张量，load后才是数值张量。

#### 5.3 mask必须遵循逻辑坐标

A只有3列，所以每行第4个逻辑位置都不应读。第一行偏移3虽然恰好落在A下一行的合法存储里，却不是A[0,3]这个不存在坐标的正确值。

因此K方向需要mask `offs_k < K`。本例mask每行为`[T,T,T,F]`。它表达数学有效性，不只是检测地址是否仍在整个分配范围内。6.2会解释官方余数索引和K尾块的具体策略。

### 6. 三类“排列”不要混用

| 概念 | 改变什么 | 本章中的例子 |
|---|---|---|
| 张量stride/视图 | 逻辑坐标如何找到内存元素 | 转置视图stride从(3,1)变(1,3) |
| 逻辑shape/广播 | 操作数的维度怎样对应 | (2,1)+(1,4)→(2,4) |
| GPU layout | 逻辑值怎样分布到线程/寄存器等 | 本章尚未决定，7.1具体带读 |

一个广播表达式不自动决定最终线程分布；一个GPU layout也不会替你修复传错的stride。先保证地址语义正确，再讨论访存效率。

> 回到论文：§3.2.2/Listing 4和§4.2.2/Figure 4讲广播；Listing 1含二维tile指针。现代语法的None对应今天接口，不是旧newaxis语法的原样代码，见[映射](../appendices/reference.md#sec-a-1)。

### 自测

1. **FP16、stride=(5,1)，坐标(2,3)元素偏移与字节偏移？** 13元素、26字节。
2. **转置视图shape=(3,2)、stride=(1,3)，(1,1)偏移？** 4元素。
3. **一个指针张量是否已经含输入数值？** 没有，需要load。
4. **地址落在分配内，mask就能省略吗？** 不能，逻辑坐标仍可能无效，或属于别的行。
5. **本章公式决定哪个lane读哪项了吗？** 没有，那是layout与lowering的工作。

**从地址到谁在执行。** 0.3 只回答“数据在哪里”，还没有回答“哪个 program、哪个线程、哪块存储在做”。0.4 把逻辑 tile、program instance、CTA、warp、SM、寄存器和共享内存放在同一张图里，并解释合并访存、同步和小 kernel 的启动开销。

<a id="sec-0-4"></a>
## 0.4 把 program、线程与存储放在同一张图里

本章以经典单CTA的NVIDIA入门kernel为范围，解释逻辑程序怎样和物理执行联系。AMD、多CTA、warp specialization的不同安排在[10.1](10-modern-extensions.md#sec-10-1)单独讲，不把基础图强套所有后端。

### 1. 先把两条时间线分开

```text
host：创建张量 → 必要时编译 → 向stream提交 → Python继续执行
GPU：                                  → 等待前序工作 → 执行kernel
```

host提交完成不保证GPU已完成。stream可先当作具有顺序关系的工作队列；相同stream上的依赖按顺序安排，跨stream依赖需明确连接。

源文件中`@triton.jit`函数定义靠前，也不代表定义时已经执行。这是源程序顺序、编译顺序、设备工作顺序三个层次，后面的章节会反复标出边界。

### 2. grid、CTA、warp与SM各管哪一层

| 名称 | 入门含义 | 不能误解成 |
|---|---|---|
| grid | 一次启动的program/块组织 | 输入张量shape |
| program instance | 同一kernel的一份逻辑工作 | 一个必须只算一个元素的线程 |
| CTA/thread block | 一组可协作、共享资源的线程 | 整张GPU |
| warp | NVIDIA执行组织中的32线程组 | 一个矩阵tile |
| SM | 承载线程块执行的硬件单元 | 一根简单CPU式线程 |

对本章限定的经典单CTA路线，可以先把一个program对应一个thread block。未来多CTA和warp-specialized程序会引入更复杂的组织，不用这个简化推出普遍一一映射。

NVIDIA后端的[CUDABackend选项](../../triton/third_party/nvidia/backend/compiler.py)第121–124行分别可见`num_warps=4`和`warp_size=32`。这不是说所有GPU后端都使用相同大小。

### 3. 读到 BLOCK_SIZE 时，先问逻辑工作量

[向量加法](../../triton/python/tutorials/01-vector-add.py)第39、44–45行，删除中间注释后：

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

pid是当前program编号；起点是pid乘B；arange给出一组逻辑位置。若B=1024，offsets含1024个整数。源码没有说“1024个线程各拿一个”。

假设经典配置num_warps=4，对NVIDIA即128个线程。一个逻辑tile的元素可以由线程各持有若干项，也可能存在布局允许的复制。`1024/128=8`只给平均工作量直觉，不能断言线程t必须持有连续8项。

最终映射看GPU layout，7.1会从真实测试手算，不在这里编造一个“默认必然布局”。

### 4. 用标量线程例子看差异

下面是**CUDA教学片段**，未编译，不是本仓库向量加法的输出代码：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) out[i] = x[i] + y[i];
```

一个线程先用块号、块内线程数和线程号算i，再处理一个位置。若希望每线程处理多项，需要继续写循环或其他分工。

Triton的逻辑program直接写一组offsets，再由编译器选具体线程分布。CUDA也能采用分块算法，Triton也最终需要线程执行；区别在源程序把哪些实现选择交给编译器。

### 5. 数据可以放在哪里

对本章范围，可先采用如下层次直觉：

```text
设备全局地址中的输入/输出
         ↓ load/store，可能经过缓存
线程执行时持有的寄存器值
         ↔ 必要时通过warp指令或共享内存交换
同一个CTA可使用的共享存储
```

寄存器帮助保存执行时的值；共享内存给块内协作提供可寻址存储；设备全局存储容量更大、访问代价通常更高。实际GPU还有缓存、专用矩阵存储和异步设施，不能用这张图代替完整硬件手册。

`tl.load(ptrs)`返回逻辑值，不是在源码层显式申请共享内存。编译器可以根据布局和操作选择寄存器与中间存储。源码里一个变量，也不保证对应一个物理寄存器。

### 6. 合并访存为什么要看一组线程

假设四个参与访问的lane需要四个FP32值。下面只画地址关系，不固定硬件事务大小：

```text
连续例：lane0→1000，lane1→1004，lane2→1008，lane3→1012
分散例：lane0→1000，lane1→2024，lane2→3048，lane3→4072
```

硬件以一定事务粒度搬运数据，连续且适合对齐的访问往往能更充分利用搬来的数据。分散访问可能需要更多事务；具体数目还取决于指令、缓存、架构与参与mask。

“连续的逻辑offsets”是否变成“相邻lane连续访问”，取决于layout。因此优化需要同时知道指针连续性和执行分布，不能只看到`arange`就宣称访存已经最优。

### 7. 为什么有时必须同步

考虑下面的**依赖教学伪代码**：

```text
一组线程将A写入共享缓冲S
另一组线程读取S并计算B
```

第二步必须看见第一步对应的写入。若没有正确的同步或完成性等待，读取可能发生太早。这是RAW（read after write，写后读）依赖。若缓冲被复用，旧读未完成就覆盖它，又涉及WAR（write after read，读后写）。

这些是程序顺序依赖，不意味着看到每条load都加一次全块barrier。7.3会追存活区间、别名、分配与membar，解释哪些访问确实可能冲突。

普通经典Triton的块编程模型让编译器安排必要协作；你仍需尊重模型，不应通过内联汇编偷偷制造随lane变化的source-level标量，或者假设不同CTA会按指定顺序工作。相关维护规则见[上游AGENTS](../../triton/AGENTS.md)。

### 8. 小kernel可能被启动开销主导

FP32向量加法每个有效输出大致读两次、写一次，共12个有效字节，算一次加法。用有效工作量估算，算术强度约为`1/12 FLOP/byte`。

一个粗略分析下界是：

```text
time >= max(运算量/计算吞吐，数据量/带宽)
```

这不是实际延迟预测。小N可能主要受主机提交与设备启动开销影响；缓存会让实际DRAM流量区别于有效字节量；资源占用、依赖和调度也会限制吞吐。

矩阵乘通过复用提升运算/数据比，但shape太小仍可能没有足够并行工作。不要把一个Roofline公式当作本机速度结果。

> 回到论文：§3.3/Figure 3（PDF第4页）讨论源程序模型；§5.2.1/Figure 5和§5.2.2/Figure 6（第6页）连接分层与访存；§1/Figure 1（第1页）给出历史Roofline比较。它们不是现代GPU所有特性的实现说明，见[映射](../appendices/reference.md#sec-a-1)。

### 自测

1. **B=1024能推出thread block有1024线程吗？** 不能。
2. **grid=(3,)表示三个输入元素吗？** 不表示，它启动三份program工作。
3. **连续逻辑索引总能保证最佳合并访存吗？** 不能，还要看lane映射、对齐和实际访问。
4. **共享缓冲重用只需分配地址不重叠吗？** 不够，必须考虑不同时间的读写完成与生命周期。
5. **为何没跑GPU仍能学习本章？** 地址、依赖与职责关系可静态推演；硬件性能和具体生成布局仍需另行验证。

**从概念到仓库。** 到这里，你已经有了职责、语法、地址和执行四张基础地图。下一部分不再讲新概念，而是把这些地图落到目录：以后遇到“这个 pass 在哪”“这个 Python 名字从哪来”“旧版源码在哪里”，先按职责找，而不是从目录名猜。

<a id="sec-0-5"></a>
## 0.5 仓库地图：以后去哪里找代码

本章不要求顺读所有目录。目标是遇到一个问题时，能选择正确的层次，并用符号定位到实际实现。

### 1. 先区分三种语言角色

| 文件形式 | 在仓库里的典型作用 | 先学什么 |
|---|---|---|
| `.py` | 用户 kernel、主机运行时、前端、后端阶段组织 | 区分 host Python 与 JIT kernel |
| `.cc` / `.cpp` / `.h` | 绑定、IR、分析、变换、lowering | 类、继承、指针、namespace、visitor |
| `.td` | LLVM TableGen 描述，定义操作、属性和 pass 等 | 读名称、参数、结果与约束；它不是 C++ 源码 |
| `.mlir` | IR 示例与编译器测试 | 操作、类型、属性、SSA 值与检查行 |
| `.rst` / `.md` | 教程与设计文档 | 先区分历史说明和当前实现证据 |

`include/` 往往给出接口或声明，`lib/` 实现相应逻辑。但 Python 调度文件不只在 `python/`：硬件后端也有自己的 Python 入口。

### 2. 现代仓库按职责导航

| 位置 | 负责什么 | 第一次怎样读 |
|---|---|---|
| [README.md](../../triton/README.md) | 项目入口、构建与兼容说明 | 只记当前版本要求，不急着安装 |
| [python/tutorials](../../triton/python/tutorials) | 用户算子教程 | 从 `01-vector-add.py` 开始 |
| [python/triton/__init__.py](../../triton/python/triton/__init__.py) | 顶层导出与小工具 | 找 `cdiv` 和 `next_power_of_2` |
| [python/triton/language/core.py](../../triton/python/triton/language/core.py) | `tl` 公共 API、类型与操作入口 | 找 `program_id`、`arange`、`load`、`store` |
| [python/triton/language/semantic.py](../../triton/python/triton/language/semantic.py) | 类型、shape 与语义检查，调用 IR builder | 追一个操作，不从头读到尾 |
| [python/triton/compiler/code_generator.py](../../triton/python/triton/compiler/code_generator.py) | AST 遍历与 IR 构造 | 找 `CodeGenerator` 和 `ast_to_ttir` |
| [python/triton/runtime/jit.py](../../triton/python/triton/runtime/jit.py) | JIT 函数、参数绑定、专门化、启动主线 | 找 `KernelInterface`、`JITFunction.run` |
| [python/triton/compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) | 编译组织、缓存恢复、编译产物 | 找 `ASTSource`、`compile`、`CompiledKernel` |
| [python/triton/runtime/cache.py](../../triton/python/triton/runtime/cache.py) | 缓存管理 | 区分内存缓存与持久缓存 |
| [python/src](../../triton/python/src) | Python 到 C++ 的绑定 | `ir.cc`、`passes.cc` 连接两侧符号 |
| [include/triton/Dialect](../../triton/include/triton/Dialect) | 方言的操作、类型、属性与 pass 声明 | 先看 Triton 和 TritonGPU |
| [lib/Dialect](../../triton/lib/Dialect) | 方言实现与内部变换 | 对照头文件和 pass 注册 |
| [lib/Analysis](../../triton/lib/Analysis) | 数据流、布局、分配等分析 | 带着“这个 pass 需要哪些信息”读 |
| [lib/Conversion](../../triton/lib/Conversion) | 不同 IR 层次间的转换 | 看 TritonToTritonGPU、TritonGPUToLLVM |
| [third_party/nvidia](../../triton/third_party/nvidia) | NVIDIA 编译阶段、lowering、驱动 | 先读 `backend/compiler.py` |
| [third_party/amd](../../triton/third_party/amd) | AMD 后端 | 不直接套用 PTX/cubin 路线 |
| [third_party/proton](../../triton/third_party/proton) | profiling 相关组件 | 后续测量章节再进入 |
| [test](../../triton/test) | 编译器 IR 测试 | 学会从 `RUN` 和 `CHECK` 读预期变化 |
| [python/test](../../triton/python/test) | Python/API/kernel 等测试 | 阅读覆盖条件，不直接跑全套 |
| [unittest](../../triton/unittest) | C++ 单元测试 | 分析底层工具时查具体测试 |
| [docs](../../triton/docs) | 编程指南、API 与设计说明 | 文档帮助理解，源码确认本版本行为 |

目录大小或名字不等于重要程度。你第一次读 `lib/` 的目标不是“学完 C++”，而是追到一个已经在 Python 教程中见过的操作。

### 3. 用一个问题串起这些目录

问题：`tl.load(x_ptr + offsets, mask=mask)` 怎样成为 GPU 的读取？

```text
教程：描述要读哪些地址
  ↓
language/core.py：公开 API、参数说明
  ↓
language/semantic.py：类型与形状规则，调用 builder
  ↓
python/src/ir.cc：绑定到 C++ IR 构造
  ↓
Triton IR：保留逻辑张量、指针和 mask
  ↓
TritonGPU 变换：决定布局等实现细节
  ↓
目标后端 lowering：生成 LLVM/目标相关表达
  ↓
PTX 与二进制（本图限定 NVIDIA）
```

这是一条职责链。AST 访问期间编译器会调用这些 API 来构造 IR，不是在 GPU 执行时逐层调用上述 Python 函数。

相关操作声明可在 [TritonOps.td](../../triton/include/triton/Dialect/Triton/IR/TritonOps.td) 搜索 `LoadOp`；NVIDIA lowering 可先定位 [LoadStoreOpToLLVM.cpp](../../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp)。第一次只确认入口，不要求现在理解全部重写规则。

### 4. 先会用三个搜索动作

以下命令从两个仓库与手册所在的共同父目录执行，只读，不需要安装 Triton。

```bash
rg --files triton/python | rg 'jit.py|compiler.py|semantic.py'

rg -n 'class JITFunction|def run|def __getitem__' triton/python/triton/runtime/jit.py

rg -n 'LoadOp|create_load' triton/include/triton/Dialect/Triton triton/python/src
```

`rg -n` 给行号，`rg --files` 给文件名；`|` 把前一命令的输出交给后一命令。没有 `rg` 的环境可使用编辑器的文件/符号搜索，不是阅读依赖。

行号只是定位辅助。正文以文件相对链接和符号名为主，因为源码更新后行号容易漂移。GitHub 等渲染器中的 `#L42` 也不保证在本地 Markdown 阅读器中生效，所以不用它承担主要导航。

### 5. 读一个函数时画出“输入—输出”

例如第 4、5 章会遇到 `compile`。先写下：

```text
输入：源程序包装、目标设备、编译选项
输出：可用于后续装载与执行的 CompiledKernel
关键副作用：可能读写缓存、调用编译工具
下一跳：backend.add_stages / src.make_ir
```

然后看条件分支：缓存命中是否提前返回？哪条分支才真正编译？函数有没有提交 GPU 工作？这样才不会把函数名当成行为证明。

再看数据对象：变量是源码字符串、AST、IR module，还是设备句柄？同一个名字 `module` 在不同层可能指 MLIR module、LLVM module 或驱动模块，必须结合类型和上下文。

### 6. 历史源码要单独用一张地图

[triton-legacy](../../triton-legacy/README.md) 的官方 v0.1 快照保留旧架构。它晚于论文发表，准确版本信息见 [历史记录](../appendices/legacy-version.json)。

| 历史位置 | 作用 | 对应阅读问题 |
|---|---|---|
| [lib/runtime/function.cc](../../triton-legacy/lib/runtime/function.cc) | 源码到 IR，再到二进制的总入口 | 谁按顺序调用旧 pass？ |
| [lib/lang/parser.cc](../../triton-legacy/lib/lang/parser.cc) | Triton-C 解析 | token 怎样形成 AST？ |
| [lib/lang/code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) | AST 到自有 IR | 指针加法和数值加法如何区分？ |
| [include/triton/ir](../../triton-legacy/include/triton/ir) | 旧类型、值、指令与 builder | tile 在类型系统中如何表示？ |
| [lib/codegen/analysis](../../triton-legacy/lib/codegen/analysis) | 布局、存活期、分配等分析 | 后端需要哪些全局信息？ |
| [lib/codegen/transform](../../triton-legacy/lib/codegen/transform) | peephole、coalesce、membar 等 | 改写什么，为什么正确？ |
| [lib/codegen/selection/generator.cc](../../triton-legacy/lib/codegen/selection/generator.cc) | 旧 IR 到 LLVM | 一个逻辑操作如何展开？ |

这张表提供学习入口，不声明每项都有一个一一对应的现代类。旧 `ir::module` 与 MLIR 的 module 名字相似，但类型和 API 都不同。

> **回到论文**：[Figure 2、§3、§4、§5](../appendices/reference.md#sec-a-1) 给出旧语言、IR 和编译器的分工。把它作为职责地图，再分别打开两代源码确认实现。

### 7. 新仓库里的 Gluon 先怎样处理

[experimental/gluon](../../triton/python/triton/experimental/gluon) 与 [Gluon 教程](../../triton/python/tutorials/gluon) 是现代仓库的重要路线，手册后面会讲。它允许更显式地控制布局等细节。

目前先沿经典 `tl` 建立逻辑 tile 与 lowering 的主线；看到 Gluon 的显式 layout 时，先标记“另一种前端/编程接口”，不要把它的约束套回所有 `tl` 例子。

同样，`third_party` 不意味着“可以全部跳过”：当前 NVIDIA 主后端就在这里。判断代码是否属于主线，要看实际调用关系。

### 自测与参考答案

**要理解一个 public API 的参数语义，先去哪？** `language/core.py` 与 `semantic.py`，再按需要到 IR 与 lowering。

**要查今天 NVIDIA 编译阶段的顺序，先去哪？** `third_party/nvidia/backend/compiler.py` 的 `add_stages` 和各个 `make_*`。

**要找论文时代的 C parser，先去哪？** `triton-legacy/lib/lang/parser.cc`。

**为什么先读测试常常有帮助？** 小测试会明确输入模式和预期变换，但测试覆盖范围不能当作全部语义；仍需读定义与实现。

**现在应该读哪个 kernel？** 下一章只贯穿向量加法，不先扩展到矩阵乘；等这条路径走通，再进入 Softmax 和 GEMM。

<a id="sec-0-6"></a>
## 0.6 读代码时反复问的五个问题

这一部分不引入新源码，而是把整章压缩成五个可以反复使用的检查问题。后面每读一段 kernel、IR 或 pass，都可以先按顺序问一遍。

### 1. 每个输入的 shape、dtype、stride 和 device 是什么？

这四个属性决定“有多少元素、每个元素怎样解释、相邻元素隔多远、数据在哪”。只看变量名或数学公式不够。例如 `x + y` 只有在 shape 可广播、dtype 有定义、device 一致时才有明确语义。

### 2. 当前 program 负责哪些输出？不同 program 会不会写同一地址？

`pid`、grid 和 `offsets` 共同回答这个问题。若两个 program 写同一位置，要么是合法的覆盖，要么需要同步或原子操作；不能因为“每个 program 都执行同一段代码”就假设它们互不干扰。

### 3. 哪些值编译时已知，哪些要运行时读取？

`BLOCK_SIZE: tl.constexpr`、宏、布局参数通常在编译时决定程序形状；普通输入张量的元素值在设备执行时才读取。把这两类信息混在一起，就会误以为“换一组输入值一定要重新编译”。

### 4. 哪些元素可能越界？mask 是否保护了真正的内存操作？

mask 必须交给 `load`、`store` 或对应的内存操作本身。先读取、再用 `where` 选择结果，不能撤销已经发生的越界访问。还要区分“地址落在整份分配内”和“逻辑坐标有效”：二者不是一回事。

### 5. 当前解释来自论文、源码，还是我们自己的推断？

论文讲的是作者当时的设计与实验；源码是固定提交的真实实现；纸笔推演只能证明教学例子的数学关系。三者不能互相冒充。遇到“为什么这样写”“是否更快”这类问题，先确认证据属于哪一层。

### 自测

1. 看到 `tl.load` 就能说“现在从显存读了数据”吗？不能，要先看它是在描述设备程序，还是已经执行。
2. `BLOCK_SIZE=1024` 表示 1024 个硬件线程吗？不表示，它是逻辑元素数，线程分布由布局和后端决定。
3. shape 合法就说明地址一定对吗？不一定，还要看 stride、视图起点和 mask 的逻辑坐标。
4. 一个 kernel 只执行一次，是否意味着每个输入元素只读一次？不意味着，源码可能多次 load，缓存与优化也会改变实际流量。

## 第 0 章小结

| 你现在应该能区分 | 一句话检查 |
|---|---|
| 模型 / 算子 / kernel | 模型组合层，算子规定数学结果，kernel 是设备程序 |
| host / GPU | host 准备参数、编译和提交；GPU 执行已编译程序 |
| shape / stride / 地址 | shape 是有多少，stride 是怎样走，地址是具体位置 |
| 逻辑 tile / 硬件线程 | tile 是逻辑元素集合，一个 tile 可由多个线程协作持有 |
| 编译时 / 运行时 | constexpr 和类型参与编译，输入元素值在执行时读取 |

下一章从 2019 论文时代的 Triton-C 开始，沿源码字符串、parser 和 AST 走一遍历史前端。如果你想先看现代实现，也可以从第 4 章的向量加法开始。
