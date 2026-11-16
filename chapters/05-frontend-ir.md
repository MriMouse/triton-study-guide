# 第 5 章 前端与第一份 IR：Python 源码怎样变成编译器对象

[返回总目录](../README.md) · [上一章：向量加法与 JIT](04-vector-add-jit.md) · [下一章：Softmax 与 GEMM](06-classic-kernels.md)

第 4 章停在“缓存未命中时构造 `ASTSource`，compile 返回产物”。但编译器到底读到了什么？这一章打开中间过程：

- **5.1** 追 `ast_to_ttir`：Python AST 怎样经过 `CodeGenerator`、`semantic` 和 C++ builder，构造出第一份 IR；
- **5.2** 换一种读法：拿上游测试里的真实 IR，逐行识别 module、function、SSA 值、类型、操作和数据流，再进入 TTIR→TTGIR 的阶段。

这两部分合在一章，是因为 5.1 交出“编译器构造出来的 module”，5.2 立刻问“这个 module 怎样读”。如果拆成两章，读者会在最需要看到对象长相的时候停一次。

## 本章地图

| 部分 | 输入 | 输出 | 读完后能回答 |
|---|---|---|---|
| 5.1 | `ASTSource`、签名、常量、Python AST | 初始 TTIR module | `output = x + y` 怎样变成一条浮点加法操作 |
| 5.2 | 上游测试的真实 IR、TTGIR 片段 | 对 IR 数据流和布局的读法 | 谁定义值、谁使用值、shape/类型/布局写在哪 |
| 5.3 | 实际编译的 add kernel | TTIR/TTGIR/LLIR/PTX 片段 | 真实编译产物和上游测试文本有什么对应关系 |

**读法：** 始终区分三种“x”：host 的 `torch.Tensor`、编译期的 `tl.tensor` 包装、IR 里的 SSA 值。5.1 造出第三种，5.2 教你怎么读它。

<a id="sec-5-1"></a>
## 5.1 编译器怎样读懂 x + y：AST、语义对象与 builder

本部分接住 4.2 的 ASTSource：它保存向量加法的 JITFunction、参数类型、B=4 的编译常量和参数属性。目标是解释 `output = x + y` 怎样变成 IR 中的浮点加法操作。x/y 仍对应 N=10 的输入，但**编译期间不取出它们的十个浮点数来相加**。

固定源码提交 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。5.1–5.2 为静态核对；5.3 节给出从该提交构建后的实际编译与 IR 导出。代码摘录许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. make_ir 收到的五个输入是什么

在 [compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) 第300–307行，compile 准备上下文与后端辅助对象，再调用来源对象：

```python
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)

codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
try:
    module = src.make_ir(target, options, codegen_fns, module_map, context)
```

摘录保留 try 开头，异常处理省略。context 可以先理解为 IR 类型、操作等对象所属的管理环境；load_dialects 注册编译器会用到的操作体系。“dialect，方言”在这里是一组有命名空间的 IR 类型和操作，例如 tt 的加载、arith 的加法。

src 就是前章创建的 ASTSource。target 描述目标；options 是已解析的编译选项；codegen_fns 是后端提供的辅助代码生成函数；module_map 供前端映射模块；context 是这次生成 IR 的上下文。它们都不是设备张量。

同文件 `ASTSource.make_ir` 第78–81行原样摘录：

```python
def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
    from .code_generator import ast_to_ttir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                       module_map=module_map)
```

这里 self 是 ASTSource。`self.fn` 作为第一个实参，对应 ast_to_ttir 的 fn；self 本身作为第二个实参，对应 src。函数返回的 module 原样交回 compile 的 module 变量。target 没有直接传给 ast_to_ttir，目标影响已通过后端选项和辅助对象进入，不应凭函数名字想象额外参数。

### 2. 从字符串类型到前端类型

[compiler/code_generator.py](../../triton/python/triton/compiler/code_generator.py) 第1745–1750行：

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    arg_types = [None] * len(fn.arg_names)
    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = str_to_ty(v, None)
```

fn.arg_names 是 4.2 中按定义顺序保存的五个参数名。src.signature 是“名字→类型描述”。循环把每个 k 找回对应位置 idx，再用 str_to_ty 把字符串描述转为前端类型对象，保存在 arg_types。

本例三个输入/输出地址的类型描述是 FP32 指针，n_elements 对应整数类型；B 的 constexpr 标记随后还会被实际常量的类型补齐。`None` 初始化只是在创建五个空槽位，不是说设备参数的值为 None。

第1768–1771行继续：

```python
for path, value in src.constants.items():
    apply_constexpr_types(arg_types, list(path)[::-1], value)

prototype = ASTFunction([], arg_types, src.attrs)
```

path 是常量参数位置，本例 `(4,)`；value 为4。apply_constexpr_types 在第1752–1766行把对应参数改成携带编译常量信息的类型。`[::-1]` 反转路径列表，配合递归 pop 访问；本例只有一个位置，不必猜测复杂嵌套。

ASTFunction 在这里是前端使用的函数原型对象：输入类型来自 arg_types，属性来自 src.attrs，开始的空列表用于返回类型。它不是 Python 标准库的 AST FunctionDef 节点。

**确认题：** 这里知道 B=4，是否知道 x[0]=0？前者知道，后者不知道。输入数值属于设备执行时的数据；指针类型与常量值属于当前编译条件。

### 3. 语法树是什么，谁访问它

同文件第1778–1790行，调整跨行构造参数的排版并删除注释后作教学整理：

```python
generator = CodeGenerator(context, prototype, gscope=fn.get_capture_scope(),
                          function_name=fn.repr(proxy), jit_fn=fn,
                          is_kernel=True, file_name=fn.file_name,
                          begin_line=fn.def_file_line_number,
                          begin_col=fn.def_file_col_number,
                          options=options, codegen_fns=codegen_fns,
                          module_map=module_map, module=module, is_gluon=fn.is_gluon())
generator.visit(fn.parse())
module = generator.module
module.context = context
if not module.verify():
    if not fn.is_gluon():
        print(module)
    raise RuntimeError("error encountered during parsing")
return module
```

proxy 由紧邻的第1773–1777行构造，装着 constants 与 signature，供函数命名表示使用。gscope 接收函数能捕获到的名字，例如 tl；prototype 仍是上一节创建的对象。generator 是一个 CodeGenerator 实例，而不是 GPU 的工作队列。

fn.parse 的实现位于 [runtime/jit.py](../../triton/python/triton/runtime/jit.py) 第564–568行：

```python
def parse(self):
    tree = ast.parse(self._src)
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef)
    return tree
```

此时 self 是 JITFunction，`_src` 是函数源码文本。ast.parse 返回 Python 抽象语法树：树中表示“这里是赋值，右侧是加法，左右分别是名字 x/y”，而不是存着 x/y 的设备数值。

`CodeGenerator(ast.NodeVisitor)` 的继承声明在 code_generator.py 第294行。visitor 是按节点种类分派的访问方式：visit 遇到 BinOp 节点就找 visit_BinOp，遇到 Name 就找 visit_Name。可以理解为“统一入口根据对象类别选择具体处理函数”，不是另一个 GPU 线程。

verify 检查的是生成 module 的结构与类型等 IR 约束。通过它不表示输出已经与 PyTorch 比较，也不表示算法数值正确。

### 4. builder 与 semantic 从同一个 generator 中产生

[code_generator.py](../../triton/python/triton/compiler/code_generator.py) 第299–308、322行，不连续摘录经典 tl 分支：

```python
self.context = context
self.is_gluon = is_gluon
from triton.language.semantic import TritonSemantic
self.builder = ir.builder(context)
self.semantic = TritonSemantic(self.builder)
self.module = self.builder.create_module() if module is None else module
```

此处 self 是 CodeGenerator。builder 是构造 IR 操作的工具，持有“接下来插到哪里”的状态；semantic 负责检查运算是否合法、转换类型、广播 shape，再调用同一个 builder。module 是保存 IR 函数的容器。

[language/semantic.py](../../triton/python/triton/language/semantic.py) 第34–38行把这段关系写得很直接：

```python
def __init__(self, builder):
    self.builder = builder

def make_tensor(self, handle, type) -> TensorTy:
    return self.tensor(handle, type)
```

make_tensor 把 IR value 的 handle 与前端类型包装起来；这不是分配一个新的 torch.Tensor。

[language/core.py](../../triton/python/triton/language/core.py) 第905–916行的 tensor 构造器保存 handle、shape、type、dtype。关键第908–915行：

```python
self.handle = handle
self.shape = type.shape if type.is_block() else ()
self.numel = constexpr(math.prod(self.shape))
self.type = type
self.dtype = type.scalar
```

这里的 `tl.tensor` 是编译期间的语言值包装：handle 指向 IR 中某个值，shape 描述逻辑块，dtype 是元素类型。它不像 torch.Tensor 那样在 host 代表一份已分配的设备存储。编译器可以创建一个“未来 load 的四个结果”的 tl.tensor，此刻尚未读取任何输入。

### 5. kernel 参数怎样进入函数体的名字表

`visit_FunctionDef` 的真实流程见 [code_generator.py](../../triton/python/triton/compiler/code_generator.py) 第676–692行；下面删去与本例无关的 caller_context 和位置命名行：

```python
visibility = "public" if self.is_kernel else "private"
fn_ty = self.prototype.serialize(self.builder)
self.fn = self.builder.get_or_insert_function(self.module, self.function_name, fn_ty, visibility, self.noinline)
self.module.push_back(self.fn)
entry = self.fn.add_entry_block()
arg_values = self.prototype.deserialize(self.fn)
for arg_name, arg_value in zip(arg_names, arg_values):
    self.set_value(arg_name, arg_value)
self.builder.set_insertion_point_to_start(entry)
self.visit_compound_statement(node.body)
```

fn_ty 是把前端原型转为 IR 函数类型的结果。self.fn 是 IR 函数对象；与 ast_to_ttir 的 fn（JITFunction）不是同一类对象。entry 是函数入口基本块，即一段依次执行、由终结操作结束的 IR 操作序列。

deserialize 在同文件第263–285行从 IR 函数参数 handle 重建前端值包装，同时处理编译常量。zip 配对名字与值，因此名字 `x_ptr` 可以在正文中解析到对应的参数包装；B 对应已知常量。

set_value 的第459–460行将关系写入两个字典：

```python
self.lscope[name] = value
self.local_defs[name] = value
```

lscope 是本地名字表。builder 的插入点移到 entry 开头后，函数体中每次构造操作就会进入这个函数的基本块。没有这一步，知道要造加法还不足以知道加法属于哪一段程序。

### 6. 走到 output = x + y，左右的 x/y 来自前面的 load

4.1 的 kernel 先有 `x = tl.load(...)`、`y = tl.load(...)`，再有 `output = x + y`。前两个调用被访问时生成 load 操作，其返回的前端 tl.tensor 按名字保存；本节集中追最后这个表达式，不重复整个 load 实现。

为了避免重名混淆：host 的 x 是 torch.Tensor；kernel 源码里的 x 是本 program 加载所得逻辑块；编译器名字表里 x 是代表该逻辑块的 tl.tensor。B=4 时，其前端块 shape 是 `(4,)`、元素 FP32。即使尾块只有两个有效输入，类型的长度仍为4；mask 不会把类型变成长度2。

`visit_Assign` 在 [code_generator.py](../../triton/python/triton/compiler/code_generator.py) 第789–798行先访问 node.value，再 assignTarget；右侧不完成，左侧 output 不会获得新值。其核心次序可概括成教学伪代码：

```text
访问“x+y”节点 → 得到一个前端结果值 → 将名字output绑定到该值
```

这不是执行设备 store；只有随后 `tl.store(output_ptr + offsets, output, mask)` 才描述内存写回。

### 7. visit_BinOp 递归得到两个操作数

同文件第882–889行，连续摘录方法与操作表开头：

```python
def visit_BinOp(self, node):
    lhs = self.visit(node.left)
    rhs = self.visit(node.right)
    operator = self._get_operator(node, node.op, "binary")
    return self._apply_binary_method(node, operator, lhs, rhs)

_operators: Dict[Type[ast.AST], Tuple[str, str, str]] = {
    ast.Add: ('__add__', '__radd__', '+'),
```

操作表只展示第一项，后续项与结束花括号有意截断；这个代码块不是可直接执行的完整类或函数文件。

lhs/rhs 不是 Python 浮点0和1，而是递归访问 Name 节点的结果。第818–821行：

```python
def visit_Name(self, node):
    if type(node.ctx) is ast.Store:
        return node.id
    return self.dereference_name(node.id)
```

读取 x/y 时走 dereference_name，查询前面建立的作用域；写左值时才直接给名字。`ast.Store` 在此是 AST 的“名字被写入”上下文标签，与 `tl.store` 的设备内存写入不是一个操作。

操作表告诉前端：加法对应正向 `__add__` 和反向 `__radd__`。第858–878行按优先级选择 owner 与方法；普通 tl.tensor 左右相加时，可以调用 lhs 的 `__add__(rhs)`。这保留了自定义语言值的运算规则，并非把所有加号都强行当整数加法。

### 8. 为什么 __add__ 收得到 _semantic

[core.py](../../triton/python/triton/language/core.py) 第928–930行：

```python
@builtin
def __add__(self, other, _semantic=None):
    return add(self, other, sanitize_overflow=True, _semantic=_semantic)
```

这里 self 已经是左侧 tl.tensor，other 接右侧 tl.tensor。builtin 标记表明这是编译器识别的语言内建。第2912–2915行自由函数 add 继续转交：

```python
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.add(x, y, sanitize_overflow)
```

_semantic 不是用户在 kernel 里手写的额外参数。前端 `_call_operator` 第841–847行将找到的方法交给 call_Function；后者在第1496–1511行识别 builtin/语言对象方法，检查签名是否含 `_semantic`，有则添加当前 generator 的 semantic：

```python
if '_semantic' in sig.parameters:
    extra_kwargs["_semantic"] = self.semantic
ret = fn(*args, **extra_kwargs, **kws)
```

这是一处重要的文件衔接：CodeGenerator 负责调用，core.py 提供语言表面函数，semantic.py 接收同一组前端值并进行类型/shape 检查。用户不必传 builder，但编译器通过 semantic 保持了同一个构造上下文。

### 9. semantic 先决定运算种类，再造 IR

[semantic.py](../../triton/python/triton/language/semantic.py) 第259–287行包含指针、浮点、整数三个分支。以下是不连续短摘录：

```python
input, other = self.binary_op_type_checking_impl(input, other, True, True)
input_scalar_ty = input.type.scalar
other_scalar_ty = other.type.scalar
elif input_scalar_ty.is_floating():
    return self.make_tensor(self.builder.create_fadd(input.handle, other.handle), input.type)
```

这段省略了 if 主分支，展示用的 elif 不是独立可执行代码。`binary_op_type_checking_impl` 第185–201行先判断元素类型、转换成共同计算类型，再调用 broadcast_impl_value 让 shape 相容。

在本例中两者都是 FP32、shape `(4,)`，不需要扩成更大块，结果仍为 FP32 的四元素块。若一个操作数是 shape `()` 的标量、另一个是 `(4,)`，广播使标量逻辑上用于四个位置；这不等于复制四份输入张量到显存。

为什么同一个加号在 4.1 中能用于指针？`x_ptr + offsets` 的元素类型含指针，因此 semantic 走 create_addptr；`x + y` 的元素类型为浮点，走 create_fadd。类型不仅供注释，也决定生成哪种操作。

### 10. Python 方法怎样落到 C++ 的 IR 构造

[python/src/ir.cc](../../triton/python/src/ir.cc) 第1379–1382行是实际绑定：

```cpp
.def("create_fadd",
     [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
       return self.create<arith::AddFOp>(lhs, rhs);
     })
```

`.def` 给 Python builder 暴露名为 create_fadd 的方法。lambda 中 self 是 C++ TritonOpBuilder 的引用；lhs/rhs 接收两个 IR value，来自刚才 input.handle/other.handle。`<arith::AddFOp>` 指定要创建浮点加法这一类操作；返回 Value 是该操作的结果值。

结果经 semantic.make_tensor 包装后，沿返回路径回到 core.add、tensor.__add__、visit_BinOp、visit_Assign。最终名字 output 指向这个新结果包装。这里“一条语句变成操作”的小闭环才完成。

编译器可以在 host 上构造“读取四个数并相加”的指令，而不具备四个数本身。这与编辑菜谱时写下“把两碗相加”，并未开始做菜一样；设备执行才消费实际输入。

### 11. 现在我们交出了什么

generator.visit 完成后，ast_to_ttir 返回 module。它包含函数、基本块、参数、load/add/store 等 IR 操作。compile 中的 module 接到它，后续交给编译阶段函数继续变换。

**自测：** 编译器访问 `output = x + y` 时，哪个对象保存源码结构，哪个对象保存 IR 结果引用？AST 的 BinOp/Name 节点保存源码结构；tl.tensor.handle 引用已构造的 IR value。哪个对象决定插在哪里？builder 的插入点。哪一步写入 output 的设备存储？设备执行后续 store 时，不是 set_value 更新名字表时。

> 回到论文：§3.2.2 Broadcasting（PDF第3–4页）说明成块操作为何需要广播；§4.1 Structure（PDF第4–5页）帮助理解函数与SSA结构。对应关系见[论文定位表](../appendices/reference.md#sec-a-1)。本章 Python AST visitor 与 MLIR builder 属于现代实现，不能把它们说成原论文 Triton-C parser 或旧 Triton-IR。

下一部分拿实际上游测试里的短 IR 逐行读；因为本次没有真实编译 4.1，不会声称某段文本就是 N=10、B=4 在本机生成的 dump。

**第一部分交出的边界。** 5.1 结束时，`ast_to_ttir` 返回一个包含函数、基本块、参数、load/add/store 等操作的 module。它已经是 IR，但还没有带 GPU 布局。5.2 先教你读这份逻辑 IR，再看 TTGIR 怎样加入 layout 编码。

<a id="sec-5-2"></a>
## 5.2 第一次读 IR：名字、类型、操作与数据流

5.1 说明 builder 怎样构造浮点加法。本章把构造出来的东西当作一种可读的程序：谁定义值、谁使用值、shape 写在哪里、哪些信息到 TTGIR 才出现。读完应能手工追踪一段 range→地址→load→add 的数据流，而不是只背 TTIR/TTGIR 两个缩写。

依据仍为提交 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`，代码许可见 [CODE-NOTICE](../CODE-NOTICE.md)。**5.2 的真实文本来自上游测试输入，不是本机生成的 4.1 dump；5.3 节另给出本机生成的 TTIR/TTGIR/LLIR/PTX 片段。** 测试摘录、推演与教学改写各自标注；未运行 triton-opt/lit。

### 1. 先读一句，不急着读完整文件

在 [test/Triton/vecadd.mlir](../../triton/test/Triton/vecadd.mlir) 第28行有这句真实测试输入：

```mlir
%22 = arith.addf %19, %21 : tensor<256xf32>
```

可以按四段拆开：

| 文字 | 含义 |
|---|---|
| `%22 =` | 给这次操作产生的结果取一个 SSA 名字 |
| `arith.addf` | arith 方言的浮点逐元素加法操作 |
| `%19, %21` | 两个已经存在的输入值 |
| `tensor<256xf32>` | 逻辑块有256个元素，每个元素FP32 |

`%22` 不是“第22号物理寄存器”，`256` 也不是线程数。IR 文本里的名字允许打印器重新编号，只要定义与引用对应，数字本身没有数学意义。

SSA 是 static single assignment，静态单赋值。最初可以理解为：每个这种结果名字只有一个定义，使用该名字就能找到唯一来源。`%22` 的值来自这一次 addf；下一次加法要用新名字。控制流合并和循环会通过块参数等结构表达，后面再看一个真实例子。

### 2. module 和 function 把操作装在哪里

同一测试文件第3–8行原样摘录：

```mlir
module {
  tt.func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
```

module 是顶层容器；大括号中有一个 tt.func。`@...` 是函数符号名，用于识别这个函数，不是某个数据地址。函数圆括号内声明参数：前三个是 FP32 元素指针，后三个是32位整数。

这份测试带循环，还有六个参数；**它并非 4.1 的五参数 Python kernel 的直接转写**。本章先读共同的索引/访存语义，再单独解释其循环，不靠文件名推断来源。

`!tt.ptr<f32>` 是 Triton 方言的指针类型，所指元素为 f32；`i32` 是32位整数。感叹号表示方言定义的类型写法，不是逻辑“非”。每个 `%argN` 是函数参数值，进入函数时由调用方提供，因此不需要另一个操作定义它。

在函数入口的这段顺序操作位于同一个基本块；没有分支时，你可以先顺着文字向下读。模块不等于 GPU 内存，函数也不等于当前某一个 program 的运行实例：同一份函数代码会被不同 program 使用。

### 3. 属性与操作数有什么不同

看 make_range：

```mlir
%2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
```

花括号里的 start/end 是操作属性，编译器处理这条操作时就知道它们；因此这条 range 描述固定的0到255。它不需要 GPU 从某个参数寄存器读取“end值”。冒号后的 `i32` 标明属性数值的类型。

对比 `%1 = arith.muli %0, %c256_i32`：两个 `%...` 是操作数引用，指向 IR 值。尽管其中一个来源是 constant 操作，语法角色仍是“这个操作的输入值”。属性与常量操作都可能携带常量信息，但结构不同，不能看到数字就当作同一表示。

`tt.get_program_id x` 中 x 表示网格的 x 轴。假设执行 program 2，则 `%0=2`；编译时这条操作仍表示“运行时取得本 program 编号”，不是编译器只生成 program 2 的专用函数。

### 4. 从标量到逻辑块：为什么多一条 splat

第9–12行：

```mlir
%3 = tt.splat %1 : i32 -> tensor<256xi32>
%4 = arith.addi %3, %2 : tensor<256xi32>
%5 = tt.splat %arg3 : i32 -> tensor<256xi32>
%6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
```

前面的 `%1` 是一个标量 block_start；`%2` 是256个范围值。splat 把同一个标量逻辑上扩展到256个位置，使两个加法输入 shape 匹配。`->` 在这条操作类型说明里隔开输入和输出类型，不是程序跳转箭头。

`arith.addi` 是整数逐元素加法，得到 offsets。把 N 也 splat 后，`cmpi slt` 表示带符号整数小于比较，得到与 offsets 同形状的布尔块。其结果元素类型是 i1，虽然这条自定义打印语法在末尾写的是操作数的 tensor<256xi32>。

重要的是不能机械地把每个操作冒号后文本都当成“输出类型”。不同操作有不同简写格式，需结合操作语义、显式箭头或定义来读。

**纸笔推演：** 令 program_id=0、arg3=10，则 `%1=0`，`%4=[0,1,...,255]`，`%6` 前十项为 true，其余 false。这里没有实际执行，只是在给操作假定输入后解释其含义。

### 5. 指针块是地址集合，load 结果才是数值块

第13–18行原样摘录：

```mlir
%7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%cst = arith.constant 0.000000e+00 : f32
%11 = tt.splat %cst : f32 -> tensor<256xf32>
```

`%7` 是重复的基指针，`%8` 才加上每个元素的偏移。指针按所指元素类型计步；FP32 的第 i 项地址比首项多 `4*i` 字节，不是多 i 字节。

`tensor<256x!tt.ptr<f32>>` 是256个逻辑地址，`tensor<256xf32>` 是256个逻辑浮点值；两种类型不能混用。`%11` 的零块准备给后面的循环累加初值；不是把整个输出显存清零。

现在手工画一段 def-use 关系：

```text
%0(program id) ─┐
%c256_i32 ─────┴→ %1 → %3 ─┐
%2(range) ─────────────────┴→ %4(offsets) ─┬→ %6(mask)
%arg0 → %7(base pointers) ──────────────────┴→ %8(input pointers)
```

def 是定义，use 是引用这个定义。比如 `%4` 被比较和地址计算共同使用。若优化改变 `%4` 的定义或替换它，必须让所有使用仍指向语义等价的值；不能只修改图上的一条边。

### 6. 读真实循环时，先认出谁在迭代之间传递

这个测试第19–21行还引入循环，不要把它忽略后声称已读完整函数。原样摘录：

```mlir
%c0_i32 = arith.constant 0 : i32
%c32_i32 = arith.constant 32 : i32
%15:3 = scf.for %arg6 = %c0_i32 to %arg4 step %c32_i32 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) : i32 {
```

scf 是结构化控制流方言。`%arg6` 是循环计数，从0到 arg4、每次加32，右界不含在内。`iter_args` 声明三条循环携带值：累加值 arg7、两个当前输入指针块 arg8/arg9。第一次从 `%11/%8/%10` 取初值，之后从上一轮 yield 接收。

`%15:3` 表示循环有三个结果，后面可以用 `%15#0`、`%15#1`、`%15#2` 分别引用。不是把同一个名字赋值三次。循环体中的块参数名字表示“本轮收到的值”，所以SSA仍能表达每一轮不同的内容。

第22–34行的关键操作如下，省去两组零常量与splat定义行（%18和%20都是256项FP32零块），**非连续摘录**：

```mlir
%19 = tt.load %arg8, %6, %18 : tensor<256x!tt.ptr<f32>>
%21 = tt.load %arg9, %6, %20 : tensor<256x!tt.ptr<f32>>
%22 = arith.addf %19, %21 : tensor<256xf32>
%23 = arith.addf %arg7, %22 : tensor<256xf32>
%24 = tt.splat %arg5 : i32 -> tensor<256xi32>
%25 = tt.addptr %arg8, %24 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%26 = tt.splat %arg5 : i32 -> tensor<256xi32>
%27 = tt.addptr %arg9, %26 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
scf.yield %23, %25, %27 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
```

两次 load 的第二个操作数是 mask，第三个操作数 `%18/%20` 是前面构造的零块：无效位置用零。接着先求 x+y，再加到上轮累加值 arg7。最后两个指针按 arg5 个元素前进，yield 按顺序把三个值交给下一轮；最后一轮结束时，它们成为循环三个结果。

如果假设 arg4=32，则循环只跑计数0这一轮；零累加初值加上 x+y，结果就是一次加法。arg4 若更大，就可能读取后续指针位置，不再是 4.1 的算法。指针是否合法还依赖调用方合同，IR 的结构合法性不会自动证明它们指向足够大内存。

### 7. 写回为什么没有等号

第36–39行原样摘录：

```mlir
%16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%17 = tt.addptr %16, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
tt.store %17, %15#0, %6 : tensor<256x!tt.ptr<f32>>
tt.return
```

store 接地址、循环最终累加块、mask。它产生的是内存副作用，没有供后续表达式使用的 SSA 结果，所以前面没有 `%新值 =`。tt.return 结束函数；这份 kernel 的计算结果通过内存写出，不通过函数返回一个张量值。

因此不能只删去“结果没人使用”的 store；SSA use 不是判断副作用是否必要的唯一标准。后面的 pass 章会用具体规则解释安全改写。

### 8. 回到 N=10、B=4：只作明确的教学替换

上面的真实测试是256元素并含循环。为了连接 4.1，可以在纸上把索引段中的256统一替成4，再不考虑这个测试额外的循环。**这是教学推演，不是有效完整 dump，也不是对上游文件的修改。**

对 program2：

| 中间结果 | 纸上内容 |
|---|---|
| block_start | `2*4=8` |
| range | `[0,1,2,3]` |
| offsets | `[8,9,10,11]` |
| mask，N=10 | `[true,true,false,false]` |
| x有效读值 | `[8,9,?,?]` |
| y有效读值 | `[9,10,?,?]` |
| 有效加法结果 | `[17,19,?,?]` |

这里 `?` 仍遵循 4.1 原 load 没有指定 other 的边界；不能因为本章测试显式补零，就把 4.1 尾块也说成零。两份程序的区别需要在文本中保留。

**确认题：** 把所有 `%4` 改成 `%offsets` 会改变结果吗？只要合法地同时改定义和全部引用就不会；改变 range 的 end、指针类型或 mask 则可能改变语义。

### 9. TTIR 与 TTGIR：增加布局信息，但仍然是 IR

前面读的逻辑 tensor 类型尚未带具体线程布局。另看真实的 TTGIR 测试 [test/TritonGPU/combine.mlir](../../triton/test/TritonGPU/combine.mlir) 第3–4、23、27–33行，不连续摘录并删除 CHECK 注释：

```mlir
#layout0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = ttg.convert_layout %cst : tensor<1024xi32, #layout0> -> tensor<1024xi32, #layout1>
  tt.return %1: tensor<1024xi32, #layout1>
}
```

这是较大文件里的一个函数片段，外层 module 的结束括号不在摘录范围。`#layout0` 是布局属性别名，`#` 与 SSA 值 `%` 不同。tensor 类型现在除了1024和i32，还带有布局。module 的属性还记录 warp/CTA 数量；不是程序调用时传入的浮点数据。

sizePerThread、threadsPerWarp、warpsPerCTA 描述不同层级的分布；order 给出维度顺序。先知道“同一个逻辑 tensor 可以有不同分布”即可，如何逐个映射到线程寄存器由 7.1 专门手算。

convert_layout 的前后逻辑形状与元素类型相同；改变的是表示分布。它不等于把矩阵转置，也不能一概认为只改标签、必定零开销。这个测试使用全零常量，恰好可以由优化直接在目标布局产生常量；源码 CHECK 检查消除转换，但那是测试的期望，本章没有运行该测试验证。

### 10. 初始 module 如何进入编译阶段

[compiler/compiler.py](../../triton/python/triton/compiler/compiler.py) 第290–292行注册阶段：

```python
stages = dict()
backend.add_stages(stages, options, src.language)
first_stage = list(stages.keys()).index(src.ext)
```

stages 是按插入顺序组织的“阶段名→函数”字典。ASTSource.ext 是 ttir，因此从这条路线对应位置开始。NVIDIA 的经典 tl 注册见 [third_party/nvidia/backend/compiler.py](../../triton/third_party/nvidia/backend/compiler.py) 第661–665行：

```python
def add_stages(self, stages, options, language):
    capability = self._parse_arch(options.arch)
    if language == Language.TRITON:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
```

lambda 把当前 options/capability 保存起来，将来阶段调用时只需再给 module 和 metadata。这里局部形参 src 代表输入 IR，与外层 compile 中 ASTSource 的名字 src 相同但对象层次不同。

第292–305行的 make_ttir 构造 pass manager，依序添加内联、规范化、combine、广播重排、公共子表达式消除等，然后 `pm.run(mod, 'make_ttir')`。因此前端刚生成的 module 与最终保存的 `.ttir` 可以不同：名为 ttir 的阶段还会优化初始 IR。

第316行将逻辑 Triton IR 转为带GPU布局的表示：

```python
passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
```

这是此处核对的 NVIDIA 路径，32是传给该转换的每warp线程数；不要把它原样套到所有后端。

外层 compile 在 compiler.py 第326–327、350行推动流水线，非连续摘录：

```python
for ext, compile_ir in list(stages.items())[first_stage:]:
    next_module = compile_ir(module, metadata)
    module = next_module
```

上一阶段返回值交给 next_module，再成为下一阶段的 module。后面可能是其他形式的文本或二进制，不能永远把这个 Python 局部名字当成 MLIR module 类型。7.2 细读一个 pass，8.1 再完成后端产物链。

### 11. 自测与论文回读

**问题一：** `%8` 代表某个输入浮点数吗？不，它是逻辑指针块；load 的结果才是数值块。**问题二：** 编译器创建 tt.store 时是否已经修改 output？没有；这是记录未来执行的写入。**问题三：** tensor 有1024项能否直接推断1024个线程？不能，布局决定元素怎样分配，同一线程可能持有多项。

> 回到论文：§4 The Triton IR 与 Listing 5（PDF第4–5页，Listing在第4页），§4.1 Structure，§4.2 Data-Flow/Figure 4（PDF第5页），可对照函数、SSA与显式tile数据流；见[定位表](../appendices/reference.md#sec-a-1)。论文中的旧 Triton-IR 不是这里的 MLIR TTIR，打印语法与类型系统不能混读。

本章已经能读出操作之间的数据依赖；5.3 节给出实际编译产物，但仍没有测得性能。接下来先用这些概念读一个真正包含归约的算法：稳定 softmax。

## 5.3 实际运行：同一 kernel 的 IR 与目标代码

`actual_ir_dump.py` 编译 4.1 的 add kernel，把 TTIR/TTGIR/LLIR/PTX 写到 `/research/runs/book-actual/stages`，并打印关键行。完整文件不在仓库内；下面只摘足够对照的短段。

**命令。**

```bash
python examples/actual_ir_dump.py
```

**输出节选。**

```text
--- ttir: 50 lines, 3113 bytes ---
7: tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> ..., %y_ptr: !tt.ptr<f32> ..., %out_ptr: !tt.ptr<f32> ..., %n_elements: i32 ...) {
11: %offsets = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
18: %x_5 = tt.load %x_4, %mask_3 : tensor<4x!tt.ptr<f32>>
21: %y_7 = tt.load %y_6, %mask_3 : tensor<4x!tt.ptr<f32>>
22: %out = arith.addf %x_5, %y_7 : tensor<4xf32>
25: tt.store %1, %out, %mask_3 : tensor<4x!tt.ptr<f32>>
--- ttgir: 52 lines, 3552 bytes ---
1: #blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
19: %x_4 = tt.load %x_3, %mask_2 : tensor<4x!tt.ptr<f32>, #blocked>
22: %y_6 = tt.load %y_5, %mask_2 : tensor<4x!tt.ptr<f32>, #blocked>
26: tt.store %1, %out, %mask_2 : tensor<4x!tt.ptr<f32>, #blocked>
--- llir: 65 lines, 3978 bytes ---
6: define ptx_kernel void @add_kernel(ptr addrspace(1) align 16 %0, ...)
14: %14 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %13
18: %18 = fadd float %15, %17
--- ptx: 182 lines, 5077 bytes ---
.visible .entry add_kernel(
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
add.rn.f32 	%r3, %r1, %r2;
@%p2 st.global.b32 [ %rd3 + 0 ], { %r3 };
n_regs 12
n_spills 0
shared_bytes 0
num_warps 4
```

**怎样读。**

- TTIR 保留了逻辑块：`tensor<4xi32>`、`tensor<4xf32>`、`tt.load`、`arith.addf`、`tt.store`。`tt.make_range` 的 `end=4` 来自 `BLOCK_SIZE=4`。
- TTGIR 的 tensor 类型多了 `#blocked` 编码。逻辑 shape 和操作没变，但编译器已经开始决定这些元素怎样分布到 lane/warp/register。
- LLIR 里出现 `getelementptr`、`fadd`；PTX 里出现 `ld.global.b32`、`add.rn.f32`、`st.global.b32`。这些是同一程序在更接近目标机器时的表示。
- 这些阶段不是 GPU 把同一计算执行四遍；它们是编译流水线里依次细化的对象。

**边界。** 这里展示的是一次具体编译的产物，不是所有 shape 的固定布局；完整的 TTIR/TTGIR/LLIR/PTX 文本保存在 `/research/runs/book-actual/stages`，仓库只保留生成脚本。

## 第 5 章小结

| 问题 | 这一章给出的答案 |
|---|---|
| `output = x + y` 在编译期做了什么？ | AST 访问得到两个 `tl.tensor`，`semantic` 检查类型/shape，builder 创建 `arith.addf` |
| `x`/`y` 的十个数值参与编译了吗？ | 没有；编译期只知道类型、shape 和常量条件 |
| IR 里的 `%22` 是物理寄存器吗？ | 不是，是可重命名的 SSA 结果名 |
| `tensor<256xf32>` 的 256 是线程数吗？ | 不是，是逻辑元素数 |
| TTIR 和 TTGIR 的区别是什么？ | TTGIR 在逻辑 tensor 类型上增加了 layout 编码 |
| `convert_layout` 一定搬运数据吗？ | 不一定；可能被消除、只重排线程内值，或需要 shuffle/共享内存，需看 lowering |

### 综合检查

1. 为什么 `x + y` 和 `x_ptr + offsets` 都写 `+`，却生成不同操作？因为操作数的元素类型不同：浮点走 `create_fadd`，指针走 `create_addptr`。
2. `visit_BinOp` 为什么要递归访问左右孩子？它要先把两个操作数从名字表/子表达式还原成前端值，才能调用 `semantic`。
3. 真实测试 IR 和 4.1 的 kernel 是同一份 dump 吗？不是；测试带循环和六个参数，本章只借用它讲共同的索引/访存语义。
4. `module.verify()` 通过能证明数值正确吗？不能，它检查 IR 结构与类型约束，不比较 PyTorch 结果。
