# 第 11 章 两代实现对照：前端、IR 与优化

[返回总目录](../README.md) · [上一章：现代扩展](10-modern-extensions.md) · [附录：论文定位、术语与练习](../appendices/reference.md)

这一章不引入新算法，而是把前面两条真实路线放在同一张桌上：

- **11.1**：同一个加法，两代前端怎样接住它；
- **11.2**：range、指针加法、浮点加法怎样在两代 IR 中构造；
- **11.3**：同一个合并访存问题，两代优化怎样分析、改写并保住消费者。

读这一章前，最好已经读过第 4–5 章（现代前端/IR）和第 1–3 章（历史路线）。如果还没读，至少先知道两边的版本：现代固定 `46eccb199bfd`，历史固定 v0.1 `b9e36331be85`。

## 本章地图

| 比较项 | 现代路线 | 历史路线 | 共同问题 |
|---|---|---|---|
| 用户程序 | `@triton.jit` + Python 源码 | Triton-C 源码字符串 | 表达块级计算 |
| 参数化 tile | `signature` / `constexprs` / `attrs` | 宏 `opt.defines` / `num_warps` | 让 tile 形状进入编译 |
| 前端树 | Python AST | 自有 parser/AST | 语法结构 → 中间表示 |
| 生成器 | `CodeGenerator` + `semantic` + MLIR builder | `Generator` + 自有 builder | 访问树、构造 IR |
| IR | MLIR TTIR/TTGIR | 自有 `ir::module` | 值、类型、使用关系 |
| 布局/优化 | TritonGPU pass、LinearLayout、ConvertLayout | align/layouts/coalesce/membar | 连续性与消费者接口 |
| 后端 | MLIR→LLVM→PTX→cubin | 旧 IR→LLVM→PTX→driver | 目标代码与启动 |

**读法：** 不要按文件名一一配对。先问“这一步的输入/输出对象是什么”，再看两边用什么机制实现。名称相同不代表实现相同；名称不同也不代表思想没有延续。

<a id="sec-11-1"></a>
## 11.1 同一个加法，两代前端怎样接住它

本章的前提是你已经分别读过第 4–5 章（现代前端/IR）与第 1 章（历史前端）。我们不按目录名配对，而是让同一个计算经过两条真实入口，明确哪种对象由谁接收。版本固定见[现代](../appendices/source-version.json)和[历史](../appendices/legacy-version.json)。

### 1. 先只要求数学合同相同

继续使用N=10、B/TILE=4、x[i]=i、y[i]=i+1。两代程序都希望有效位置得到`2*i+1`。下面摘取各自构造索引与读写的部分；**省略函数头、host包装与其他行**，不能直接单独运行。

现代来源：[01-vector-add.py](../../triton/python/tutorials/01-vector-add.py)第44–54行，删除注释：

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

历史来源：[custom-operation.rst](../../triton-legacy/docs/tutorials/custom-operation.rst)第17–26行中`__global__ void add`的主体，删除注释并省略函数头/闭括号；完整逐行带读见1.1：

```c
int pid = get_program_id(0);
int offset[TILE] = pid * TILE + 0 ... TILE;
float* pz[TILE] = z + offset;
float* px[TILE] = x + offset;
float* py[TILE] = y + offset;
bool check[TILE] = offset < N;
*?(check)pz = *?(check)px + *?(check)py;
```

现代的mask与旧check都按元素比较索引和有效长度；现代load/store是显式函数，旧例通过特殊的masked解引用表达。数学目标接近，不代表token、AST结构或IR节点一致。

还要注意参数次序：旧例签名以z输出在前，现代签名以x/y输入在前。把一边的调用实参原样塞给另一边会出错，即使变量都叫x/y。

### 2. 历史入口先处理一段C风格源码字符串

在[旧function.cc](../../triton-legacy/lib/runtime/function.cc)第270–279行，保留语句并删除两行注释：

```cpp
TokenSequence tokens;
Preprocessor cpp(&src_, true);
for(auto it: opt.defines)
  cpp.AddMacro(it.first, &it.second);
cpp.Process(tokens);
Parser parser(tokens);
parser.Parse();
auto ir = make_ir(parser);
```

按对象传递逐句走：

1. tokens是待填的token容器，不是GPU输入数组。
2. src_是runtime::function保存的源码字符串；预处理器拿到它的地址。
3. opt.defines含宏名与宏值，例如TILE的候选配置；AddMacro注册后，预处理过程才知道源码中的TILE表示什么。
4. Process填入tokens；Parser接收这些token并建立AST。
5. Parse完成后，make_ir拿parser作为参数，将它持有的AST转成旧自有IR。

在这条路径中，TILE首先以宏配置影响源码处理。输入x/y的十个数没有被文本替换进源码，它们仍由执行参数提供。

第254–268行还有PTX缓存分支；缓存命中时可能提前恢复caller，跳过本次解析。因此这不是每次调用都重复执行的固定步骤。1.2/3.2分别展开缓存与执行边界。

### 3. 现代入口包装的是JIT函数及其编译条件

在[compiler.py](../../triton/python/triton/compiler/compiler.py)第54–60行，摘录构造函数开头：

```python
def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
    self.fn = fn
    self.language = Language.TRITON
    self.ext = "ttir"
    self.name = fn.__name__
    self.signature = signature
    self.constants = dict()
```

fn来自JITFunction，指向被标记的kernel及其源码信息；signature是类型签名，不是十个输入数值；constexprs描述编译时常量；attrs提供相关参数属性。

前两次赋值保存函数并注明经典Triton语言路线；ext表明前端要提供TTIR阶段的IR，不表示原始Python源码已经是TTIR。name用于标识；signature保存到对象；constants先创建空容器，后续第61–65行将常量按参数路径填入。

例如BLOCK_SIZE=4会进入编译条件。它不是旧Preprocessor里注册的C宏，也不是所有时候都作为普通运行时标量传给GPU。名字相似的“参数化tile”在这里用了不同前端机制。

同一文件第78–81行原样摘录：

```python
def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
    from .code_generator import ast_to_ttir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                       module_map=module_map)
```

这里相对import明确连接到code_generator。self.fn作为fn，包装对象self作为src交给ast_to_ttir；签名和常量因此有明确来源。返回值是IR module，直接返还编译调用方。

### 4. 两边的visitor究竟消费什么

旧[make_ir](../../triton-legacy/lib/runtime/function.cc)第191–196行：

```cpp
std::unique_ptr<ir::module> function::make_ir(Parser& parser) {
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  return std::unique_ptr<ir::module>(module);
}
```

先创建旧IR容器，再让Generator访问parser里的AST并填入容器，最后返回独占所有权对象。没有Python AST参与这段历史流程。

现代[ast_to_ttir](../../triton/python/triton/compiler/code_generator.py)第1782–1785行：

```python
generator.visit(fn.parse())
module = generator.module
module.context = context
```

generator已经由前面的CodeGenerator构造语句创建，带有原型、捕获作用域和编译选项。fn.parse给出Python AST，visit生成现代MLIR内容；随后从generator取出module，并保留其IR context生命周期。

两边都可概括为“访问AST生成IR”，但输入树不同、builder不同、IR模块类型不同。这就是概念可以迁移、代码不能互换的具体原因。

### 5. 把每个接口的合同对齐

| 接口问题 | 2020历史快照 | 现代快照 |
|---|---|---|
| 用户程序输入 | Triton-C源码字符串 | JIT函数与可解析Python源码 |
| tile参数如何进入 | opt.defines等编译选项 | signature/constexprs/attrs |
| 解析基础 | 自有预处理、scanner、parser | Python AST，再按受支持语义遍历 |
| 生成器访问谁 | parser持有的旧AST节点 | fn.parse产生的Python AST |
| 返回什么 | 自有ir::module | MLIR module |
| 普通张量内容是否在此读取 | 不应这样理解 | 不应这样理解 |

因此比较时应先问“该参数在什么时候已知”“当前对象是哪一层”，不能只问哪个文件叫parser。

### 6. 一个有用的反例：交换输入值但不改类型

将x内容从0到9改成全1，N、dtype、B等条件不变。数学结果改变，但前端表达式和类型关系可能不变。不能据此断言必须重新生成一个包含十个1的程序。

若B从4改为8，两边都可能改变逻辑tile形状与编译决策。旧端通过宏/选项走不同配置，现代端通过constexpr等专门化条件区分。两者都不能与普通输入数值更新混为一谈。

> 回到论文：§3.1–§3.3（PDF第3–4页）是原语言与模型；Figure 2（第2页）是旧系统组织。Python AST与MLIR不是该论文前端的原样实现，具体范围见[映射](../appendices/reference.md#sec-a-1)。

### 自测

1. **ASTSource.signature是实际输入值数组吗？** 不是，它描述参数类型等编译信息。
2. **旧TILE宏与现代constexpr只是拼写不同吗？** 不是，二者进入不同前端处理机制。
3. **两边都用visitor，AST和IR便相同吗？** 不相同，visitor是一种组织方法，不指定表示格式。
4. **本章比较过哪一边的性能？** 都没有，只比较已固定源码中的对象和接口。

**从入口进入 IR。** 11.1 比较的是“参数和源码怎样进入编译器”；11.2 接着看进入之后的三站：整数 range、指针加法、浮点加法。三站在两代里都产生值和使用关系，但类型系统、builder 和 IR 文本完全不同。

<a id="sec-11-2"></a>
## 11.2 跟着 range、指针加法和浮点加法比较两代 IR

前端输入已经不同，本章继续跟踪同一个小表达式如何产生IR。先读[5.1/5.2](05-frontend-ir.md#sec-5-1)与[2.1](02-legacy-ir-passes.md#sec-2-1)更容易理解。以下都是静态源码与教学对象表，没有编译运行产生的dump。

### 1. 为什么不直接把旧 IR 文本换个后缀

两代都要表达“一组整数”“一组指针”“一组浮点值”，但旧版使用自己的C++ type/value/instruction，现代经典路线使用MLIR操作与类型。

把`.ll`、`.ttir`当可任意互换的文件名，会忽略解析器、类型系统和操作定义。我们比较的是同一语义任务的实现接口，不声称存在逐字翻译器。

### 2. 第一站：0到3的整数range

旧源码写`0 ... TILE`，TILE=4。进入[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc)第65–70行：

```cpp
case Token::ELLIPSIS: {
  auto clhs = dynamic_cast<ir::constant_int*>(lhs);
  auto crhs = dynamic_cast<ir::constant_int*>(rhs);
  if(!clhs || !crhs)
    error_not_implemented("ellipsis between variables not implemented");
  return set_ret(bld_->insert(ir::make_range::create(clhs, crhs)));
```

这段截取了switch分支主体，闭括号在下一行。lhs/rhs是前面递归访问AST获得的IR值；dynamic_cast要求它们真是编译器的整数常量。若不是，就进入“不支持变量端点”的错误处理。

最后从常量端点创建旧make_range指令，再通过builder插入当前基本块，set_ret把结果保存为本次AST访问返回值。**得到的是表示整数序列的IR对象，不是在这里为GPU输入分配一个四元素数组。**

现代入口写`tl.arange(0,BLOCK_SIZE)`。先经过[core.py](../../triton/python/triton/language/core.py)第1890–1893行：

```python
def arange(start, end, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    return _semantic.arange(start, end)
```

start/end先从可能的constexpr包装中取出；_semantic由编译器前端传入，携带语义检查和builder。返回值因此来自下一层语义方法，不是调用Python标准库range。

进入[semantic.py](../../triton/python/triton/language/semantic.py)第630–637行：

```python
range = end - start
if (range & (range - 1)) != 0:
    raise ValueError("arange's range must be a power of 2")
shape = [range]
if ret_ty is None:
    ret_ty = tl.block_type(tl.int32, shape)
ret_ty_ir = ret_ty.to_ir(self.builder)
return self.make_tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)
```

这里变量range只是长度，不是Python的range类型。前面第624–629行还检查端点是整数、在支持范围内且end>start；不能只保留这一小段就认为完整合法性规则只有二次幂。

本例长度4通过按位检查；shape=[4]；构造元素为int32的逻辑块类型；将类型转成IR类型，再让builder创建make_range，最后用make_tensor把IR handle与逻辑类型一起包装。

再往下，在[ir.cc](../../triton/python/src/ir.cc)第1288–1291行找到绑定：

```cpp
.def("create_make_range",
     [](TritonOpBuilder &self, Type retTy, int start, int end) -> Value {
       return self.create<MakeRangeOp>(retTy, start, end);
     })
```

`.def`注册Python可调用名字；lambda接收Python层交来的IR类型与端点；模板`create<MakeRangeOp>`构造现代操作并返回Value。Python方法名、C++操作类与IR打印名字是三层名字，不一定相同。

### 3. 第二站：给指针加上这些索引

旧[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc)第72–79行：

```cpp
case '+':
  if(binary->lhs_->Type()->ScalarType()->ToPointer()){
    return set_ret(bld_->create_gep(lhs, {rhs}));
  }
  else if(flt)
    return set_ret(bld_->create_fadd(lhs, rhs));
  else
    return set_ret(bld_->create_add(lhs, rhs));
```

binary是AST二元表达式。先看其左操作数标量类型是不是指针，是则创建GEP；否则依flt区分浮点加法与整数加法。

`{rhs}`是初始化一个索引参数集合，不表示把rhs解引用。GEP负责描述地址运算，load才负责读取内容。接下来的[旧builder.cc](../../triton-legacy/lib/ir/builder.cc)第195–196行：

```cpp
value* builder::create_gep(value *ptr, const std::vector<value*>& idx_list, const std::string &name){
  return insert(getelementptr_inst::create(ptr, idx_list), name);
```

先构造getelementptr_inst，再insert放入当前IR容器并给可选名字；返回IR值给前端。后续uses可以依赖它，地址在设备执行时才按生成代码求值。

现代同类语义在[semantic.py](../../triton/python/triton/language/semantic.py)第279行将指针加法交给`create_addptr`。它对应的[ir.cc](../../triton/python/src/ir.cc)第1493–1496行是：

```cpp
.def("create_addptr",
     [](TritonOpBuilder &self, Value &ptr, Value &offset) -> Value {
       return self.create<AddPtrOp>(ptr.getType(), ptr, offset);
     })
```

ptr和offset已经是MLIR值；结果类型取ptr的类型，操作明确保留“指针加偏移”语义。现代TTIR能在后续继续分析它，不能因为最终还会去LLVM，就把它说成此刻已是LLVM GEP。

### 4. 第三站：相同的加号用于浮点值

旧分支中的`create_fadd(lhs,rhs)`与现代[ir.cc](../../triton/python/src/ir.cc)第1379–1382行相对照：

```cpp
.def("create_fadd",
     [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
       return self.create<arith::AddFOp>(lhs, rhs);
     })
```

现代浮点加法在这里落为arith方言的AddFOp，不是所有操作都必须叫`tt.*`。MLIR module可以同时容纳多个方言；Triton的逻辑张量语义与通用算术操作协作。

一个program的x/y逻辑值均为shape=(4,) FP32，AddFOp表达对应位置相加。它与第二站地址运算的操作数类型不同，即使源程序都用`+`。

### 5. 把三站组成一个def-use链

下面是**教学对象流图，不是可解析IR文本**：

```text
range值r ───────→ 加上program块起点 → offsets
                                       ↓
基指针p ───────────────────────────→ 指针加法 → ptrs
                                                    ↓
边界mask ───────────────────────────────────────→ load → x
另一输入同样生成load → y ──────────────────────────────┘
x、y → 浮点add → result → masked store
```

每个操作生成的结果被后续操作使用，叫def-use关系。两代都要维护这种关系，优化才能替换一个结果并找到所有消费者；但它们的Value/User类与替换API并非同一份实现。

当源码写`x=x+y`，SSA中不会简单把同一个IR值的内容覆盖。编译器可以创建新值，再将后续源名字绑定到新值；控制流合并则需相应结构。旧phi不能直接解释为论文psi，现代region与block argument也需按其定义理解。

### 6. 形状、类型与布局分别在哪

| 信息 | 旧快照 | 现代经典路径 |
|---|---|---|
| 四个int32元素 | 旧tile类型与make_range指令 | tl.block_type包装与MLIR张量类型 |
| 指针加法 | getelementptr_inst | TTIR AddPtrOp，后续再降低 |
| 浮点加法 | 自有binary instruction | arith::AddFOp |
| 线程/存储分布 | 后续analysis::layouts等 | TTGIR编码属性、LinearLayout等 |
| 最终LLVM表示 | 旧selection generator构造 | 多个MLIR lowering再转LLVM module |

因此“旧IR基于LLVM思想”不能推成它就是LLVM支持所有工具可直接解析的格式，更不能推成旧IR与现代TTIR相同。接口证据比名称联想可靠。

> 回到论文：§4.1–§4.2（PDF第4–5页）组织结构、tile类型与操作；§4.3的PSSA方案需与所选快照差异一起读。本章只核实上面真实构造路径，未声称找齐论文每个特殊节点，见[论文索引](../appendices/reference.md#sec-a-1)。

### 自测

1. **make_tensor会把GPU数据复制到CPU吗？** 这里不是，它包装IR handle与逻辑类型。
2. **现代create_addptr为什么不能直接称最终LLVM GEP？** 它创建的是当前方言的AddPtrOp，后续才降低。
3. **`arith::AddFOp`不带Triton名字，是否不属于编译主线？** 它是主线使用的通用方言操作。
4. **同一shape便意味着相同lane映射吗？** 不意味着，layout是另一层实现信息。

**从表示进入改写。** 11.2 证明了两代 IR 都能表达同一组 range/指针/浮点加法，但表示不同。11.3 选一个具体优化问题——合并访存——看两边怎样从分析找到目标、怎样改写，以及为什么必须保住消费者的接口。

<a id="sec-11-3"></a>
## 11.3 同一个访存问题，两代优化怎样分析和改写

本章只选“合并访存”贯穿旧新优化。目标不是比较两个文件名，而是回答：分析读了什么、变换修改什么、消费者为什么还能收到正确值。版本固定，未运行编译或性能实验。

### 1. 要解决的问题不是把数学数组换一组值

假设一个逻辑块内有同一组待加载元素，不同执行分布会让相邻线程访问不同地址组合。选择适合内存访问的分布，有机会减少浪费或提高向量化机会；计算结果仍须对应原逻辑坐标。

具体收益由地址、对齐、参与mask、目标指令和硬件共同决定。不能只看到pass名字叫coalesce就断言一定减少某个固定数目的事务。

两代都需要知道“指针沿哪些轴连续”，又都必须处理变换后其他消费者期待的分布。这是共同问题；实际分析和重写机制分别如下。

### 2. 旧版先从uses里找真正的内存访问

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

输入v来自当前旧布局分析关联的值集合。遍历其users，是从一个值反查哪些指令使用它；dynamic_cast筛出I/O指令；指针相等判断进一步确认它确实用v作为地址，而不是存入内存的数值。

result按引用传入，所以insert直接改变调用方的集合。函数没有修改设备数据，也没有在这个步骤选择新布局。

接下来的第27–32行：

```cpp
void coalesce::extract_ld(ir::io_inst* i, std::map<int, std::vector<ir::io_inst*>>& result) {
  ir::value *ptr = i->get_pointer_operand();
  auto contiguous = align_->contiguous(ptr);
  auto it = std::max_element(contiguous.begin(), contiguous.end());
  int axis = std::distance(contiguous.begin(), it);
  result[axis].push_back(i);
```

这里截去结尾的闭括号。先取地址操作数，再通过已有align分析查询各轴连续性信息；找最大项及其轴编号；最后按axis把I/O指令归组。

`align_`不是从天而降，在构造函数第15–16行由调用方提供。旧[make_bin](../../triton-legacy/lib/runtime/function.cc)第217行创建`coalesce(&align,&layouts)`；第227–230行先运行相关分析，第231行再运行coalesce。因此要读构造参数和pipeline，才能知道这个成员里的信息何时有效。

### 3. 旧版选中之后怎样连接新的值

[旧coalesce.cc](../../triton-legacy/lib/codegen/transform/coalesce.cc)第129–135行，保留原语句：

```cpp
auto& inst_list = r->get_parent()->get_inst_list();
auto pos = ++std::find(inst_list.begin(), inst_list.end(), r);
builder.set_insert_point(pos);
if(dynamic_cast<ir::load_inst*>(r)){
  ir::instruction *cts = builder.insert(ir::copy_to_shared_inst::create(r));
  r->replace_all_uses_with(cts);
  cts->replace_uses_of_with(cts, r);
```

这是run函数中“已选中待处理I/O”的后段，省略外层循环和闭括号；不能把它当所有load无条件运行的规则。

先找到r所属基本块中的位置，把builder插入点设到r后面。若r是load，就创建以r结果为输入的copy_to_shared。此时还只是添加IR指令，实际共享内存分配和同步在后续分析中安排。

`replace_all_uses_with`让原先消费r的其他位置改消费cts。为什么还需最后一行？因为新cts自己也用r；一并替换后会错误地让cts引用自己，所以要把cts内部那个自引用恢复为r。

这段解释的是历史变换的意图，不是对旧容器实现运行安全性的测试。2.2/3.1保留了相关实现边界；本书没有构建验证该版本。

旧版还包含rematerialize逻辑，会递归克隆某些操作或保留已有共享数据。它不等于把所有数学索引强制变连续；选中范围来自布局组、连续轴分组和真实操作类型。

### 4. 现代版先为每个内存操作计算目标编码

[现代Coalesce.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)第77–86行：

```cpp
void runOnOperation() override {
  // Run axis info analysis
  ModuleOp moduleOp = getOperation();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  CoalescedEncodingAnalysis coalescedEncodingAnalysis(axisInfoAnalysis);

  // For each i/o operation, we determine what layout
  // the pointers should have for best memory coalescing
  llvm::MapVector<Operation *, Attribute> layoutMap;
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
```

runOnOperation是pass框架进入该变换的方法，getOperation取得当前模块。先创建轴信息分析，再让coalescedEncodingAnalysis依赖它；layoutMap准备记录“操作→目标编码”，warp大小从模块属性读取，不在本段硬编码所有后端为32。

后面第87–105行遍历module：取得内存指针，排除不是指针张量的情形，获取numWarps、CTA相关布局和shape，再调用getEncoding。筛选不通过只跳出当前回调，不是终止整个pass。

这里将“分析决定哪个编码”与“用该编码改写IR”分开。第117–118行进入实际重写：

```cpp
for (auto &kv : layoutMap) {
  convertDistributedOpEncoding(kv.second, kv.first);
```

kv的first是旧操作，second是目标编码。不要把这两个参数次序反过来：函数需要先知道encoding，再知道要改哪个op。

### 5. 现代重写如何保住原消费者的接口

跟到[Utility.cpp](../../triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp)第1325–1337行，摘取函数开头，删除注释：

```cpp
Operation *convertDistributedOpEncoding(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  SmallVector<Value, 4> newArgs;
  for (auto &opOperand : op->getOpOperands()) {
    Value operand = opOperand.get();
    if (isDistributedOpEncodingOperand(opOperand)) {
      Type newType = getNewType(operand.getType(), encoding);
      newArgs.push_back(triton::gpu::ConvertLayoutOp::create(
          builder, op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
```

这仍是有意截断的短段。builder在旧op处创建新操作；遍历每个操作数；需要分布式编码转换的操作数先得到新类型，再插入ConvertLayoutOp；无需转换的操作数原样放入newArgs。

`SmallVector<Value,4>`里的4是小容量存储优化，不是“本操作必须有4个张量元素”。编译器容器参数和用户tile形状属于不同层次。

函数接着处理结果类型，在第1348–1349行创建同名的新操作：

```cpp
Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                  newArgs, newTypes, op->getAttrs());
```

它保留原位置、操作名字与属性，使用新的操作数和结果类型；不是仅给旧op打一个“现在更快”的标签。

第1352–1360行将新结果恢复到旧消费者期望的布局，再接上users：

```cpp
for (size_t i = 0; i < op->getNumResults(); i++) {
  Value newResult = newOp->getResult(i);
  if (newTypes[i] != op->getResultTypes()[i]) {
    newResult = triton::gpu::ConvertLayoutOp::create(
        builder, op->getLoc(), op->getResult(i).getType(), newResult);
  }
  op->getResult(i).replaceAllUsesWith(newResult);
}
op->erase();
```

循环对每个结果处理；若新旧结果类型不同，插一个转换回旧类型的操作；再让旧结果的所有users消费这个新值。最后旧op已不再是必要计算来源，才erase。

因此现代思路可以画成**教学对象图**：

```text
旧消费者期望L1

原来：输入L1 → 内存操作L1 → 消费者L1
改写：输入L1 → convert到L2 → 内存操作L2 → convert回L1 → 消费者L1
```

这张图解释语义接口，不预测转换最后使用shuffle、共享内存还是可被消除；后续lowering和优化才决定实际代价。没有转换回去的需要时，也不意味着一定有物理数据搬运。

图示覆盖普通分布式load/store的典型路径；真实实现还单独选择descriptor布局，并对哪些操作数要转换、异步操作的结果类型作例外处理，不能把图当作所有内存操作的无条件模板。

### 6. 哪些知识能迁移，哪些不能

| 比较项 | 两边共同的问题 | 实现不能直接等同之处 |
|---|---|---|
| 连续性 | 哪个轴适合组织访问？ | 旧align/layout组与现代AxisInfo/编码分析不是同一算法 |
| 重写输入 | 消费者能否使用新结果？ | 旧共享copy/rematerialize与现代ConvertLayout构造不同 |
| 数据依赖 | 替换值后users怎样改接？ | C++自有user容器与MLIR use机制不同 |
| 资源与同步 | 新中间存储何时读写？ | 现代还有descriptor/async/warp-specialized等额外语义 |
| 性能 | 访存组织是否更适合目标？ | 名称相同不保证事务数、寄存器数或延迟相同 |

读完这一条真实路径后，其他机制用相同问题检查：先确认输入条件，再看分析依赖，再看改写和消费者，最后检查资源/同步与验证。不要把这个步骤机械套成性能结论。

> 回到论文：§5.2.2/Figure 6（PDF第6页）给出合并访存问题；§5.2.3–5.2.4讨论存储与同步。当前代码分别来自2020与2026快照，均不能当作论文算法逐行保存的证明，见[论文索引](../appendices/reference.md#sec-a-1)。

### 自测

1. **旧extract_io_use为什么检查pointer_operand？** v可能也作为被存数值使用，只筛I/O类型仍不够。
2. **现代layoutMap保存已经搬好的数据吗？** 不，它保存编译期操作与编码属性的对应。
3. **为什么新操作后可能还要convert回旧编码？** 维持已有消费者的类型与布局接口。
4. **看见convert_layout可以直接数出共享内存往返吗？** 不行，需继续看优化和lowering。
5. **这两份实现谁更快？** 本章没有执行，不能由代码阅读作出性能排名。

## 第 11 章小结

| 层次 | 可以迁移的理解 | 必须重新学习的实现 |
|---|---|---|
| 前端 | 都是“解析源码 → 访问树 → 构造 IR” | Python AST vs 自有 C 前端；宏 vs constexpr |
| IR | 都用值、类型、使用关系表达数据流 | MLIR TTIR/TTGIR vs 自有 `ir::module` |
| 优化 | 都先分析连续性/消费者，再改写并改接 uses | AxisInfo/ConvertLayout vs align/layouts/coalesce |
| 后端 | 都要降低到 LLVM/目标代码并装载发射 | 阶段名、产物格式、驱动路径不同 |

### 综合检查

1. 旧 TILE 宏和现代 `constexpr` 只是拼写不同吗？不是；它们进入不同前端机制，但都在编译时影响 tile 形状。
2. 旧 `getelementptr_inst` 和现代 `AddPtrOp` 都是“指针加偏移”，可以逐字互换吗？不能；类型系统、builder 和后续 lowering 不同。
3. 两代 `replace_all_uses_with` 的行为一样吗？概念目标相似，但容器、API 和 IR 使用关系不同；不能直接套用。
4. 为什么比较时必须先限定版本？因为论文、2020 v0.1 和现代 main 是三个不同证据层。
5. 读完对照章，最应该带走什么？不是“谁更快”，而是“同一问题在不同表示下需要哪些对象和合同”。
