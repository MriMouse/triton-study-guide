# 第 2 章 历史路线（二）：旧 IR 与局部优化

[返回总目录](../README.md) · [上一章：Triton-C 与 parser](01-legacy-frontend.md) · [下一章：旧布局与后端](03-legacy-layout-backend.md)

第 1 章把 AST 交给 `function::make_ir`。这一章从那里继续：

- **2.1**：追 AST 怎样变成旧 `ir::module`——module/function/basic_block、tile 类型、value/user 图、builder 插入、masked load 和 phi；
- **2.2**：追 `make_bin` 里真实的 pass 顺序和一次 `reduce → reshape` 改写，再单列论文的 prefetch/PSSA 思想与快照差异。

这一章的核心不是“记住旧 IR 的类名”，而是理解**对象在编译期怎样被创建、插入、使用和替换**。2.2 的改写会用到 2.1 建立的 `value/user` 图；如果 2.1 没读清，2.2 的 `replace_all_uses_with` 就只是几个字符串。

## 本章地图

```text
AST
  → 2.1：Generator::Gen
      → ir::module
          → function → basic_block → instruction
          → value(user) 图：operand / use
          → masked_load / GEP / fadd / range / phi
  → 2.2：make_bin
      → dce → disassociate → peephole → align/cts/axes/layouts
      → coalesce → 清理 → 重算布局 → liveness → allocation → membar
      → 一次 reduce → reshape 的真实改写
      → 论文 prefetch / PSSA 思想（与快照差异）
```

| 部分 | 核心问题 | 读完后能回答 |
|---|---|---|
| 2.1 | AST 怎样变成旧 IR？ | 一条 GEP 的基指针和 offset 怎样连到 value/user 图 |
| 2.2 | pass 怎样改写旧 IR？ | 匹配条件、插入点、替换 uses、DCE 清理怎样配合 |

**读法：** 每读一个旧 IR 构造，先画三个框：产生的结果值、它的操作数、它的使用者。2.2 的所有改写都在移动这三者之间的边。

<a id="sec-2-1"></a>
## 2.1 沿一个指针加法走进旧 Triton-IR

在本章里，IR 是编译器内存里的程序表示。一个 builder 方法会造出一个值或一条操作，并将它放进程序的适当位置。例如 `create_gep` 记录「基址加元素偏移」的规则；真正的张量地址在设备运行时才会参与计算。

1.2 交来 Parser，其中 unit_ 指向 AST 根。
本章沿 `0 ... TILE`、`x + offset` 和 masked load，解释树怎样变成可分析的旧 IR 对象。
继续使用 TILE=4；纸笔可取 pid=2，offset=[8,9,10,11]。
编译器此刻不知道某次 program 的实际 pid，也不读取 x[8]；它构造计算这些值的指令。

源码固定为 2020-05-04 v0.1、`b9e36331be854e37aa14c538861689533ec0dfb0`。
这不是现代 MLIR TTIR，也未证明与 2019 论文实验精确一致。
以下只有静态带读；示意 IR 不是实得 dump。许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. 模块由谁创建，交给谁

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 191–196 行：

```cpp
std::unique_ptr<ir::module> function::make_ir(Parser& parser) {
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  return std::unique_ptr<ir::module>(module);
}
```

parser 是调用者仍存活的 Parser 引用。
`new ir::module` 创建一份旧 IR 容器；空字符串是模块名，ctx_ 提供 IR 类型等共享上下文。
module 是局部裸指针，指向刚创建的对象。
`Generator gen(&parser)` 在栈上创建 AST→IR 生成器，借用 parser 地址。
`gen.Gen(module)` 通过指针修改该模块，没有返回一串设备计算结果。
最后 unique_ptr 接管模块独占所有权，并返回给 function::make 中的 `auto ir`。

`ir::` 是命名空间限定；`std::unique_ptr<T>` 是管理 T 指针生命周期的 C++ 模板类型。
它与 1.2 的裸 AST 指针不同，不能把两者都说成自动所有权树。

### 2. Generator 借来同一个 builder

[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) 第 506–513 行：

```cpp
void Generator::Gen(ir::module *mod) {
  mod_ = mod;
  ctx_ = &mod_->get_context();
  bld_ = &mod_->get_builder();
  assign_ = new LValAssigner(this);
  VisitTranslationUnit(parser_->Unit());
  delete assign_;
  assign_ = nullptr;
}
```

mod 形参接收上一节 module 指针；mod_ 保存它，两个名字指向同一对象。
ctx_ 和 bld_ 分别借用模块的 context、builder 地址。
builder 是“向指定位置加入 IR 指令”的辅助对象，不是 GPU 执行器。
assign_ 处理左值赋值，例如最后对 masked pz 写入，以及局部变量定义。
this 把当前 Generator 交给 LValAssigner，供它调用同一套表达式生成逻辑。

`parser_->Unit()` 返回 AST 根，VisitTranslationUnit 顺序访问其顶层节点。
同文件第 499–503 行的循环是 `for (auto extDecl: unit->ExtDecls()) Visit(extDecl);`。
`Visit` 在 [code_gen.h](../../triton-legacy/include/triton/lang/code_gen.h) 第 55 行调用 `node->Accept(this)`。
节点再按自身种类调用 VisitFuncDef、VisitBinaryOp 等具体函数。
这叫 visitor 分派：由对象的实际类型决定下一次调用，不是把 C++ 函数名当字符串猜测。

Gen 完成后只销毁临时 assign_，已插入的 IR 留在模块中。
Generator 借用 AST 产生新 IR；两种树/图的节点不是同一对象。

### 3. 函数入口怎样接上参数和基本块

[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) 第 467–496 行。
先摘出函数创建部分：

```cpp
Stmt *body = funcDef->body_;
const std::string& name = funcDef->Name();
FuncType* type = funcDef->FuncType();
auto prototype = dynamic_cast<ir::function_type*>(GenIRType(type, *ctx_));
// 省略：prototype 非空检查
ir::function *fn = mod_->get_or_insert_function(name, prototype);
std::vector<ir::argument*> args = fn->args();
```

body 来自 1.2 的 SetBody；name 是 add；type 是 AST 层函数类型。
GenIRType 把 AST 类型转成旧 IR 类型，dynamic_cast 检查结果确实是 function_type。
fn 是模块内的 IR 函数，args 是该函数的 IR 形参对象集合。
这些 argument 代表运行时的 z/x/y/N，不包含当前 GPU 上的实参数据。

第 477–489 行依次遍历 AST 参数，把名称绑定到 args[i]。
其中两行关键语句为：

```cpp
mod_->set_value(name, nullptr, args[i]);
mod_->get_scope().types[name] = args[i]->get_type();
```

循环中这个 name 是参数名，例如 x；它遮蔽了外层函数名 name。
set_value 保存“源码名字 x 对应哪个 IR value”，nullptr 用于这项函数参数绑定。
types 同时保存该名字的 IR 类型，以便后面构造值或 phi。
不能因为局部变量重名就把参数名误读成函数名 add。

第 491–496 行建立顺序指令的容器：

```cpp
ir::basic_block *entry = ir::basic_block::create(mod_->get_context(), "entry", fn);
mod_->seal_block(entry);
mod_->get_builder().set_insert_point(entry);
VisitStmt(body);
if(!dynamic_cast<ir::return_inst*>(ret_))
  mod_->get_builder().create_ret_void();
```

entry 是 fn 内的入口基本块。基本块可先理解为从头顺序执行到分支/返回的一段 IR。
seal_block 表示前驱信息可确定；与后面 phi 的补全相关。
set_insert_point 让随后创建的指令插到 entry 中。
VisitStmt 开始生成函数体；最后必要时补 void return。
整个过程仍在建程序，尚未执行 return 或访问 x。

### 4. 一次递归调用怎样通过 ret_ 交回结果

[code_gen.cc](../../triton-legacy/lib/lang/code_gen.cc) 第 9–10 行把 value 指针写入成员 ret_。
VisitBinaryOp 第 20–34 行：

```cpp
Visit(binary->rhs_);
ir::value* rhs = ret_;
if(binary->op_ == '=')
  return set_ret(assign_->GenExpr(binary->lhs_, rhs));
Visit(binary->lhs_);
ir::value* lhs = ret_;
auto type = binary->lhs_->Type()->ScalarType();
auto flt = type->IsFloat();
auto sign = !type->IsUnsigned();
```

binary 是本次 AST 二元表达式节点，例如 `x + offset`。
先访问右孩子，递归结束时 ret_ 指向右边生成的 IR value；立即复制指针到局部 rhs。
赋值要特殊处理，因为左边是存放位置，不能总按普通读取求值。
对加法则继续访问左孩子，ret_ 被覆盖；局部 rhs 保留了此前结果。
再将新的 ret_ 保存为 lhs，取得 AST 左操作数的标量元素类型。
flt/sign 用来选择浮点、整数或有符号比较等 IR 指令。

ret_ 是临时返回槽，不是唯一 IR 节点，也不是 kernel 的返回值。
rhs 与 lhs 指向的节点已由其他容器/图关系保留；更新 ret_ 不会把前一条指令改写成后一条。
这种“递归 + 成员暂存返回值”是本实现的约定，不等于每种 visitor 都必须这样写。

### 5. 先走 range，再走 offset

VisitBinaryOp 第 65–70 行：

```cpp
case Token::ELLIPSIS: {
  auto clhs = dynamic_cast<ir::constant_int*>(lhs);
  auto crhs = dynamic_cast<ir::constant_int*>(rhs);
  if(!clhs || !crhs)
    error_not_implemented("ellipsis between variables not implemented");
  return set_ret(bld_->insert(ir::make_range::create(clhs, crhs)));
}
```

对 `0 ... 4`，右孩子常量 4 先进入 VisitConstant，左孩子常量 0 随后进入。
VisitConstant 在第 319–325 行把 AST 整数转成 `ir::constant_int`。
dynamic_cast 检查两端确实是编译期整数；变量端点在这个分支不支持。
`make_range::create` 创建 range 指令，insert 将它放进当前块，set_ret 把其 value 指针交回外层。
range 在语义上代表四个偏移，但编译器没有把 pid=2 当常量代入。

外层 `pid * TILE + range` 再生成乘法与整数加法。
广播相关节点可能由前端类型转换生成，不能把 AST 字面节点数直接当成最终 IR 指令数。
这一点也解释了为什么下面只给教学对象图，不伪造精确 dump。

### 6. 同一个加号怎样变成地址计算

VisitBinaryOp 第 72–79 行：

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

处理 `x + offset` 时，VisitObject/VisitIdentifier 通过 mod_->get_value 找到 x 的 argument。
offset 对应前面生成并绑定的 IR value。
左边元素类型为指针，所以 create_gep 的 ptr 形参收到 lhs；索引列表收到只含 rhs 的 vector。
`{rhs}` 是 C++ 初始化列表，这里组成一个索引集合，不是创建 GPU 张量。
GEP 是 get element pointer，只表示地址计算。
若是 masked load 后两个 float tile 相加，左元素是浮点，分支就变成 create_fadd。

### 7. create 与 insert 是两个连续动作

[builder.cc](../../triton-legacy/lib/ir/builder.cc) 第 195–196 行：

```cpp
value* builder::create_gep(value *ptr, const std::vector<value*>& idx_list, const std::string &name){
  return insert(getelementptr_inst::create(ptr, idx_list), name);
}
```

ptr/idx_list 分别接上节 lhs/{rhs}。
工厂函数 create 返回新 GEP 节点；insert 把它加入程序顺序，再返回同一节点指针。
[builder.h](../../triton-legacy/include/triton/ir/builder.h) 第 51–59 行：

```cpp
template<typename InstTy>
InstTy* insert(InstTy *inst, const std::string &name = ""){
  assert(block_);
  block_->get_inst_list().insert(insert_point_, inst);
  inst->set_parent(block_);
  inst->set_name(name);
  // 原文件第57–58行为已注释代码，此处省略
  return inst;
}
```

模板让同一 insert 可接收 GEP、range 等不同指令类型。
block_ 与 insert_point_ 来自前面设置的插入点。
assert 检查已有块；list.insert 建立顺序位置；set_parent 建立反向的所属块关系。
set_name 设置可读名字，不改变 GPU 数据。
这就是“指令对象已构造”和“指令已属于某基本块”的区别。

### 8. mask 的左右孩子怎样接到 load

VisitBinaryOp 第 58–63 行：

```cpp
case Token::MASKED_DEREF: {
  ir::type* ret_ty = GenIRType(binary->Type(), *ctx_);
  ir::value* false_value = ir::undef_value::get(ret_ty->get_scalar_ty());
  if(ret_ty->is_tile_ty())
    false_value = bld_->create_splat(false_value, ret_ty->get_tile_shapes());
  return set_ret(bld_->create_masked_load(rhs, lhs, false_value));
}
```

1.2 已看到该 AST 节点左边是 predicate，右边是地址。
所以 rhs 传入 load 的指针参数，lhs 传入掩码参数，不是写反了。
ret_ty 是 load 结果类型；false_value 是同元素类型的 undef，tile 时复制成同形状。
最后创建 masked load，结果指针仍由 ret_ 带回外层加法。
“创建 load”不是现在在 host 执行 load；GPU 运行生成代码时才读取有效位置。

### 9. type、value、user 的关系落到一条 GEP

[value.h](../../triton-legacy/include/triton/ir/value.h) 第 22–44 行：value 保存 ty_ 和 users_。
type 描述值的类别，例如 float 指针 tile、整数 tile；value 描述一个具体计算结果/参数。
同类型的 px 与 py 仍是不同 value，不能因为类型相等就合并。
第 51–78 行的 user 继承 value，还保存自己的操作数列表 ops_。
一个 GEP 因而既产生指针值，也使用基指针和 offset。

[instructions.cc](../../triton-legacy/lib/ir/instructions.cc) 第 369–371 行在 GEP 构造中调用：

```cpp
set_operand(0, ptr);
for(size_t i = 0; i < idx.size(); i++)
  set_operand(1 + i, idx[i]);
```

[value.cc](../../triton-legacy/lib/ir/value.cc) 第 43–46 行说明两边怎样相连：

```cpp
void user::set_operand(unsigned i, value *x) {
  assert(i < ops_.size() && "set_operand() out of range!");
  ops_[i] = x;
  x->add_use(this);
}
```

GEP 的 ops_[0] 指向基指针 x；同时 x 的 users_ 加入这个 GEP。
前向操作数关系适合从结果追来源，反向 users 适合从旧结果寻找需要重写的使用者。
2.2 用 replace_all_uses_with 改写归约时，就依赖这张图。

### 10. 先用示意值图合上向量加法

以下是**教学示意 IR**，省略广播、类型细节与实际打印语法：

```text
%r    = range(0, 4)
%pid  = program_id(0)
%base = mul(%pid, 4)
%off  = add(%base, %r)
%px   = gep(%x, %off)
%py   = gep(%y, %off)
%pz   = gep(%z, %off)
%mask = signed_less_than(%off, %N)
%a    = masked_load(%px, %mask, undef)
%b    = masked_load(%py, %mask, undef)
%sum  = fadd(%a, %b)
masked_store(%pz, %sum, %mask)
return_void
```

百分号名字用来区分定义结果；每个结果由某条指令/形参产生，后面的指令引用它。
它没有写“每次给 off 变量原地加值”，因此适合追踪定义与使用。
对 pid=2 才能纸笔算出 off=[8,9,10,11]；这不是编译时常量折叠的实得结果。

### 11. 分支汇合时 phi 从哪里来

向量加法没有标量分支汇合，无需为了讲 phi 伪造它的 dump。
另设**教学示意**：then 路径定义 t=1，else 路径定义 t=2，merge 块使用 t。
merge 中 t 取哪一个，要由实际控制流来自哪个前驱决定。

[module.cc](../../triton-legacy/lib/ir/module.cc) 第 84–92 行补齐 phi 输入：

```cpp
ir::basic_block *block = phi->get_parent();
for(ir::basic_block *pred: block->get_predecessors()){
  ir::value *value = get_value(name, pred);
  phi->add_incoming(value, pred);
}
```

name 是待合并的源码变量名；get_value(name,pred) 查询该前驱的定义。
add_incoming 同时登记 value 和 pred，不能只给两个无来源的数字。
第 96–117 行在块尚未 sealed 时先建未完整 phi，在多前驱时创建 phi 并补操作数。
第 142–149 行 seal_block 才完成未决输入并尝试消去平凡 phi。
[instructions.cc](../../triton-legacy/lib/ir/instructions.cc) 第 49–53 行还要求输入类型与 phi 类型一致。

phi 表示跨基本块的值合并；它不是按 tile 每个 lane 的 predicate 选择。
论文 PSSA/ψ 的实现映射在该快照尚未核实，不能把 phi 直接改叫 ψ。
2.2 会用明确标注的论文思想伪码解释差别。

> 回到论文：§4.1、Listing 5，PDF 第 4–5 页；§4.2、Figure 4，第 5 页；§4.3、Listing 6，第 5 页。先区分结构、tile 数据流与逐元素控制流，再对照本章的 2020 类与函数。精确位置见[论文表](../appendices/reference.md#sec-a-1)。

自测：builder 现在读了 x[8] 吗？没有。
ret_ 被覆盖会丢失已插入指令吗？不会，基本块和操作数图仍保存关系。
为何 x+offset 与 a+b 用不同指令？一个左元素是指针，另一个是浮点。
下一部分接收的是什么？一个含函数、基本块、类型和值使用关系的旧 `ir::module`。

**第一部分交出的边界。** 2.1 结束时，旧 module 里每个操作数都指向一个 value，每个 value 又记录自己的 users。这个双向关系是后面所有 pass 改写的基础。2.2 先读 `make_bin` 的真实运行顺序，再挑一个“长度为 1 的归约轴”小例，逐句看匹配、插入、替换 uses 和 DCE 清理。

<a id="sec-2-2"></a>
## 2.2 一次旧 IR 改写怎样真正发生

一次 pass 相当于用一段编译器程序处理一遍 IR。看到 `run(module)`，先看它是收集信息还是改变操作。后者如替换 `sum([[7],[9]], axis=1)`，由于每行仅一项，只需要改形状就能得到 `[7,9]`。

这一章从旧 ir::module 出发，先核对真实 pass 顺序，再亲手追一次“单元素归约换成 reshape”。
它回答的不是“有哪些优化名字”，而是匹配条件、插入点、使用关系和清理怎样连起来。
最后单列论文的 prefetch 与 PSSA 思想，避免因当前源码缺失对应实现而完全跳过机制。

固定版本为 2020-05-04 v0.1、`b9e36331be854e37aa14c538861689533ec0dfb0`。
没有构建或运行 pass；数值与 IR 都会标注为纸笔示意。
许可见 [CODE-NOTICE](../CODE-NOTICE.md)。

### 1. make_bin 接到的三个输入

[function.cc](../../triton-legacy/lib/runtime/function.cc) 第 199–205 行：

```cpp
std::unique_ptr<driver::module> function::make_bin(ir::module &module,
                                                   driver::context *context,
                                                   const options_t& opt) {
  std::unique_ptr<codegen::target> target = context->device()->make_target();
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(module.get_name(), ctx));
```

module 是 2.1 生成的旧 IR，以引用传入，后面的 pass 会直接改它。
context 来自调用流的设备上下文；它决定目标设备和资源上限。
opt 是当前候选的宏与 num_warps 配置，已经不是完整搜索集合。
target 保存设备相关代码生成规则。
ctx 与 llvm 是新建的 LLVM 层对象，此刻旧 module 与 LLVM Module 同时存在。
本地变量名字叫 llvm，并不表示旧 module 已经被改造成 LLVM 对象。

### 2. 分析对象之间为何互相传地址

同文件第 207–219 行，摘录需要的数据依赖：

```cpp
codegen::analysis::align align;
codegen::analysis::axes axes;
codegen::analysis::layouts layouts(&axes, &align, opt.num_warps);
codegen::analysis::liveness liveness(&layouts);
codegen::analysis::allocation allocation(&liveness);
codegen::transform::membar barriers(&liveness, &layouts, &allocation);
// 省略：其他无构造参数的变换对象
codegen::transform::coalesce coalesce(&align, &layouts);
// 省略：cts
codegen::generator isel(&axes, &layouts, &align, &allocation, target.get(), opt.num_warps);
```

`align align` 前面是类型、后面是变量名；C++ 容许它们同名。
`&axes` 取得局部对象地址，layouts 以后从该对象查询轴关系。
构造对象不等于已经执行分析；必须等 run(module) 填入结果。
先构造 allocation、后运行 liveness 是合法的：allocation 构造时只是保存指针。

依赖可以读成一条数据链：轴/连续性→布局→共享布局存活期→分配区间→同步判断。
isel 同样保存这些分析对象地址，以便将已决定的布局与存储位置降到 LLVM。
这些对象都在 make_bin 的栈作用域内，直到 isel 使用完毕才离开。
不要将构造顺序误写成实际 pass 执行顺序。

### 3. 将真实 run 顺序逐段读出来

下面来自同文件第 222–247 行，**连续原样摘录**：

```cpp
dce.run(module);
disassociate.run(module);
dce.run(module);
peephole.run(module);
dce.run(module);
align.run(module);
cts.run(module);
axes.run(module);
layouts.run(module);
coalesce.run(module);
dce.run(module);
align.run(module);
dce.run(module);
reassociate.run(module);
cts.run(module);
peephole.run(module);
dce.run(module);
align.run(module);
axes.run(module);
layouts.run(module);
liveness.run(module);
allocation.run(module);
if(allocation.allocated_size() > context->device()->max_shared_memory())
  throw std::runtime_error("using too much shared memory");
barriers.run(module);
isel.visit(module, *llvm);
```

第一段用 dce、disassociate、peephole 清理/改变逻辑 IR。
随后 align、cts、axes、layouts 准备与生成物理存储相关的表示和分析。
coalesce 消费第一轮布局与连续性，可能再改变 IR；所以后面还有清理与重算。
reassociate、第二轮 cts/peephole 之后，再次运行 align、axes、layouts。
最终 liveness/allocation 用这一版 IR 的共享布局计算资源。
超出设备共享内存上限会抛异常，当前候选可能编译失败。
barriers 在分配之后运行，因为它要检查实际存储区间的冲突。
isel.visit 是旧 IR→LLVM 的阶段边界，3.2 才展开。

这里没有现代 MLIR PassManager 的字符串管线。
同一个 C++ 对象可 run 多次；重复出现有明确位置，不能精简成“每个 pass 各运行一次”。
也不能把第 222–247 行硬拆成完全独立的机器无关/机器相关两个列表，然后忘记二者在实现中穿插。
论文分组是解释职责的方式，源码顺序才是本快照真正做事的顺序。

### 4. 为何用一个小归约作带读

向量加法没有归约，因此这里另设一个明确的小例。
输入 a 的 shape 为 `[2,1]`，纸笔值为 `[[3],[7]]`。
沿 axis=1 做和归约，结果 shape 为 `[2]`，值为 `[3,7]`。
每一行只有一个数，不需要真的做两个数相加。
但结果 shape 与输入不同，不能直接把结果用户改成输入 a 而忽略类型。

教学示意：

```text
%a : float[2,1]
%r = reduce_sum(%a, axis=1) : float[2]
%z = add(%r, [1,1])
```

希望改写为：

```text
%r_new = reshape(%a, [2]) : float[2]
%z = add(%r_new, [1,1])
```

这是为解释匹配而写的对象示意，不是从运行中取得的 IR，也不承诺某个用户写法一定产生完全相同的节点顺序。

### 5. 匹配条件精读：只消掉长度为 1 的轴

[peephole.cc](../../triton-legacy/lib/codegen/transform/peephole.cc) 第 99–111 行：

```cpp
bool peephole::rewrite_unit_red(ir::instruction *value, ir::builder& builder){
  auto x = dynamic_cast<ir::reduce_inst*>(value);
  if(!x)
    return false;
  ir::value *arg = x->get_operand(0);
  auto shapes = arg->get_type()->get_tile_shapes();
  if(shapes[x->get_axis()] == 1){
    builder.set_insert_point(x);
    ir::value* new_red = builder.create_reshape(arg, x->get_type()->get_tile_shapes());
    x->replace_all_uses_with(new_red);
    return true;
  }
  return false;
}
```

value 形参接收遍历中的一条任意 IR 指令；并不是一开始就保证为归约。
dynamic_cast 成功时 x 指向同一个 reduce_inst，失败时为 nullptr。
`if(!x)` 使不适用的指令迅速返回 false，表示本规则未修改它。

arg 是归约第 0 个操作数，即本例 a。
`arg->get_type()->get_tile_shapes()` 读到 `[2,1]`。
`x->get_axis()` 返回 1，因此条件检查 shapes[1] 是否等于 1。
若输入是 `[2,4]`，这一规则不会尝试把四项求和改成 reshape。

set_insert_point(x) 把新指令放在旧归约之前，保证原来使用 r 的位置能够使用新结果。
create_reshape 的数据输入为 arg，目标 shape 来自 x 的结果类型 `[2]`。
注意不是取 arg 的 shape，否则根本没有移除长度为 1 的归约轴。
new_red 是新 reshape 节点的 value 指针，名字沿用 red 不代表它仍在计算归约。

replace_all_uses_with 把所有原来引用 x 的用户改成引用 new_red。
最后返回 true，让外层记录“这条指令被一个规则处理过”。
该函数没有直接执行 `erase_from_parent`，所以不要在这里就说旧归约已经从块列表消失。

**确认：** a=`[[3],[7]]` 时 z 是什么？改写前后都是 `[4,8]`；变化的是计算表示与 shape 转换，不是输入数据本身。

### 6. replace_all_uses_with 怎样改接一根边

[value.cc](../../triton-legacy/lib/ir/value.cc) 第 62–73 行：

```cpp
void user::replace_all_uses_with(value *target) {
  for(auto it = users_.begin(); it != users_.end(); it++){
    (*it)->replace_uses_of_with(this, target);
  }
}
void user::replace_uses_of_with(value *before, value *after) {
  for(size_t i = 0; i < ops_.size(); i++)
    if(ops_[i] == before)
      ops_[i] = after;
  after->add_use(this);
  before->erase_use(this);
}
```

对本例，this 首先是旧归约 x，target 是 new_red。
users_ 中包含 add；调用 add 的 replace_uses_of_with 时，this 已切换为这个 add 对象。
before 仍是旧归约，after 是 reshape，所以 add 的操作数槽从 before 指针改成 after 指针。
随后把 add 登记为 reshape 的用户，并从旧归约用户集合移除。

这段历史实现直接遍历并修改 users_，本章展示其意图与真实源码，不将它当作容器迭代安全性的通用范例。
静态阅读也没有替该实现提供全部运行正确性证明。
学习关键是看清“替换使用者中的操作数”，不是简单给旧值改一个名字。

### 7. 规则由谁调用，旧节点由谁清理

[peephole.cc](../../triton-legacy/lib/codegen/transform/peephole.cc) 第 167–206 行先遍历函数、块和指令，先处理 dot，再处理其他规则。
第 197–204 行连续摘录：

```cpp
bool was_modified = false;
was_modified = was_modified || rewrite_mult(i, builder);
was_modified = was_modified || rewrite_cts_cfs(i, builder);
was_modified = was_modified || rewrite_trans_phi(i, builder);
was_modified = was_modified || rewrite_unit_red(i, builder);
was_modified = was_modified || rewrite_gep_ptr_min_off_plus_off(i, builder);
if(was_modified)
  seen.insert(i);
```

builder 来自 mod.get_builder，i 来自正在遍历的指令列表。
`||` 有短路行为：前面某个规则成功，后面的规则在这一轮不再执行。
seen 记录已成功处理的旧节点，循环继续到 seen 数量不再变化。
不要将这段写成“每条指令每次都执行五个 rewrite”。

回到 make_bin，peephole 后紧接 dce。
[dce.cc](../../triton-legacy/lib/codegen/transform/dce.cc) 第 16–38 行以 store、return、branch、atomic、barrier 等必要节点初始化工作集。
第 43–51 行从它们向操作数反向标记依赖；第 63–70 行收集并移除未标记指令。
本例旧归约的用户已全部改接 reshape，从最终输出回溯不再需要它，所以可被后面的 DCE 清除。
新 reshape 若仍通往 store，则在依赖路径上保留。

### 8. 论文 prefetch：缺源码映射，也要讲清机制

> 回到论文：§5.1.1、Listing 7，PDF 第 5 页。原文讨论在循环中提前发出 tile load，以便与独立计算重叠。已核对[作者 PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)。

本快照的 make_bin 没有独立 prefetch pass，当前查阅范围也未核实与 Listing 7 一一对应的变换实现。
不能把 cts 或 3.1 的 double buffering 检测直接改名为“论文 prefetch 实现”。
但可用下列**根据论文思想整理的教学伪码**解释重排，不是原文逐字抄录，也不是本快照代码：

```text
for k in 0 .. K-1:
    cur = guarded_load(ptr(k))
    acc = compute(acc, cur)

if K > 0:
    cur = guarded_load(ptr(0))
for k in 0 .. K-1:
    if k+1 < K:
        next = guarded_load(ptr(k+1))
    acc = compute(acc, cur)
    if k+1 < K:
        cur = next
```

cur 保存本轮数据；next 保存下轮数据。
把下轮 load 提前到本轮 compute 之前，才可能留下隐藏读延迟的独立工作。
`K>0` 防止空循环仍读取第 0 块；`k+1<K` 防止最后一轮越界预取。
guarded_load 还必须包含原先每个 tile 元素的边界掩码。
伪码赋值 cur=next 在 SSA 中需要用循环 phi 表达初值与回边值，不是原地修改某个 SSA 定义。
是否真有延迟收益、寄存器压力是否增加，必须测量；“提早写了 load”不自动等于更快。

### 9. PSSA/ψ：逐元素合并与基本块 phi 的区别

> 回到论文：§4.3、Listing 6，PDF 第 5 页；cmpp 产生互补谓词，ψ 合并被谓词保护的值。原文位置同[作者 PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)。

以下也是**论文思想教学伪码**，不是可提交给本快照的语法：

```text
x = [3, 7]
pt = (x < 5)        # [T, F]
pf = complement(pt) # [F, T]
under pt: a = y + 1
under pf: b = y - 1
z = psi((pt, a), (pf, b))
```

若 y=[10,20]，第 0 位置从 a 取得 11，第 1 位置从 b 取得 19。
这里两个位置可以选择不同定义，不必让整个 tile 走同一条基本块分支。
2.1 的 phi 则根据“执行从哪个前驱基本块过来”选择输入。
同有“合并”二字，不代表数据结构或选择依据相同。

本快照有 masked load/store 和 phi_node，但尚未核实 ψ/cmpp 的对应节点与完整 PSSA 管线。
masked load 防止某些位置访问内存，不等于完整表示任意 predicated 指令流的 ψ 合并。
若要继续历史考证，需要定位论文当时的提交和对应节点定义，不能仅凭名字相似补全结论。

### 10. 到下一章时，带走什么

实际 IR 改写闭环是：遍历找到 reduce→检查归约轴长度→在原位置前插 reshape→改接 users→后续 DCE 清旧节点。
论文机制闭环是另一条证据：它解释设计思想，但本快照缺失的映射仍保留缺口。
3.1 继续真实 make_bin 中的 layouts、coalesce、liveness、allocation 和 barriers。

自测：把 axis 长度 4 的归约改为 reshape 安全吗？不安全，本规则不匹配。
构造 layouts 时分析已经完成了吗？没有，run 才填结果。
本章是否声称实际运行过改写？没有，代码静态核对，示意值纸笔计算。

## 第 2 章小结

| 改写步骤 | 需要检查什么 | 常见错误 |
|---|---|---|
| 找到生产者 | `reshape` 的输入是否由 `DescriptorLoadOp` 产生 | 只看 reshape，不看来历 |
| 检查使用次数 | load 是否只有一个 user | 直接改类型，忽略其他消费者 |
| 检查 rank/shape | 被删除的前缀维是否都为 1，尾部 shape 是否匹配 | 元素数相同就认为可改写 |
| 改类型 | `modifyOpInPlace` + `setType` | 以为创建了第二次设备 load |
| 替换 uses | `replaceOp` 把消费者接到新结果 | 只删旧节点，不重连消费者 |
| 清理 | 后续 DCE 删除无用户节点 | 在改写函数里强行 erase |

### 综合检查

1. `value` 和 `user` 的双向关系为什么重要？pass 既要能从结果追来源，也要能从旧结果找到所有消费者并改接。
2. `create_range` 和 `insert` 是一步还是两步？工厂创建节点是第一步，`insert` 把它放进基本块并建立 `parent` 是第二步。
3. `replace_all_uses_with` 改的是 C++ 局部变量吗？不是；它改的是 IR 使用者列表中的操作数指针。
4. `peephole` 规则返回 `true` 后旧节点立刻消失吗？不一定；通常等后续 DCE 清理。
5. 论文 prefetch 和 PSSA 在本快照中有逐行对应实现吗？没有核实到；正文明确把论文思想与快照实现分开。
6. 为什么同一个 `+` 有时生成 GEP、有时生成 fadd？操作数的元素类型决定语义。
