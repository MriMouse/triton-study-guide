# 第 7 章 布局、Pass 与内存同步

[返回总目录](../README.md) · [上一章：Softmax 与 GEMM](06-classic-kernels.md) · [下一章：后端、调试与调优](08-backend-debug.md)

第 6 章结束时，你已经能算出一块输出。但还有三个问题没有回答：

1. 同一个逻辑块在 GPU 上由哪个 lane、哪个 warp、哪个寄存器槽持有？
2. 编译器怎样识别一段 IR 模式，并把旧操作替换成新操作？
3. 一旦引入共享内存，编译器怎样计算存活期、复用字节，并在正确位置插入同步？

这一章按这个顺序回答：**7.1 布局 → 7.2 pass 改写 → 7.3 内存与同步**。它们不是三个孤立主题，而是一条依赖链：布局决定数据在哪里，pass 可能改变布局和内存操作，分配与同步又依赖改写后的布局和存活期。

## 本章地图

```text
逻辑值
  → 7.1：layout（lane/warp/register/block）
  → TTGIR 中的 encoding
  → 7.2：pass 匹配、改类型、替换 uses
  → 可能引入共享内存
  → 7.3：liveness → allocation → RAW/WAR → barrier
  → 交给后端 lowering
```

| 部分 | 核心问题 | 读完后能回答 |
|---|---|---|
| 7.1 | 元素由谁持有？ | 一个 128 元素块怎样映射到 lane/warp/register |
| 7.2 | 编译器怎样安全改写？ | 匹配条件、插入点、`replace_all_uses_with`、DCE 怎样配合 |
| 7.3 | 共享存储何时冲突？ | 存活期、字节区间、RAW/WAR 与 barrier 插入 |

**读法：** 不要把这些 pass 当成“优化名字清单”。每读一个变换，先问：它的输入是什么对象？匹配条件是什么？改写后谁的使用关系被替换？旧节点由谁清理？

<a id="sec-7-1"></a>
## 7.1 一个元素由哪个 lane、warp 和寄存器持有

warp 是 NVIDIA GPU 上一起组织执行的一组线程，lane 是组内编号。布局可以让一个 lane 持有多个逻辑元素，常用不同的寄存器槽区分。读到 thread 或 lane 前先记下手里的数字是元素下标还是线程编号。

本章固定现代源码 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。源码与纸笔映射已静态核对，未生成 IR、未运行 GPU。

6.2 中一个 program 计算一块矩阵。`tl.arange(0, 128)` 给的是128个**逻辑位置**，没有要求启动128个线程。现在补上中间缺口：这些位置怎样分配给线程，以及乘法需要另一种分配时怎样表达。

### 1. 先分开三个坐标

假定 x 是连续 `float32` 数组，地址从1000开始。逻辑元素 `x[37]` 的字节地址为 `1000 + 37*4 = 1148`。这是张量存储规则。

如果把元素37交给 warp 1 的 lane 1 的寄存器槽2，地址仍是1148。后者是**值的持有位置**。寄存器槽是布局中的抽象编号，不是承诺最终机器寄存器名为 `r2`：LLVM/ptxas 还会分配真实寄存器、产生溢出。

同一个逻辑值也可能被多个线程复制。因而不能从“张量有128个元素”倒推线程数、不能从 lane 编号直接推出全局地址。

| 名称 | 本章意义 | 举例 |
|---|---|---|
| dim0 | 一维逻辑坐标 | 37 |
| lane | 一个 warp 内的线程位置 | 1 |
| warp | 一个 CTA 内的 warp 位置 | 1 |
| register | 每个线程持有的第几个逻辑槽 | 2 |
| block | 布局中 CTA 集群的坐标 | 本例只有0 |

CTA 是 cooperative thread array，CUDA 中常说的 thread block；Triton 的一个普通 program 常对应一个 CTA，但现代多CTA/warp specialization 需要更细区分。

### 2. 用真实单测核验最小映射

来源：[LinearLayoutConversionsTest.cpp](../../triton/unittest/Dialect/TritonGPU/LinearLayoutConversionsTest.cpp)，`SimpleBlocked`，154–165行。下面原样摘录主体，空白整理：

```cpp
auto layout =
    toLinearLayout({16}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
EXPECT_THAT(layout, LinearLayout(
    {
        {S("register"), {}},
        {S("lane"), {{1}, {2}}},
        {S("warp"), {{4}, {8}}},
        {S("block"), {}},
    },
    {S("dim0")}));
```

`layout` 是 C++ 编译器/测试进程中的映射对象，不是 GPU 数据。`toLinearLayout` 第一个实参 `{16}` 是逻辑 shape；第二个实参是 `blocked` 测试辅助函数创建的编码属性。辅助函数在同文件33–42行，把 `spt`、`tpw`、`wpb` 等参数交给 `BlockedEncodingAttr::get`。

本例 `spt={1}` 表示基础块中每线程一元素；`tpw={4}`、`wpb={4}` 分别把4个 lane 与4个 warp 排入这一个维度。这是**布局代数的合成单测**，不是声称 NVIDIA 真实 warp 只有4个线程。真实 NVIDIA pipeline 向转换传入32，见第4节。选择该单测是为了能写出全部16个位置。

`S("lane")` 把字符串变成 MLIR 上下文中的名字。`{{1},{2}}` 两个基向量对应 lane 编号的二进制位：低位贡献1，高位贡献2。warp 两位分别贡献4和8；register 与 block 没有位，因此各只有一种取值0。

LinearLayout 的一般组合是二进制线性映射，用异或组合置位对应的基。这里各基互不重叠，因此可以用普通加法手算：

```text
逻辑位置 = lane + 4 * warp
warp 0: lane 0,1,2,3 →  0, 1, 2, 3
warp 1: lane 0,1,2,3 →  4, 5, 6, 7
warp 2: lane 0,1,2,3 →  8, 9,10,11
warp 3: lane 0,1,2,3 → 12,13,14,15
```

`EXPECT_THAT` 比较计算出的映射和明确写出的期望对象。它没有执行这16个元素的加法；它检查编译器的布局转换数学。

**确认：** 元素14在哪里？warp3、lane2、register0。字节地址仍由输入指针和元素14决定，不能写成 `base+lane*4`。

### 3. 元素多于线程时，register 维怎样出现

同文件193–205行 `ShapeLargerThanLayout` 的真实主体：

```cpp
auto layout =
    toLinearLayout({128}, blocked({1}, {4}, {4}, {1}, {1}, {0}, {0}));
EXPECT_EQ(layout, LinearLayout(
    {
        {S("register"), {{16}, {32}, {64}}},
        {S("lane"), {{1}, {2}}},
        {S("warp"), {{4}, {8}}},
        {S("block"), {}},
    },
    {S("dim0")}));
```

相比上一段，仅 shape 从16变成128。一个基础布局覆盖16个位置，现在要重复8次，register 编号需要3个二进制位。编号 `r=0..7` 依次贡献 `16*r`。在这个无重叠例子：

```text
i = lane + 4*warp + 16*r
i=37: r=2, 剩余5；warp=1, 剩余1；lane=1。
i=127: r=7, warp=3, lane=3。
```

每线程8个逻辑元素，总计 `4*4*8=128`。因此逻辑块扩大，线程数可以不变，代价可能变为每线程更多寄存器和指令。**不能据此直接预测耗时**：硬件warp大小、访存向量化、活跃值数量及实际 lowering 还没有进入这个合成例子。

把单测外推到真实32-lane一维 blocked 教学配置 `sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4]`，shape128时每线程一个元素，常见映射是 `i=lane+32*warp`。这句是该配置的纸笔推演，不是 4.1 在 B=4 下生成的真实布局。

### 4. 谁把布局加入 TTGIR

来源：[NVIDIA compiler.py](../../triton/third_party/nvidia/backend/compiler.py)，`make_ttgir`，308–320行，摘录两条相邻阶段语句：

```python
pm = ir.pass_manager(mod.context)
dump_enabled = pm.enable_debug()
emuTF32 = (capability // 10 >= 8)
passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
```

`mod` 是前一阶段交来的 MLIR module；`mod.context` 是管理类型/属性等编译对象的上下文。`pm` 收集待运行的 pass。`capability` 来自目标架构，`opt` 是已解析编译选项；这条注册把目标名、warp数、每warp32线程和CTA数传入 TTIR→TTGIR 转换。它现在只添加变换，随后 `pm.run` 才执行编译变换，GPU仍未读入 x/y。

TTIR 中一个张量类型强调 shape 和元素类型；TTGIR 加入 layout encoding，保留“算什么”并开始决定“谁持有哪个值”。转换不等于最终寄存器分配。

### 5. dot 的输入为什么要 convert_layout

来源：[TritonToTritonGPUPass.cpp](../../triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp)，dot 转换分支，224–255行。非连续摘录，省略B的对称分支：

```cpp
Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
    getContext(), origShape, retSizePerThread, retOrder, numWarps,
    threadsPerWarp, numCTAs);
RankedTensorType retType = origType.cloneWithEncoding(dEncoding);
auto aType = cast<RankedTensorType>(adaptor.getA().getType());
Attribute aEncoding = aType.getEncoding();
Value a = adaptor.getA();
Value c = adaptor.getC();
if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
  Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
      getContext(), 0, dEncoding, aEltType);
  auto dstType = aType.cloneWithEncoding(encoding);
  a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstType, a);
}
c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);
```

`origShape/origType` 来自旧 dot 的结果类型；`retSizePerThread/retOrder` 是此分支前面算出的输出分块参数。第一句创建结果编码，第二句复制原 shape/dtype 并换上编码，得到 `retType`，不是分配一个结果张量。

`adaptor` 提供转换框架重映射后的操作数。`getA()` 取左矩阵的 IR value；其 `aType` 上可能已有编码。`aEltType` 在231行从该类型提取元素类型。`a` 是局部 C++ 句柄，不是 Python 输入数组。

条件检查A是否已经是 dot operand 编码；不是时，`0` 表示左操作数（B用1），`dEncoding` 是关联的结果布局，`aEltType` 决定相关类型约束。`dstType` 保留逻辑矩阵的 shape/dtype。`create` 在当前位置插入转换操作，返回的新 IR value 替代局部变量 `a`。这不在编译时搬运 GPU 数据；它表达未来设备代码必须取得相同逻辑值的另一种分布。

`c` 是累加器，转为结果布局。紧随其后的256–259行用 `replaceOpWithNewOp<DotOp>` 建立新 dot，接收新 `a/b/c`，并替换旧操作的使用者。

实际转换可能消除、只重命名/重排线程内值、使用 shuffle，或通过共享内存跨warp交换。仅看到 `convert_layout` 不能断言一定有共享内存或barrier；后续布局优化与 lowering 决定实现。

### 6. 纸笔闭环与自测

给128元素合成布局中的每个元素赋值 `v[i]=100+i`。元素37初始由 `(r2,warp1,lane1)` 持有，值137。若目标教学布局把元素37交给 `(r2,warp0,lane5)`，转换后该位置应持有137；逻辑索引和值均不变。此目标只是展示转换合同，未声称是上游实际生成的目标。

1. `register` 基是16/32/64，是否说明每元素占16字节？不是，基的单位是逻辑坐标。
2. 改 dtype 为 fp16 后，元素37的地址偏移？74字节；布局坐标公式没有因此自动改变。
3. 为什么一个不需要转换的输入也可以留下 `convert_layout`？中间IR允许冗余表达，优化pass可折叠；是否存在须看实际IR。
4. `retType` 是设备输出内存吗？它是编译器中的类型对象，输出内存仍由host指针提供。

> 回到论文：§5.2.1/Figure 5、§5.2.2/Figure 6，PDF第6页，讨论分层分块和合并访存。现代 LinearLayout、MLIR编码以及 dot operand 类型属于实现演进，不能说2019原文已经定义了本章的 C++ API。参见[论文定位表](../appendices/reference.md#sec-a-1)。

下一部分保留这里的“编译器对象”视角，追一个 pass 怎样注册、匹配并替换 IR。

**第一部分交出的边界。** 7.1 说明了布局是“逻辑值到 lane/warp/register/block 的映射”，并且 dot 的输入可能需要 `convert_layout`。现在把镜头转向编译器：这些编码和操作写在 IR 里，pass 怎样找到它们、判断能否改写，又怎样保证消费者的接口不变？

<a id="sec-7-2"></a>
## 7.2 一个 pass 怎样真正改掉一段 IR

use 是某条操作拿另一个 IR 值作为输入这件事。假如 `%b = add %a, 1`，add 就有一处对 `%a` 的使用。替换 uses 要改使用者的输入；删掉旧定义之前要确认还有没有操作依赖它。

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`；本章为静态源码带读，IR示意明确标注，未运行编译器。

编译器的 pass 是接受并检查/修改IR的一次程序处理。它不是 GPU kernel，也不是“让GPU再跑一遍”。本章沿 `make_ttir → add_combine → CombineOpsPass → RankedReduceDescriptorLoads`，看一个能完全列出安全条件的局部重写。

### 1. Python 阶段怎样拿到 C++ pass

[NVIDIA compiler.py](../../triton/third_party/nvidia/backend/compiler.py)291–305行 `make_ttir`，摘录主干：

```python
pm = ir.pass_manager(mod.context)
pm.enable_debug()
passes.common.add_inliner(pm)
if capability // 10 < 9:
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
passes.common.add_canonicalizer(pm)
passes.ttir.add_combine(pm)
passes.ttir.add_reorder_broadcast(pm)
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)
passes.ttir.add_loop_unroll(pm)
pm.run(mod, 'make_ttir')
return mod
```

`mod` 是前端生成的module；`pm` 与它共用context。inliner处理函数内联，canonicalizer处理通用规范化。对较早架构，descriptor先转成pointer路径，因此本章选择的descriptor模式未必有机会出现在那条pipeline中。

`add_combine(pm)` 接收同一个pass manager，将具体pass追加进去。它不返回新张量；最后 `pm.run` 按顺序处理 `mod`，原module被修改，返回值仍是这个module句柄。

Python名字不是凭命名约定自动发现的。[passes.cc](../../triton/python/src/passes.cc)47行明确绑定：

```cpp
ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
```

这是C++宏，用工厂 `createTritonCombineOps` 生成Python可调用包装；`0`表示没有额外pass选项参数。对应声明在[Passes.td](../../triton/include/triton/Dialect/Triton/Transforms/Passes.td)6–7行（仅摘录开头，后续description省略）：

```tablegen
def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
```

TableGen是编译时生成C++接口的描述语言。`triton-combine` 是命令行注册名；`mlir::ModuleOp` 说明pass作用范围是module。这里定义接口，实际算法在下一段。

### 2. 谁收集模式，谁反复尝试

[Combine.cpp](../../triton/lib/Dialect/Triton/Transforms/Combine.cpp)275–294行 `CombineOpsPass::runOnOperation`，非连续摘录，省略其他模式注册：

```cpp
MLIRContext *context = &getContext();
RewritePatternSet patterns(context);
ModuleOp m = getOperation();
patterns.add<RankedReduceDescriptorLoads>(context);
if (applyPatternsGreedily(m, std::move(patterns)).failed())
  signalPassFailure();
```

`getContext()` 来自pass基类，取module所属上下文；`patterns` 是待匹配规则的集合；`getOperation()` 返回本次pass处理的module `m`。模板参数 `<RankedReduceDescriptorLoads>` 选择具体C++模式类。

真实代码还加入 dot/add、select/load、addptr等规则，不能把本段省略理解为pass只处理一种模式。`std::move` 将模式集合移交给greedy驱动器。驱动器遍历、尝试匹配，重写后继续尝试直至收敛或失败；不是恰好单遍每条规则。

`.failed()` 失败时把pass标记失败，让pipeline停止并上报，而不是继续输出自称有效的IR。

### 3. 先给出要处理的具体逻辑

一个descriptor读取 shape `[1,4]`，随后 reshape 为 `[4]`。因为最前面的维度只有1，元素顺序没有变化：

```text
读到的二维值：[[10,20,30,40]]
reshape 后：  [10,20,30,40]
```

这些数用于说明合同；编译器重写时并不知道也不读取10/20/30/40。

以下为**教学IR结构示意**，不是本书生成的 dump，也没有声称是可直接提交 `triton-opt` 的完整语法：

```text
before:
  %a : tensor<1x4xf32> = descriptor_load(%desc, %indices)
  %b : tensor<4xf32>   = reshape(%a)
  consumer(%b)
after:
  %a : tensor<4xf32>   = descriptor_load(%desc, %indices)
  consumer(%a)
```

本模式把 rank reduction 直接放到descriptor load的结果类型中，去掉 reshape。是否能做，由下面所有条件共同决定；不能凭图形相似就写自己的替换规则。

### 4. 逐句读匹配条件

真实来源：[Combine.cpp](../../triton/lib/Dialect/Triton/Transforms/Combine.cpp)197–225行，`RankedReduceDescriptorLoads` 的方法，主体原样摘录：

```cpp
mlir::LogicalResult
matchAndRewrite(triton::ReshapeOp reshapeOp,
                mlir::PatternRewriter &rewriter) const override {
  auto loadDef = reshapeOp.getSrc().getDefiningOp<triton::DescriptorLoadOp>();
  if (!loadDef || !loadDef->hasOneUse())
    return failure();
  int loadRank = loadDef.getType().getRank();
  int reshapeRank = reshapeOp.getType().getRank();
  if (!(reshapeRank < loadRank))
    return failure();
  ArrayRef<int64_t> loadShape = loadDef.getType().getShape();
  ArrayRef<int64_t> reshapeShape = reshapeOp.getType().getShape();
  for (int i = 0; i < loadRank - reshapeRank; ++i) {
    // Only rank reduce unit dims.
    if (loadShape[i] != 1)
      return failure();
  }
  if (loadShape.take_back(reshapeRank) != reshapeShape)
    return failure();
  rewriter.modifyOpInPlace(
      loadDef, [&]() { loadDef.getResult().setType(reshapeOp.getType()); });
  rewriter.replaceOp(reshapeOp, loadDef.getResult());
  return success();
}
```

`reshapeOp` 是驱动器刚匹配到的候选reshape；`rewriter` 是管理IR修改和驱动器通知的对象。第一句取reshape的输入SSA value，再找产生它的操作；模板要求该生产者必须是 `DescriptorLoadOp`。若输入来自add或普通load，`loadDef` 为空，返回 `failure()` 表示“本规则不适用”，不表示整个编译失败。

`hasOneUse()` 要求load结果只被使用一次。为什么？下一步直接修改该结果类型，其他消费者可能仍要求旧rank。单使用避免这种额外修复责任。

`loadRank/reshapeRank` 来自操作结果类型，不是设备读取的数值。对于 `[1,4]→[4]` 分别为2、1，满足严格降rank。对 `[4]→[1,4]` 返回失败。

`ArrayRef` 是只读数组视图，不复制shape。循环次数为 `loadRank-reshapeRank`；本例只检查最前一维等于1。`take_back(reshapeRank)` 取最后1维，必须恰等于 `[4]`。因此 `[2,2]→[4]`、`[1,4]→[2,2]` 都不能匹配；元素数量相同仍不足以满足该规则。

### 5. 改类型与替换使用者是两件事

`modifyOpInPlace` 告诉框架对现存load作原地修改。`[&]` 是C++ lambda按引用捕获，里面能访问当前 `loadDef/reshapeOp`。`setType` 把load结果类型换为reshape的结果类型；指针、索引操作数仍属于现存load，没有创建第二次设备load。

`replaceOp(reshapeOp, loadDef.getResult())` 把所有原本读取reshape结果的uses接到load结果，并删掉被替换的reshape。它不只是改一个C++局部变量；SSA使用关系真的被重连。

`success()` 告诉greedy驱动器IR已改变，可以重新考虑邻近规则。这与前面的 `failure()` 是模式匹配结果；最外层 `.failed()` 则检查驱动器整个处理是否失败，不要混淆层次。

### 6. 手动执行一次规则

输入 `[1,1,4]→[4]`，load仅一个use：

1. 找到生产者descriptor load；通过。
2. rank从3降为1；通过。
3. 前两维1、1；通过。
4. 剩余尾部 `[4]` 与结果 `[4]` 相同；通过。
5. load结果改为rank1，消费者改用它；reshape删除。

如果同一个load还有一个消费者 `sum(%a,axis=0)`，第1组条件就拒绝。不能绕过单使用检查，否则该消费者面对的新rank和axis含义可能不同。

若输入 `[4,1]→[4]`，虽然数学上也能reshape，本规则检查被删除的**前缀**维为1，发现前维4，拒绝。这说明“数学上可能存在别的优化”不等于“这个实现会做”。

### 7. 自测与下一跳

- 为什么本pass运行时不需要GPU输入值？规则依赖类型、use关系与操作种类，处理的是IR。
- 为什么不能仅删掉reshape？消费者还指向它的结果，必须替换uses。
- `[1,8]→[8]` 一定在最终pipeline消掉吗？若满足规则且到达该pass则有机会；其他阶段及目标分支影响实际IR，未运行前不承诺最终dump。
- 这是autotune吗？不是。它不试跑候选配置、不测延迟，是有条件的IR重写。

> 回到论文：§5.1.2 Tile-Level Peephole，PDF第6页，帮助理解利用tile形状作局部等价变换。本部分的 descriptor/MLIR 模式是后续实现，不能充当原论文该 pass 的源码；历史实现另见 2.2 及[论文定位表](../appendices/reference.md#sec-a-1)。

下一部分关心有些转换引入共享存储后，怎样判断它能否复用、什么时候需要同步。

**第二部分交出的边界。** 7.2 的改写只动 IR 结构和类型，还没有分配真实共享字节。一旦某个转换需要跨线程交换数据，编译器就要回答：这块共享存储存活多久？能否和其他值复用同一片字节？复用前后哪些读写必须同步？这就是 7.3 的 liveness、allocation 和 membar。

<a id="sec-7-3"></a>
## 7.3 共享存储复用后，谁还在读旧内容

共享内存是同一线程块内可协作读写的存储。RAW 表示先写再读，读者要等数据写好。WAR 表示先读再写，复用存储的写者要等旧数据读完。两种依赖都要问涉及哪些字节、哪些线程。

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。本章只静态分析，内存时间线为教学推导，未运行GPU。

本章闭环是：一个跨线程交换需要临时共享存储，编译器怎样计算其生存期、怎样把存储分配信息交给同步分析、怎样决定在冲突访问前插入barrier。共享内存是同CTA线程能共同访问的片上存储；一个线程完成写入，不自动说明其他线程已经读完。

### 1. 用四个数说明两种危险

设CTA内一组线程把 `[10,20,30,40]` 写入共享槽 `S[0..3]`，另一种分布读取它们，实现7.1的layout转换。这里是计算合同的**教学时间线**，不是断言该布局一定如此lower：

```text
W0: 各线程写 S
B0: 等待参与线程写完
R0: 各线程按目标分布读 S
B1: 等待参与线程读完
W1: 同一片共享字节被下一次转换重写
```

缺B0可能发生 read after write（RAW，写后读）：一个快线程读取另一慢线程尚未写完的位置。缺B1可能发生 write after read（WAR，读后写）：一个快线程覆盖另一个慢线程还没读完的旧内容。这里的“after”描述逻辑上应发生的顺序，实际硬件若无同步可能乱序交错。

只有同一物理区域、相关参与者、相关访问顺序才构成危险。两个全局load没有通过共享区交换，不能因它们都叫load就加CTA barrier；两个纯读也不因读同位置而发生写冲突。

### 2. 生存期先是编译器中的区间

[Allocation.cpp](../../triton/lib/Analysis/Allocation.cpp)455–495行 `AllocationAnalysis::resolveLiveness`，非连续摘录：

```cpp
DenseMap<Operation *, size_t> operationId;
operation->walk<WalkOrder::PostOrder>(
    [&](Operation *op) { operationId[op] = operationId.size(); });
Liveness liveness(operation);
auto getValueLivenessRange = [&](Value value) {
  auto liveOperations = liveness.resolveLiveness(value);
  auto minId = std::numeric_limits<size_t>::max();
  auto maxId = std::numeric_limits<size_t>::min();
  llvm::for_each(liveOperations, [&](Operation *liveOp) {
    if (operationId[liveOp] < minId) {
      minId = operationId[liveOp];
    }
    if ((operationId[liveOp] + 1) > maxId) {
      maxId = operationId[liveOp] + 1;
    }
  });
  return Interval(minId, maxId);
};
resolveExplicitBufferLiveness(getValueLivenessRange);
resolveAliasBufferLiveness(getValueLivenessRange);
resolveScratchBufferLiveness(operationId);
```

`operation` 是被分析函数/区域的根操作。`operationId` 把IR操作指针映射为整数编号；这不是GPU时钟。后序遍历先走子操作、再走外层操作，使循环等父操作编号覆盖其内部使用，避免外部定义、内部使用的value被过早判死。

`Liveness` 是MLIR分析对象，接收同一个根。lambda `getValueLivenessRange` 接收一个SSA `Value`，查询它活跃涉及的操作。遍历求最小编号和最大编号加1，返回半开区间 `[minId,maxId)`。加1使单个操作也有非空区间。

最后三句把查询函数分别交给显式buffer、alias buffer与scratch buffer处理。alias是同一存储的另一个视图，不应因产生新SSA名字就当作新存储；scratch是某个lowering临时需要的区域，不一定对应用户显式张量。

### 3. 手算生存期与物理区间

假设教学编号：A分配在2，最后使用在5，则活跃区间 `[2,6)`；B分配在6，最后使用在8，则 `[6,9)`。两者不相交，分配器可能复用同一字节区 `[0,16)`，但还须满足尺寸、对齐、别名、异步与控制流约束。这是说明可能性，不是对本书源码运行的分配结果。

若B从4开始使用，则 `[2,6)` 与 `[4,9)` 重叠，不得仅凭二者名字不同就复用同一物理字节。分配器在同文件741–754行同时考虑操作生存区间、存储尺寸区间及异步区域的重叠。

“编译器认为某SSA值不再使用”与“所有线程已经完成最后一次访问”也不同。复用后的WAR正是同步分析需要保留的边界。

### 4. allocation 对象怎样交给 membar

[NVIDIA Membar.cpp](../../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Membar.cpp)31–54行 `runOnOperation`，非连续摘录，省略cluster处理但保留依赖：

```cpp
ModuleOp mod = getOperation();
NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
ModuleAllocation allocation(
    mod, ttng::getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
auto solver = createDataFlowSolver();
auto *regions = solver->load<triton::BufferRegionAnalysis>(
    triton::BufferRegionAnalysis::Mode::AllMemory, &allocation);
if (failed(solver->initializeAndRun(mod))) {
  signalPassFailure();
  return;
}
ModuleMembarAnalysis membarPass(allocation, NVIDIA::canSkipBarSync);
membarPass.runAnalysis<MembarAnalysis>(*regions);
```

`computeCapability/ptxVersion` 是pass选项，先构造目标信息。scratch大小回调依赖目标，因此不是不分架构的一固定大小。

`allocation` 接收module及回调，组织每个函数的分配信息。`solver` 是数据流求解器，`load` 创建区域分析；`&allocation` 传地址，让区域分析能把descriptor/别名追到实际分配。`Mode::AllMemory` 表示分析范围，不是要求把所有内存放到共享区。

`initializeAndRun(mod)` 真正运行分析；失败停止pass，不能拿未知区域继续假定不冲突。`membarPass` 接收同一个allocation对象和目标相关的可跳过同步判定；`runAnalysis` 又接收已求好的 `regions`。这就是文件之间的依赖连接，不是两个独立清单。

真实代码还在中间运行cluster barrier与mbarrier初始化/arrival处理；多CTA与异步协议不能只用本章最小单CTA例子代替。

### 5. 是否重叠，不能只比较buffer编号

[Analysis/Membar.cpp](../../triton/lib/Analysis/Membar.cpp)35–48行，`AllocationSlice::intersects`：

```cpp
if (!allocationInterval.intersects(other.allocationInterval))
  return false;
if (!triton::mayOverlap(physicalFootprint, other.physicalFootprint))
  return false;
if (bufferId == other.bufferId &&
    areBufferIndicesProvablyDifferent(*this, other))
  return false;
```

`this` 和 `other` 是两次访问的slice。先检查字节分配区间；区间不重叠就无需继续。再检查物理footprint，保留CTA与存储空间等身份；数值偏移相同、但不同存储空间不自动构成冲突。

第三条允许在同buffer中证明索引槽不同，例如合法的不同流水线槽。后面的50–71行对同源subslice再比较逻辑偏移和shape，任一维不相交可证明分离；无法证明时保守返回可能重叠。不能说“bufferId不同所以永不冲突”，因为分配器可能复用同一片物理存储。

### 6. 在哪条操作前插入同步

同文件351–374行 `syncIfNeeded/insertBarrier`，删去函数声明的连续主体：

```cpp
if (!builder)
  return nullptr;
auto &pending = membarInfo->pending;
if (!requiresThreadSync(pending, effects) &&
    !pending.isIntersected(effects, filter, &allocation, sliceFilter))
  return nullptr;
builder->setInsertionPoint(op);
Operation *barrier = insertBarrier(op, builder, cluster);
membarInfo->sync();
return barrier;
```

`effects` 是当前 `op` 的读写/线程需求摘要，`pending` 累积此前尚未被同步清除的效果；`filter/sliceFilter` 携带目标相关过滤规则，`allocation` 提供分配映射。

`pending.isIntersected` 的分类并非根据操作名字猜测。[Membar.h](../../triton/include/triton/Analysis/Membar.h)254–267行明确组合三个方向；下面摘录调用关系，省略每个调用的布尔标志与filter实参：

```text
RAW: pending.syncWriteSlices 对 effects.syncReadSlices
WAR: pending.syncReadSlices  对 effects.syncWriteSlices
WAW: pending.syncWriteSlices 对 effects.syncWriteSlices
```

这是源码调用的文字整理，不是可编译C++。这三个集合装的是访问slice及其来源操作，不是设备数据。每对slice还要通过第5节的物理重叠检查和目标filter；两个read集合没有被作为写冲突组合。W0→R0会进入RAW候选，R0→W1进入WAR候选，即使两次使用来自不同SSA名字。

没有builder时只分析、不改IR。两个判据都否时直接返回空：既无线程同步需求，又无相关访问冲突，则不插入barrier。否则把插入点放在当前op之前，创建同步操作，再清除已经被这次同步覆盖的历史状态。当前op自身的新效果随后仍须加入，不能把它也当成已同步完成。

`insertBarrier` 用 `cluster` 选择cluster barrier；普通分支创建 `triton::gpu::BarrierOp`，地址空间为Local。这里构造的仍是IR操作，实际barrier指令到GPU执行时才等待线程。

### 7. async wait 不等于所有参与者已经交接

同文件168–184行的真实分类摘录：

```cpp
if (isa<ttng::WaitBarrierOp>(op))
  return {ThreadSyncKind::MBarrierWait};
if (isa<ttng::FenceAsyncSharedOp>(op))
  return {ThreadSyncKind::Completion};
if (isa<triton::gpu::AsyncWaitOp>(op))
  return {ThreadSyncKind::SharedCompletionNeedsSync};
```

异步操作可能由部分线程发出，完成与其他线程可安全消费是不同条件。分类把 `AsyncWaitOp` 标成共享完成后仍可能需要同步；后面的 `requiresThreadSync` 根据实际后继访问/发布者决定，不能改写为“任何wait后必定barrier”。WGMMA wait还区分发起warp group的局部完成与更广参与范围。

因此对流水线槽 `[slot0,slot1]`，读slot0前要有该异步复制的完成协议；覆盖slot0前还要知道旧消费者已经结束。单独保留一个wait而忽略别名与迭代边界，并不能证明双缓冲安全。

### 8. 自测

1. 两次只读同一共享slice，需要由RAW/WAR理由插barrier吗？不需要；其他线程协议可能另有要求。
2. A与B SSA名字不同，分配到同一 `[0,16)`，能否跳过冲突检查？不能。
3. 活跃区间 `[2,6)` 与 `[6,9)` 是否重叠？不重叠，但这本身不证明GPU访问交接安全。
4. membar是否在编译时等待GPU？没有，分析IR并插入未来执行的同步操作。

> 回到论文：§5.2.3/Figure 7与§5.2.4，PDF第6页，对应共享分配和RAW/WAR同步问题。现代descriptor footprint、mbarrier、异步代理与多CTA协议是后续演进，2019文字不能解释本快照全部同步语义。见[定位表](../appendices/reference.md#sec-a-1)。

## 第 7 章小结

| 层次 | 这一章的关键结论 | 常见误解 |
|---|---|---|
| 布局 | 同一个逻辑 tensor 可以有不同 encoding；元素由 lane/warp/register 共同定位 | 逻辑元素数等于线程数 |
| pass | 匹配条件先决定“能不能改”，再改类型并替换 uses | 删掉旧操作就自动完成改写 |
| 存活期 | 编译器用操作编号形成半开区间 | 存活期是 GPU 纳秒时间 |
| 分配 | 只有时间区间和字节区间都重叠才冲突 | SSA 名字不同就一定不共享存储 |
| 同步 | RAW/WAR/WAW 按读写集合和物理区间判断 | 每个 load 后都加 barrier |
| 异步 | wait 表示完成协议的一部分，不等于所有参与者已交接 | 看到 wait 就可以无条件复用 |

### 综合检查

1. `tl.arange(0, 128)` 会启动 128 个线程吗？不会；128 是逻辑元素数，布局决定每个线程持有多少。
2. `convert_layout` 一定需要共享内存吗？不一定；可能被消除、只在线程内重排，或需要 shuffle/共享内存。
3. 一个 pass 返回 `failure()` 表示编译失败吗？在模式匹配里通常只表示“本规则不适用”；最外层驱动器失败才是整个 pass 失败。
4. 两个值 SSA 名字不同，能直接复用同一片共享字节吗？不能只看名字；要看分配区间、存活期、别名和目标约束。
5. 存活期不重叠是否意味着不需要同步？不意味着；复用同一片共享存储仍可能触发 WAR，需要屏障保证旧读者读完。
