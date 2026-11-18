# 第 8 章 后端、调试与调优：从 IR 到可发射程序

[返回总目录](../README.md) · [上一章：布局、Pass 与内存同步](07-layout-passes-memory.md) · [下一章：大模型算子](09-llm-kernels.md)

第 7 章留下了带布局、共享内存分配和同步的 IR。这一章先把它送到底层，再回答“怎样证明它算对了、测到了什么”：

- **8.1**：TTGIR → LLVM 方言 → 真正的 LLVM Module → PTX → cubin → 驱动装载 → launcher 发射；
- **8.2**：最小复现、解释器、IR dump、lit 测试、数值验证、`do_bench` 与 autotune。

这两部分放在一起，是因为编译产物不是终点。看到一段 PTX 或一个 `CompiledKernel` 之后，下一步必须能回答：它是否真的算对？测到的是编译、提交还是设备执行？

## 本章地图

| 问题 | 这一章的工具/入口 | 不能证明什么 |
|---|---|---|
| IR 怎样变成目标代码？ | `make_llir`、`make_ptx`、`make_cubin`、`load_binary` | 不能证明数值正确或速度快 |
| 失败发生在哪一层？ | 前端错误、IR verifier、ptxas、资源、launch、设备数值 | 只看异常最后一行不够 |
| 索引和 mask 对不对？ | `TRITON_INTERPRET=1`、CPU 小例 | 不模拟真实线程时序、共享内存竞争 |
| pass 改对了没有？ | `triton-opt` + lit/FileCheck | 不是 GPU 数值测试 |
| 数值是否正确？ | 参考实现、容差、边界样例 | 不能用一个阈值覆盖所有 dtype/shape |
| 时间花在哪？ | `do_bench`、同步、事件计时 | 不能把 kernel 时间当请求延迟 |
| 实际输出在哪？ | 8.3 节：阶段、资源计数与一次计时 | 不能据此推广到其他 shape/设备 |

**读法：** 8.1 跟对象种类（module → 字符串 → bytes → 句柄 → launch），8.2 跟证据种类（索引 → 数值 → IR → 性能）。不要把“编译成功”当成“算对”，也不要把“解释器通过”当成“GPU 正确”。

<a id="sec-8-1"></a>
## 8.1 从 TTGIR 到可以发射的 CUDA 函数

lowering（降级转换）是把抽象操作翻成更接近目标硬件的操作。这里 TTGIR 带着布局进入后端，经 LLVM 相关阶段生成 PTX，再由 ptxas 得到 cubin。PTX 可以当汇编文本读，cubin 是驱动可加载的设备二进制容器。

固定SHA `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。本章静态核对 NVIDIA 路径，并在 8.3 节使用从该提交构建的 Triton 做一次实际编译与启动。10.4 节另讲 AMD，本章不把 CUDA 阶段当作所有硬件的共同格式。

本章接收前面已经有布局的MLIR module，最终交出host侧的launcher。一个对象可能都被叫 `src`，但它沿阶段从module变成字符串再变成bytes，必须逐段重新确认种类。

### 1. stages 保存函数，不是保存GPU结果

[NVIDIA compiler.py](../../triton/third_party/nvidia/backend/compiler.py)661–671行 `add_stages`，非连续摘录：

```python
stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
```

`stages` 是编译器创建的有序字典；前面已经放入ttir/ttgir（Gluon入口不同）。每个lambda保存当前 `self/options/capability`，未来接收前一阶段的产物 `src` 和同一个元数据字典。4.2的编译循环调用这些函数，并缓存每阶段产物。

| 阶段边界 | 输入 | 输出 | 此时有没有访问用户 x/y 内容 |
|---|---|---|---|
| make_llir | MLIR TTGIR module | LLVM IR文本字符串 | 没有 |
| make_ptx | LLVM IR字符串 | PTX字符串 | 没有 |
| make_cubin | PTX字符串 | cubin二进制bytes | 没有 |
| load_binary | bytes和资源要求 | CUDA module/function句柄 | 装载代码，未执行kernel |
| launcher | 句柄、grid、stream、参数 | 提交设备工作 | 提交后由设备执行读写 |

`llir` 标签覆盖了内部两个不同表示：MLIR中的LLVM方言，以及真正的LLVM Module，不能把两者合为同一个对象。

### 2. make_llir先在MLIR里lower

来源同文件423–473行，非连续摘录，省略插桩/调试分支与部分pass：

```python
mod = src
pm = ir.pass_manager(mod.context)
pm.enable_debug()
passes.convert.add_scf_to_cf(pm)
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
nvidia.passes.ttgpuir.add_membar(pm, capability, ptx_version)
nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
passes.ttgpuir.add_canonicalize_llvm_ir(pm)
nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
passes.convert.add_nvvm_to_llvm(pm)
pm.run(mod, 'make_llir')
```

`src` 是前一步TTGIR module，`mod=src` 没有深拷贝，仍指同一编译器对象。SCF是结构化控制流，CF是更接近基本块跳转的表示。共享分配、tensor memory分配与同步在转换到LLVM方言之前运行，原因见7.3。

`add_to_llvmir` 注册目标lowering，将高层分布式张量操作拆成更低层地址、标量/向量、目标指令等表示；后面继续清理及lower剩余NVGPU/NVVM操作。`pm.run` 才应用这些变换。此处的LLVM方言仍属于MLIR，仍使用MLIR context、operation和attribute。

这不是“把整个tile必然写入共享内存”：资源和转换取决于具体kernel。上面是pipeline阶段，不能把出现allocation pass当作每个kernel都分配同样共享容量。

### 3. 再从MLIR导出真正的LLVM Module

同文件493–511、513–545行，非连续摘录：

```python
llvm.init_targets()
context = llvm.context()
llvm_mod = llvm.to_module(mod, context)
proc = sm_arch_from_capability(capability)
features = get_features(options, capability)
triple = 'nvptx64-nvidia-cuda'
flags = get_llvm_flags()
llvm.attach_datalayout(llvm_mod, triple, proc, features, flags, abi="shortptr")
```

`llvm.init_targets` 初始化LLVM目标支持；`context` 是新建LLVM上下文，与 `mod.context` 的MLIR上下文不是同一个类型。`to_module` 接收lower后的MLIR module，创建真正LLVM module `llvm_mod`。

`proc` 是架构名，`features` 是目标特性，`triple` 确定NVIDIA PTX目标。`attach_datalayout` 把指针/类型布局等目标信息附给module；它不分配设备输入内存。

接下来的513–528行 `llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, ...)` 运行LLVM优化；选项会影响NVPTX向量化等策略。531–539行把共享内存、warp数、scratch尺寸等写入 `metadata`，供装载/启动阶段消费。

最后三句（542–545行）：

```python
ret = str(llvm_mod)
del llvm_mod
del context
return ret
```

返回的是序列化LLVM IR文本，先保存 `ret` 再销毁临时LLVM对象。下一阶段形参同名 `src`，收到的已经是字符串，不能再调用 `src.context`。

### 4. LLVM到PTX，仍没有最终机器码

同文件547–565行，非连续摘录：

```python
triple = 'nvptx64-nvidia-cuda'
proc = sm_arch_from_capability(capability)
features = get_features(opt, capability)
flags = get_llvm_flags()
enable_fpsan = is_enabled(opt, "fpsan")
ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False, enable_fpsan,
                            sched4reg=opt.sched4reg, enable_fpsan=enable_fpsan, disable_lsr=opt.disable_lsr,
                            abi="shortptr")
names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
assert len(names) == 1
metadata["name"] = names[0]
```

`translate_to_asm` 的 `src` 是上一节字符串；相同目标信息决定生成PTX。`ret` 是PTX文本。正则查找可见kernel入口名，要求恰好一个，再保存进共享的 `metadata`。后续版本/target文本处理见564–576行。

PTX是NVIDIA的虚拟指令集文本，仍需目标汇编工具生成cubin。一个PTX寄存器名也不是已经锁定的物理寄存器位置。

### 5. ptxas怎样成为一个真实子进程

同文件577–584、611–617、654–659行，非连续摘录：

```python
ptxas = get_ptxas(self.target.arch).path
with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
    tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
    fsrc.write(src)
    fsrc.flush()
    fbin = fsrc.name + '.o'
```

`get_ptxas` 根据目标选工具路径；`fsrc` 保存刚生成的PTX，`flog` 接收诊断。`flush` 确保外部工具能看见文件内容，`fbin` 是工具的输出路径。不是把 `.o` 后缀当作内容格式证据，源码随后按二进制读取它。

同一函数的命令主体：

```python
ptxas_cmd = [
    ptxas, *debug_info, *fmad, '-v', *disable_opt, *reg_alloc, *ptx_extra_options, f'--gpu-name={arch}',
    fsrc.name, '-o', fbin
]
subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
```

前面的变量来自调试、浮点融合、优化及目标选项。列表直接作为进程参数，非shell拼接；`check=True` 让失败进入异常分支。异常分支读取stderr并提供复现命令，而不是输出伪cubin。

成功后654–659行：

```python
with open(fbin, 'rb') as f:
    cubin = f.read()
if os.path.exists(fbin):
    os.remove(fbin)
return cubin
```

现在 `cubin` 是bytes，交给公共编译器持久缓存；临时文件可删，返回值已保存内容。到这里也没有发射kernel。

### 6. CompiledKernel先持有bytes，句柄惰性创建

[公共 compiler.py](../../triton/python/triton/compiler/compiler.py)407–438行 `CompiledKernel.__init__`，非连续摘录：

```python
self.metadata = KernelMetadata(**metadata)
backend = make_backend(self.metadata.target)
self.packed_metadata = backend.pack_metadata(self.metadata)
self.src = src
self.name = self.metadata.name
self.kernel = self.asm[binary_ext]
self.module = None
self.function = None
self._run = None
```

元数据先从缓存JSON恢复，`target` 还原成目标对象；`backend` 确定二进制扩展名。`self.asm` 在425–429行读取阶段文件，cubin按bytes读，其他IR按文本读。`self.kernel` 选中cubin，而不是重新编译。

`module/function` 初始为空，说明已经有cubin的Python对象还未必已经装载到当前GPU。452–498行 `_init_handles/run` 非连续摘录：

```python
device = driver.active.get_current_device()
self._run = driver.active.launcher_cls(self.src, self.metadata)
shared = getattr(self._run, "shared", self.metadata.shared)
max_shared = max_shared_mem(device)
if shared > max_shared:
    raise_(OutOfResources(shared, max_shared, "shared memory"))
self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
    self.name, self.kernel, shared, device)
```

先取当前设备，构造launcher并检查资源，再让驱动装载bytes，返回module/function句柄及资源信息。真实代码还处理tensor memory、线程上限、hooks和进程归属。`run` 属性只在 `_run is None` 时触发 `_init_handles`，然后返回launcher，不是每次都调用ptxas。

### 7. CudaLauncher怎样连接到驱动发射

[NVIDIA driver.py](../../triton/third_party/nvidia/backend/driver.py)265–305、307–355行，非连续摘录：

```python
signature = dict(src.signature)
utils = triton.runtime.driver.active.utils
expanded_signature = expand_signature(signature.values(), tensordesc_abi)
self.arg_annotations = annotate_arguments(expanded_signature)
self.launch = wrap_handle_tensordesc(utils.launch, signature, tensordesc_meta)
```

`src.signature` 来自最初kernel的参数类型合同，而不是cubin内数组值。descriptor参数需要按ABI展开。`self.launch` 包装实际驱动工具的launch，并处理descriptor传参；这些信息使Python实参能按设备函数签名打包。

`__call__` 最后接收grid、stream、function、metadata和用户参数，准备必要scratch后调用：

```python
self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
            kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
            profile_scratch, self.arg_annotations, self.kernel_signature, kernel_args)
```

`function` 是上一节装载得到的设备函数句柄，`kernel_args` 通常来自本次host调用，插桩模式可能追加参数。驱动C实现见[driver.c](../../triton/third_party/nvidia/backend/driver.c)：316–318行调用 `cuModuleLoadData/cuModuleGetFunction`；1126行通过 `cuLaunchKernelExHandle` 提交执行。

现在才越过编译/装载边界，设备随后执行程序。host的launch返回通常仅表示提交，若要读取结果或准确计时，需要遵守stream依赖与同步协议。

### 8. 用4.1的一次调用串回去

N=10、B=4是纸笔教学输入。编译期间B进入特化条件，x/y/out的类型参与签名；设备内容0到9不会被编译器逐个相加。未命中缓存时走上述编译链，命中时可直接恢复CompiledKernel。首次使用需要当前设备装载代码，随后以grid `(3,1,1)` 和本次指针/N提交工作。GPU执行后才产生十个结果。

自测：把cubin缓存从一个进程拿到另一个进程，可以直接复用CUDA function指针吗？不可以；二进制bytes与驱动句柄有不同生命周期，需要在目标运行时装载。只把PTX保存下来，也不能称已完成本章cubin路径。

> 回到论文：§5，PDF第5–7页讲JIT编译职责；当代MLIR→LLVM→PTX→ptxas、惰性装载与现代launcher是具体实现演进。历史驱动路径在3.2单独核对，不把本章反投射到2019。见[论文定位表](../appendices/reference.md#sec-a-1)。

**第一部分交出的边界。** 8.1 结束时，host 已经能拿到 `CompiledKernel`、装载后的 function 句柄和 launcher。但“能发射”不等于“算对了”，也不等于“测到了设备时间”。下一部分从 N=10、B=4 的尾块错误出发，建立一套从索引、数值到性能的验证顺序。

<a id="sec-8-2"></a>
## 8.2 先证明确实算对，再解释测到了什么

读计时前先写下计时开始点和结束点。GPU 调用通常异步提交，host 从一次调用返回可能只表示工作已排进队列。用误差检查回答「结果对不对」，再用明确包括哪段工作的计时回答「耗了多久」。

固定源码 `46eccb199bfd3f3559b512c4a15870bd45e6a74d`。8.2 给出受控模板；8.3 节执行其中的最小脚本和一次 `do_bench`，并保留实际输出。换环境时仍要重新核对版本与资源。

闭环从一个尾块错误出发：N=10、B=4，最后一个program产生索引8、9、10、11。我们只想处理前两个。如果store忘记mask，即使输出前十个看起来正确，也可能已经越界写。若load忘记mask，后面masked store不会撤销已经发生的越界读。

### 1. 把失败缩成可解释的输入

保留4.1的指针、N与B，移除模型、随机数据加载器、benchmark循环和autotune，只留一次调用与断言。记录：完整源码、SHA、实际导入的Triton路径/版本、PyTorch/CUDA或HIP版本、GPU型号、shape/dtype/stride、编译选项、seed与命令。

不要只留下异常最后一行。前端语义错误、IR verifier错误、ptxas失败、资源不足、launch参数问题和设备数值错误发生在不同层。8.1解释了这些对象边界；根据最早失败阶段选择下一步。

**教学错误示意，不能照此运行：**

```python
offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(X + offsets)       # N不是B的倍数时，尾块可能越界
tl.store(Y + offsets, x)       # 同样缺失mask
```

正确的教学改写：

```python
mask = offsets < N
x = tl.load(X + offsets, mask=mask, other=0.0)
tl.store(Y + offsets, x, mask=mask)
```

`other=0` 是无效load位置的返回值，不是往输入数组末尾真的补了两个零。mask阻止无效内存访问；输出无效位置同样必须屏蔽。

### 2. 解释器能回答什么

官方本地文档[debugging.rst](../../triton/docs/programming-guide/chapter-3/debugging.rst)介绍 `TRITON_INTERPRET=1`，66–75行明确列出限制：该快照不支持bfloat16数值操作及某些间接访存模式，例如先load一个指针，再用它load数据。

解释器适合检查逐块索引、mask、广播和数值思路。它没有复现实际GPU线程时序、共享存储竞争、Tensor Core lowering或资源占用；解释器通过不等于GPU正确，也不能把解释器耗时当kernel性能。

对于N=10/B=4，应能逐步检查三块索引 `[0,1,2,3]`、`[4,5,6,7]`、`[8,9,10,11]`，最后mask为 `[True,True,False,False]`。若这一步都不对，无需先研究PTX。

### 3. 从一次失败保留可重现IR

[上游README](../../triton/README.md) 的调试选项和[knobs.py](../../triton/python/triton/knobs.py)是当前快照的依据。常用变量：

| 变量 | 观察对象 | 本例用途 |
|---|---|---|
| `MLIR_ENABLE_DUMP=1` 或kernel名 | 每个MLIR pass前后 | 找到mask/类型在哪阶段改变 |
| `MLIR_DUMP_PATH` | dump输出位置 | 写入本任务 `/research` 子目录 |
| `LLVM_IR_ENABLE_DUMP=1` | LLVM优化过程 | 已排除前端问题后再看 |
| `TRITON_REPRODUCER_PATH` | 编译阶段MLIR复现材料 | 保存失败前module与pipeline信息 |
| `TRITON_ALWAYS_COMPILE=1` | 强制重新编译 | 避免缓存命中绕过目标pass |

不要为了获得dump删整个共享缓存。为本次复现设置独立 `TRITON_CACHE_DIR`，保留输入和失败原文；首次与缓存命中路径本来就不同。

本章不伪造 IR。5.3 节给出从 4.1 kernel 实际导出的 TTIR/TTGIR/LLIR/PTX 片段，8.3 节给出资源计数。若读到真实 `tt.load`，需要核对它的 mask 操作数是否来自 `offsets < N`；不能只搜索某个 `%mask` 名字，因为 SSA 打印名不稳定。

### 4. lit测试不是GPU数值测试

真实[test/Triton/combine.mlir](../../triton/test/Triton/combine.mlir)第1行：

```text
// RUN: triton-opt %s -canonicalize -triton-combine | FileCheck %s
```

`%s` 由lit替换为当前测试路径。`triton-opt` 读取IR并运行给定pass；`FileCheck` 用文件中的CHECK指令检查输出文本结构。它不分配x/y、不发射GPU，也不测速度。

例如同文件27–43行构造dot后add，CHECK要求合成带累加器的dot；前一测试保留多使用者反例。测试既记录期望，也记录不能变换的条件。修改编译器后须用匹配构建的工具，不能拿旧二进制验证新C++源码。

未来若已有匹配构建，可在容器的 `/research` 隔离构建/副本中运行单个lit测试。当前不安装、不构建；不存在可用 `triton-opt` 时不能报告“lit通过”。

### 5. 一个受控最小脚本

以下是**教学脚本**，不是上游摘录；8.3 节执行了 `actual_ir_dump.py` 和 `actual_bench.py`，[controlled_copy.py](../examples/controlled_copy.py) 保留为独立模板。它只检查十个确定值，计时前编译并同步一次。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(X, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, v, i < N)

def main():
    print("triton", triton.__version__, triton.__file__)
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "hip", torch.version.hip)
    print("device", torch.cuda.get_device_name())
    x = torch.arange(10, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    def launch():
        return copy_kernel[(triton.cdiv(x.numel(), 4),)](x, y, x.numel(), B=4)
    kernel = launch()                    # 首次可能包含编译与装载
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        if stage in kernel.asm:
            from pathlib import Path
            Path(f"/research/runs/triton-copy/{stage}.txt").write_text(kernel.asm[stage])
    ms = triton.testing.do_bench(launch, warmup=5, rep=20, return_mode="median")
    print("median kernel interval ms", ms)

if __name__ == "__main__":
    main()
```

此处B=4服务于最小索引复现，不能当成高性能配置。`kernel.asm` 是确实编译后才可取的产物；AMD没有PTX，条件检查会跳过。解释器模式不提供相同的编译产物接口，应只运行数值/索引部分，删除取asm和计时部分。

输出目录要预先创建。下面是运行模板；8.3 节使用同一入口执行了示例。先检查 GPU 占用并选择可用编号，脚本文件须已保存，镜像必须已有匹配依赖：

```bash
./experiment status
./experiment gpu 0 bash -lc '
  mkdir -p /research/runs/triton-copy /research/cache/triton-copy
  export TRITON_CACHE_DIR=/research/cache/triton-copy
  export MLIR_DUMP_PATH=/research/runs/triton-copy/passes.txt
  export MLIR_ENABLE_DUMP=copy_kernel
  python3 /workspace/triton-study-guide/examples/controlled_copy.py \
    > /research/runs/triton-copy/run.log 2>&1
'
```

本工作区额外设施 `.research-runtime/README.md` 规定：`/workspace` 只读源码，`/research` 放输出、缓存与隔离环境，`/models`只读。它不属于三目录搬迁包；搬迁后须使用目标工作区自己的实验入口。容器有CPU/内存/时长限制，普通volume没有磁盘硬配额。默认镜像导入的 Triton 不一定是本章固定 SHA，打印版本只是暴露差异，不能解决二进制兼容；不应简单把本地源码加进PYTHONPATH就声称运行了这一版编译器。

### 6. 数值验证要覆盖哪些变化

copy用精确相等；softmax、Norm、GEMM应与更高精度参考比较。容差要随dtype、累加长度、算法和模型需求确定，不能全书套一个阈值。至少保留：整块和尾块、N小于block、跨block、多行、非平凡数值，以及接口承诺支持的stride。

NaN、Inf、全mask、空行、极端指数等不能靠随机正常分布自然覆盖。是否支持这些输入先定义合同；不支持时明确拒绝或说明，不能拿未定义行为算“通过”。

若使用原子累加或重复执行修改输入，benchmark必须恢复状态。否则第二次运行测的是另一个数学问题，正确性和性能都失去可比性。

### 7. do_bench到底计什么

[testing.py](../../triton/python/triton/testing.py)318–357行 `do_bench`，非连续摘录：

```python
fn()
di.synchronize()
start_event[i].record()
fn()
end_event[i].record()
di.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
```

首个 `fn()` 先触发初始化/编译；同步后才估计并安排warmup/repeat。计时事件围住每次 `fn()`，结束后同步才读取间隔。真实代码还估计迭代次数、每轮清缓存、可清梯度，时间单位是毫秒。上面几句不是连续函数，`i` 来自内部重复循环。

这个口径与Python墙钟计时不同，也与CUDA graph replay、温L2缓存、模型请求端到端耗时不同。只比较同一口径、相同输入/精度与相同初始化范围；编译时间、分配、装载是否包括必须单独写清。

### 8. autotune是试配置，不是又一个IR pass

**教学装饰器改写**，不在本书中运行：

```python
@triton.autotune(
    configs=[triton.Config({"B": 128}, num_warps=4),
             triton.Config({"B": 256}, num_warps=4)],
    key=["N"],
)
@triton.jit
def tuned_copy(X, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, tl.load(X + i, i < N, other=0), i < N)
```

host须用 `grid=lambda meta: (triton.cdiv(N, meta["B"]),)`，不能固定128对应的grid后再试B256。B进入constexpr和grid；`num_warps`是编译/启动选项。两候选有可能产生不同编译产物，再各自运行测时。

[autotuner.py](../../triton/python/triton/runtime/autotuner.py)233–263行：`all_args` 合并位置/关键字参数，`key` 提取用户指定的N并追加张量dtype，转为tuple；缓存未命中时剪枝候选，用 `_bench` 测每个config，`min(timings,key=timings.get)` 选配置。它缓存的是配置选择，而JIT还缓存各配置的编译产物，是不同层的缓存。

若性能依赖stride/形状维度却没放入key，配置复用可能不合适；若输出做累加，则需要 `reset_to_zero`、`restore_value` 或钩子按合同恢复状态。对copy覆盖写，本例不需要每轮清零。

### 9. 自测

1. 只测 `time.time()` 围住launch可得GPU执行时间吗？通常只测提交，除非把完成同步纳入口径。
2. 解释器通过，是否排除共享内存race？不能。
3. lit通过，是否证明数值与速度？不能；它检查特定IR变换结构。
4. 首次autotune比后面慢是否意味着kernel变慢？可能包括多次编译与搜索，需要拆开计时。

> 回到论文：§5.3，PDF第7页讨论自动调优思想。本章讨论的 `Config/key`、缓存、解释器和现代调试工具是当前实现，不继承论文旧设备上的速度结论。见[论文定位表](../appendices/reference.md#sec-a-1)。

## 8.3 实际运行：资源计数与一次 `do_bench`

`actual_ir_dump.py` 已经给出小 kernel 的资源计数：`n_regs=12`、`n_spills=0`、`shared_bytes=0`、`num_warps=4`。这些数字来自编译产物，不是从 `BLOCK_SIZE=4` 推导出来的。下面再用一个较大的输入跑一次 `do_bench`，观察计时口径。

**命令。**

```bash
python examples/actual_bench.py
```

**输出。**

```text
n 1048576 block 1024 grid (1024,)
n_regs 26 n_spills 0
median_ms 0.014336000196635723
effective_gbytes 0.012582912
effective_GB_per_s 877.714273675364
```

**怎样读。**

- `n=1048576`、`BLOCK_SIZE=1024`，所以 grid 是 1024 个 program；这里的 1024 是 program 数，不是元素数或线程数。
- `n_regs=26`、`n_spills=0` 是这次编译的实际资源计数。B 变大后寄存器用量和小 B 不同，不能复用 4.3 的 `n_regs=12`。
- `do_bench` 先 warmup，并在每次计时前清 L2 缓存，再用 GPU 事件围住每次 launch；报告中位数 `0.0143 ms`。有效字节数按读 x、读 y、写 out 三个 FP32 流计算，约 `0.01258 GB`，换算约 `878 GB/s`。
- 这个口径不包含张量分配和首次编译，也不等于模型请求延迟；它只是一次单机小实验，不能当作 A800 的稳定上限。

**边界。** 本节没有做 CUDA graph、不同 block 搜索或跨设备比较。要回答“哪个配置更快”，需要固定输入、精度、计时口径并重复测量。

## 第 8 章小结

### 编译产物链

| 阶段 | 输入对象 | 输出对象 | 是否已读用户数据 |
|---|---|---|---|
| TTIR | AST/前端 module | 逻辑 IR module | 否 |
| TTGIR | TTIR module | 带布局的 IR module | 否 |
| LLIR | TTGIR module | LLVM 方言/LLVM 文本 | 否 |
| PTX | LLVM 文本 | PTX 文本 | 否 |
| cubin | PTX 文本 | 二进制 bytes | 否 |
| load_binary | bytes + 资源要求 | module/function 句柄 | 装载代码，未执行 |
| launch | 句柄、grid、stream、参数 | 设备执行 | 是，执行时读写 |

### 验证顺序

1. **先缩输入**：保留最小 kernel、N、B、dtype、stride，去掉模型、benchmark 和 autotune。
2. **再查索引/mask**：用解释器或 CPU 小例检查尾块，确认 load/store 都受保护。
3. **再查 IR**：用 `MLIR_ENABLE_DUMP`、`TRITON_REPRODUCER_PATH` 定位失败阶段。
4. **再查数值**：与更高精度参考比较，覆盖整块、尾块、多行、非连续等合同内情形。
5. **最后计时**：明确 warmup、同步、cache flush、编译/分配是否计入。

### 综合检查

1. `CompiledKernel` 一创建就装载到 GPU 了吗？没有；`_init_handles` 惰性装载。
2. PTX 是最终机器码吗？不是；还要经过 ptxas/driver 生成 cubin/目标代码。
3. 解释器通过能排除共享内存竞争吗？不能；解释器不复现真实线程时序。
4. lit 测试通过能证明数值正确吗？不能；它检查 IR 变换结构。
5. `do_bench` 的毫秒数能直接当模型请求延迟吗？不能；口径、缓存、输入和端到端范围都不同。
