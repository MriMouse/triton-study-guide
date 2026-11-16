# Triton 中文源码带读

一本从 2019 年论文出发、沿着历史源码与现代实现阅读 Triton 的中文教程。面向会基本 Python、刚开始接触 DNN 编译和大模型 infra 的读者。

本书以两个固定源码快照为依据：2019 论文时代的 Triton-C/旧 IR，以及今天的 Python/MLIR 实现。每一部分都尽量把代码放回它所在的对象流里：谁创建、谁接收、谁改写、最终变成什么。纸笔推演、上游源码摘录和实际运行输出会明确区分。

## 目录

| 章 | 内容 |
|---|---|
| [第 0 章 基础](chapters/00-foundations.md) | 职责分层、Python/C++ 语法、shape/stride/地址、program/CTA/warp/SM |
| [第 1 章 历史路线（一）](chapters/01-legacy-frontend.md) | Triton-C 语言、masked dereference、host 选项、源码字符串到 AST |
| [第 2 章 历史路线（二）](chapters/02-legacy-ir-passes.md) | 旧 Triton-IR、value/user 图、真实 pass 顺序、reduce→reshape 改写 |
| [第 3 章 历史路线（三）](chapters/03-legacy-layout-backend.md) | 旧布局、共享存储、存活期/分配、RAW/WAR barrier、LLVM/PTX/driver、调优 |
| [第 4 章 向量加法](chapters/04-vector-add-jit.md) | 从 host 调用、grid、地址与 mask，到 JIT 参数绑定、缓存、编译与发射 |
| [第 5 章 前端与第一份 IR](chapters/05-frontend-ir.md) | Python AST → semantic/builder → TTIR；真实测试 IR 与 TTGIR 读法 |
| [第 6 章 Softmax 与 GEMM](chapters/06-classic-kernels.md) | 两种归约方向：行内 softmax 与沿 K 的矩阵乘、分组和 autotune |
| [第 7 章 布局、Pass 与内存同步](chapters/07-layout-passes-memory.md) | lane/warp/register 映射、pass 匹配与替换、liveness/allocation/barrier |
| [第 8 章 后端、调试与调优](chapters/08-backend-debug.md) | TTGIR→LLVM→PTX→cubin→launcher；最小复现、数值、lit 与计时 |
| [第 9 章 大模型算子](chapters/09-llm-kernels.md) | Attention 在线 softmax、LayerNorm/RMSNorm、KV cache 与分页 |
| [第 10 章 现代扩展](chapters/10-modern-extensions.md) | Gluon 显式布局、低精度 block scaling、AMD 后端 |
| [第 11 章 两代对照](chapters/11-two-generations.md) | 同一加法在前端、IR、合并访存优化中的两代实现差异 |
| [附录](appendices/reference.md) | 论文定位表、术语速查、CPU 练习与运行边界 |

## 怎么读

**读完论文后（推荐）。** 先读第 0 章建立地址与执行模型，再按第 1–3 章走论文时代的历史实现，然后读第 4–10 章的现代实现，最后用第 11 章做两代对照。

```text
第 0 章 → 第 1 章 → 第 2 章 → 第 3 章 → 第 4 章 → 第 5 章 → 第 6 章
        → 第 7 章 → 第 8 章 → 第 9 章 → 第 10 章 → 第 11 章
```

**主要想学现代 Triton。** 可以从第 0 章直接进入第 4 章，把第 1–3 章当作论文背景参考；第 11 章仍建议在读完现代与历史两条路线后回读。

**主要关心 LLM 算子。** 先读第 0、4、6 章，再进入第 9 章的 Attention、Norm 与 KV Cache；布局、同步和后端遇到问题时回查第 7–8 章。

**只想查一个概念。** 从[术语速查](appendices/reference.md#sec-a-2)或[论文定位表](appendices/reference.md#sec-a-1)进入，再点回对应章节。

## 读代码时注意

- 代码块分三类：**上游摘录**、**教学整理/改写**、**实际运行输出**。文中会标注来源和固定版本。
- 纸笔推演只说明教学输入下的数学关系，不是 GPU 实测；实际运行输出会单独标明设备、版本和命令。
- 遇到 `tl.load`、`store`、`pass` 这类词，先问当前在 host、编译期、IR、产物还是设备执行层。
- 书中的门牌号、寄存器、共享内存和线程映射都是解释模型；真实硬件行为以编译产物和测量为准。

## 实际运行

第 4、5、6、8、9 章附有实际运行记录，使用 NVIDIA A800 80GB、PyTorch 2.10.0+cu129，以及从固定提交 `46eccb199bfd` 在隔离目录构建的 Triton 3.9.0。每节给出命令、环境、输出和边界；这些记录只说明对应小例子的行为，不构成性能基准。

## 源码版本

| 路线 | 提交 | 时间 |
|---|---|---|
| 现代实现 | `46eccb199bfd3f3559b512c4a15870bd45e6a74d` | 2026-11-15 |
| 历史实现 | `b9e36331be854e37aa14c538861689533ec0dfb0`（v0.1） | 2020-05-04 |

历史快照晚于 2019 论文，保留 Triton-C 与自有 IR，但不是已证实的论文实验精确版本。两份源码与本书保持并列目录：

```text
任意父目录/
├── triton/                 # 现代官方 main 快照
├── triton-legacy/          # 官方 v0.1
└── triton-study-guide/     # 本书
```

内部链接全部使用相对路径。版本和来源记录见[现代版本](appendices/source-version.json)、[历史版本](appendices/legacy-version.json)；代码摘录许可见[CODE-NOTICE.md](CODE-NOTICE.md)。

## 说明

本书是源码阅读教程，不是 Triton 的完整 API 手册，也不替代上游文档、测试和论文原文。没有逐行覆盖所有架构分支、反向训练和周边工具；遇到具体实现差异时，以固定提交的源码为准。
