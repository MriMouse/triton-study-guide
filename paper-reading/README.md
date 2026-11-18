# 从 MAPL 2019 论文读 Triton 源码

[返回教程](../README.md) · [打开作者 PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

把 PDF 和讲义并排打开。页码从 PDF 第一页数起，段序从各节标题下开始。图和代码清单用原编号定位。表里的语义提示帮助定位原句。读到关键源码时，先看正文如何拆句和代入输入，再用链接打开对应几行复查。第一次读编译系统，可以先用[一次编译的入口练习怎么读文件、函数和依赖](00-how-to-read.md)，再回到这里按原文顺序走。

| 原文 | 跟读 |
|---|---|
| 开始前 | [第一次读编译器，该怎样进文件、跟调用和返回](00-how-to-read.md) |
| 摘要、§1–§3，PDF 第 1–4 页 | [研究动机、矩阵乘清单、Triton-C 与编程模型](01-language.md) |
| §4，PDF 第 4–5 页 | [IR 中的类型、值、广播与谓词](02-ir.md) |
| §5，PDF 第 5–7 页 | [编译顺序、布局、内存与调优](03-compiler.md) |
| §6–§8，PDF 第 7–8 页 | [GEMM、卷积实验与结论](04-experiments.md) |

历史源码固定为 v0.1（`b9e36331be85`，2020-05-04）；现代源码固定为 `46eccb199bfd`。历史提交晚于论文。碰到术语相同的地方，先确认处理的是哪一种对象。碰到性能、相关工作或未来设想，先找对应的图、测试入口或项目边界；论文的实验结果需要当时的设备、输入及测量条件来解释。

第一遍直接从[认清仓库里三段程序](01-language.md)开始。遇到陌生写法，先跟着当前页的小例子找「输入是谁、结果存在哪里、下一步谁拿到它」。可以只看源码和纸笔推演。暂时卡在某个点时，下面这些入口可以单独读：

- 方括号、函数调用、`lambda`、`*args`、`**kwargs` → [普通 Python 到 Triton 启动](../chapters/00-foundations.md#python-launch-syntax)
- 看到旧 `type.h` 该读哪几行、形状会交给谁 → [从声明到 IR tile 类型，再到 load](02-ir.md#read-tile-type)
- C++ 的 `*`、`&`、`->` 与 Triton-C 的 mask → [从 masked load 追到 builder](02-ir.md#mask-to-builder)
- `opt.defines.push_back(...)`、两层下标和三元表达式怎样变成矩阵步长 → [逐项算 STRIDE_AK](01-language.md#read-stride-macro)
- 一次输入位移怎样变成指针更新 → [Listing 8](04-experiments.md#listing-8)

读到想深入的机制，各篇都链接了原教程章节。源码链接按三份目录并列的本地布局编写。
