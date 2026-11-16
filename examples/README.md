# 可独立运行的教学示例

[返回总目录](../README.md)

这些脚本对应正文中的具体章节，需要在已安装 PyTorch/Triton 的 GPU 环境中运行。正文引用的实际输出会标明设备、版本和命令；换环境复现时数值和时间可能不同。

| 脚本 | 对应章节 | 做什么 |
|---|---|---|
| [actual_vector_add.py](actual_vector_add.py) | 4.1、4.2 | N=10、B=4 的向量加法，打印 grid、输出、编译阶段和资源 |
| [actual_ir_dump.py](actual_ir_dump.py) | 5.1、5.2、8.1 | 编译同一 kernel，导出 TTIR/TTGIR/LLIR/PTX 并打印关键行 |
| [actual_softmax.py](actual_softmax.py) | 6.1 | 2×3 稳定 softmax，与 PyTorch 比较 |
| [actual_gemm.py](actual_gemm.py) | 6.2 | 2×3 与 3×2 的显式乘加 GEMM 教学实现 |
| [actual_layernorm.py](actual_layernorm.py) | 9.3 | 小矩阵 LayerNorm 与 RMSNorm 教学实现 |
| [actual_bench.py](actual_bench.py) | 8.2 | 向量加法的 `do_bench` 计时与有效带宽 |
| [controlled_copy.py](controlled_copy.py) | 8.2 | 独立的 masked copy 脚本，保存编译产物并计时 |

运行示例前先确认实际导入的 Triton：

```bash
python3 -c 'import triton; print(triton.__version__, triton.__file__)'
```

不要直接 import 上游教程来“只看定义”：它们可能在尾部执行测试或 benchmark。示例脚本只做最小正确性检查和必要计时。
