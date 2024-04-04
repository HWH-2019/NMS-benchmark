# 仓库说明

本仓库用于学习 nms 的 C++ 实现，目前主要分为 cpu 版本 和 gpu 版本。
-  **cpu 版本**：单纯按照 NMS 的基本步骤实现，没有过多关注与优化，重点在于对算法流程的理解
-  **gpu 版本**：其中又分为两个版本，一个版本是基于原生 cuda 库实现的，另一个版本是使用 Pytorch 提供的 ATen 库实现的。
  
> 本仓库中代码仅供学习使用，并不适合实际开发（或许后面可能会改一版适合项目中使用的），适合作为初学 cuda 编程的参考项目。

> cuda 编程、pybind11 使用等细节点后续将会持续更新，欢迎 star。

## 关于 non-maximum suppression

Non-Maximum Suppression (NMS) 是一种在计算机视觉和图像处理中常用的技术，特别是在对象检测算法中。它的目的是去除重叠的检测框（bounding boxes），从而保留最佳的检测结果。

在对象检测任务中，算法通常会在图像中识别出多个潜在的对象位置，并为每个位置生成一个检测框。由于这些检测框可能相互重叠，或者同一个对象可能被算法多次检测，因此需要一种方法来确定哪些检测框是最佳的，哪些应该被抑制或去除。

NMS 有效地减少了对象检测中的冗余，确保每个对象只被检测一次，并且检测框尽可能地准确。这在处理密集对象的场景或者在算法产生大量候选检测框时尤其重要。

在现代对象检测框架中，如 R-CNN 系列（包括 Fast R-CNN、Faster R-CNN 等）和 YOLO 系列，NMS 是后处理步骤中不可或缺的一部分。这些框架通常使用 NMS 来优化最终的检测结果，提高检测的准确性和效率。

## NMS 的基本步骤：

1. **得分排序**：首先，根据算法为每个检测框分配的置信度得分（confidence score）对检测框进行排序。得分最高的检测框排在最前面。

2. **选择最大的检测框**：从得分最高的检测框开始，选择一个作为基准。

3. **计算交并比**：对于剩余的检测框，计算它们与基准检测框的交并比（Intersection over Union，IoU）。IoU 是两个检测框相交部分的面积与它们并集部分的面积之比。

4. **抑制重叠检测框**：设定一个阈值（通常在 0.3 到 0.5 之间），如果某个检测框的 IoU 超过了这个阈值，意味着它与基准检测框有较大的重叠，因此应该被抑制（即从列表中移除）。

5. **重复过程**：在移除了重叠的检测框后，选择下一个得分最高的检测框作为新的基准，重复步骤 3 和 4，直到所有检测框都被考虑过。


## 运行环境

### 实验环境

- System: Ubuntu 20.04
- torch version: 1.13.1
- g++ version: 9.4.0
- python version: 3.8.18

## 代码组织

- `nms_cpu.cpp`：cpu 版本 nms 实现
- `nms_gpu.cu`：gpu 版本 nms 实现, 分别包含两种实现
- `nms_gpu_vision.cpp`：使用 pybind11 封装实现 C++ 和 python 之间的绑定
- `setup.py`：python 的 C++ 扩展模块构建脚本，负责生成动态链接库，使其能使用 python 的方式使用 nms
- `build.sh`：手动 nvcc 构建 C++ 动态链接库脚本，已弃用
- `bbox.npy`：nms 测试数据，数据格式为 `[x1, y1, x2, y2]`
- `nms_gpu_test.py`： gpu 版本 nms 测试文件


## 问题记录

#### 1. pybind11 安装与配置

本项目是使用 vscode 编写的程序，因此需要配置 pybind11 的路径（实在懒不配也行），这里记录三种配置方案：
1. 使用 `pip install pybind11` 安装 `pybind11`，然后在 `.vscode `下的` c_cpp_properties.json` 中添加 `"includePath": ["/path/to/pybind11/include"]`
2. 前往 [pybind11](https://github.com/pybind/pybind11) 下载源码，并解压到本地目录，然后在 `.vscode `下的`c_cpp_properties.json` 中添加 `"includePath": ["/path/to/pybind11/include"]`
3. 前往 torch 安装目录下，找到 `torch/include` 目录，然后在 `.vscode `下的`c_cpp_properties.json` 中添加 `"includePath": ["/path/to/torch/include"]`

针对安装的包，如 `pybind11`, `torch` 可以使用如下命令查看所在路径

```bash
pip show package_name
```

> 如果使用 `nvcc` 手动构建动态链接库，可以选择上述三种方式之一， 同时需要在命令中添加 `-I/path/to/pybind11/include` 路径。

> 如果使用 `setup.py` 构建动态链接库，则选择第三种方式为妙，在构建时会自动调用 `torch/include` 下的 `pybind11`


#### 2. 报错 cannot open source file "Python.h" (dependency of "pybind11/pybind11.h")

在使用 `pybind11` 前需要配置 `Python.h` 的路径，使用以下命令查看

```bash
python3-config --includes
```

在 `.vscode `下的`c_cpp_properties.json` 中添加 `"includePath": ["/path/to/python"]`


#### 3. 导入<torch/extension.h>报错

原因是没有在 `includePath` 中配置 `torch/extension.h` 的路径，
具体参考[这里](https://zhuanlan.zhihu.com/p/603275573)，路径一般还是在 `torch` 的目录下，对应替换即可。

#### 4. 导入 <THC/THC.h> 报错

目前随着 `pytorch` 版本的更新迭代，已经将 `THC` 移除，合并到 `ATen` 中，对于高版本的 `pytorch` （目前已知是 `1.11.1` 以上）可以参考[这里](https://blog.csdn.net/weixin_41868417/article/details/123819183)，对应修改代码。

#### 5. ImportError: libc10.so: cannot open shared object file: No such file or directory

对于编译好的包进行导入

```python
import nums_gpu
```

执行时报错

```bash
Traceback (most recent call last):
  File "nms_gpu_test.py", line 3, in <module>
    import nms_gpu
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

原因是 `libc10.so` 是基于 `pytorch` 生成的，因此需要先导入`torch` 包，然后再导入依赖于 `torch` 的包：

```python
import torch
import nums_gpu
```

#### 6. 导包时遇到 #include errors detected. Please update your includePath. Squiggles are disabled for this translation unit

还是 `includePath` 配置的问题，有些头文件的路径没有正确配置，可以检查 [问题 1](#1-pybind11-安装与配置)，[问题 2](#2-报错-cannot-open-source-file-pythonh-dependency-of-pybind11pybind11h) 和 [问题 3](#3-导入torchextensionh报错) 的配置。


## 致谢
本项目重点参考了 [NMS(in faster-rcnn) benchmark](https://github.com/fmscole/benchmark/tree/master) 与 [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) 的`nms` 代码实现，同时参考 [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 中 `setup.py` 的构建脚本及其使用 Pytorch 提供的 ATen 库的 `nms` 实现。