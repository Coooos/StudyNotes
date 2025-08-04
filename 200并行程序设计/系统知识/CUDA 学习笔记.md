---
date: 
tags:
  - study
---
硬件基础和架构：[[GPU学习笔记]]

## 1CUDA架构特点

CUDA（Compute Unified Device Architecture）的中文全称为计算统一设备架构。CUDA程序通过nvcc编译器进行编译，CUDA程序中包含CPU部分代码和GPU部分代码，编译时会分别编译，CPU部分代码叫host code，GPU部分代码一般叫device code。

### 如何使用CUDA

1. 显卡驱动，否则无法使用GPU进行计算
2. 程序代码（.cu和.cuh文件）
3. 编译代码，类似于C++程序需要编译才能运行，这个过程需要CUDA Toolkit
4. 发布编译好的程序，其他人就可以下载并直接运行，不再需要代码和CUDA Toolkit，但仍需要显卡驱动

### 什么是CUDA Toolkit

简单地，可以将CUDA Toolkit视为开发CUDA程序的工具包（类似于VS之于C++），但我们不仅在开发CUDA程序时需要CUDA Toolkit，很多时候，由于硬件、软件环境的不同，他人编译好的CUDA程序我们是不能直接运行的，于是我们就要在自己的计算机上编译他人写好的CUDA代码