---
date: 
tags:
  - study
---
硬件基础和架构：[[CUDA学习笔记]]

## 1CUDA架构特点

CUDA（Compute Unified Device Architecture）的中文全称为计算统一设备架构。CUDA程序通过nvcc编译器进行编译，CUDA程序中包含CPU部分代码和GPU部分代码，编译时会分别编译，CPU部分代码叫host code，GPU部分代码一般叫device code。