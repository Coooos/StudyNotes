[有关CUBLAS中的矩阵乘法函数 - 爨爨爨好 - 博客园](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)


---
cuBLAS中能用于运算矩阵乘法的函数有4个，分别是

**cublasSgemm（单精度实数）
cublasDgemm（双精度实数）
cublasCgemm（单精度复数）
cublasZgemm（双精度复数）

## cublasDgemm
`cublasDgemm` 用于执行双精度浮点数（`double` 类型）的矩阵乘法。它实现了以下数学操作：

C=α⋅op(A)⋅op(B)+β⋅C

其中：
- op(A) 和 op(B) 可以是矩阵 A 和 B 的转置或不转置。
- α 和 β 是标量。
- C 是结果矩阵。

### 函数原型

```cpp
  cublasSgemm(
          handle,
          CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
          CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
          C_ROW,          //矩阵C的行数
          C_COL,          //矩阵C的列数
          A_ROW,          //矩阵A的列数
          &a,             //alpha的值
          A,            //左矩阵
          A_COL,          //矩阵 A 的主维;行数
          B,            //右矩阵
          B_COL,          //矩阵 B 的主维;
          &b,             //beta的值
          d_C,            //结果矩阵C
          C_COL           //(C的列数)
  );
```
### 返回值

- **`cublasStatus_t`**：
    
    - 返回状态码，表示操作是否成功。可能的值包括：
        
        - `CUBLAS_STATUS_SUCCESS`：操作成功。
            
        - `CUBLAS_STATUS_NOT_INITIALIZED`：cuBLAS 上下文未初始化。
            
        - `CUBLAS_STATUS_INVALID_VALUE`：输入参数无效。
            
        - `CUBLAS_STATUS_EXECUTION_FAILED`：操作执行失败。

### 1. 矩阵的列数

矩阵的列数是指矩阵中实际的列数。例如，一个矩阵 A 的大小为 m×n，其中 m 是行数，n 是列数。这个 n 就是矩阵的列数。

### 2. 行间距（Leading Dimension）

行间距是指矩阵在内存中存储时，每一行的起始地址之间的距离。它通常用矩阵的行间距来表示，单位是元素的个数。行间距的值通常大于或等于矩阵的实际列数，但并不一定等于列数。


## cudaMallocPitch
`cudaMallocPitch` 用于在 GPU 设备上分配对齐的内存，特别适用于二维数组的内存分配，能提高内存访问性能。以下是其用法及示例代码：

### 函数原型
```cpp
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
```

- **`devPtr`**：指向分配的设备内存指针的指针。
    
- **`pitch`**：返回的行对齐大小（字节），通常大于或等于 `width`。
    
- **`width`**：每行的宽度（字节）。
    
- **`height`**：内存分配的行数。
    

### 优点

- **提高内存访问性能**：通过内存对齐，优化 GPU 的内存访问模式，提高缓存命中率。
    
- **简化复杂数组管理**：自动处理行对齐问题，简化二维数组的内存管理。

## cudaMemcpy2D
`cudaMemcpy2D` 用于在 GPU 设备和主机之间进行二维数据的拷贝。它特别适合用于在二维数组（如矩阵）之间拷贝数据。以下是其详细用法和示例代码：

### 函数原型

```cpp
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
```

### 参数说明

- **`dst`**：目标地址指针。
    
- **`dpitch`**：目标内存的行对齐大小（字节）。
    
- **`src`**：源地址指针。
    
- **`spitch`**：源内存的行对齐大小（字节）。
    
- **`width`**：每次拷贝的宽度（字节）。
    
- **`height`**：拷贝的行数。
    
- **`kind`**：拷贝方向，可以是以下值：
    
    - `cudaMemcpyHostToHost`：主机到主机
        
    - `cudaMemcpyHostToDevice`：主机到设备
        
    - `cudaMemcpyDeviceToHost`：设备到主机
        
    - `cudaMemcpyDeviceToDevice`：设备到设备
        

### 使用场景

- **主机到设备的二维数据拷贝**：将主机端的二维数组拷贝到设备端。
    
- **设备到主机的二维数据拷贝**：将设备端的二维数组拷贝回主机端。
    
- **设备到设备的二维数据拷贝**：在设备端的不同内存区域之间拷贝二维数据。
    
- **主机到主机的二维数据拷贝**：在主机端的不同内存区域之间拷贝二维数据。
