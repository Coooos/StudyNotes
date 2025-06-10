[有关CUBLAS中的矩阵乘法函数 - 爨爨爨好 - 博客园](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)

---
# Cublas库函数

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


# CUDA_CHECK
### CUDA_CHECK
在 CUDA 编程中，定义一个 `CUDA_CHECK` 宏来检查 CUDA 函数的返回值是一种常见的做法，这样可以简化错误检查的代码，使其更加清晰和易于维护。下面是定义和使用 `CUDA_CHECK` 宏的方法：

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// 定义 CUDA_CHECK 宏来检查 CUDA 函数的返回值
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %d: %s\\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
```

### 使用示例

在 CUDA 程序中，可以使用 `CUDA_CHECK` 宏来检查 CUDA API 调用或内核启动是否成功。例如：

```cpp
int main() {
    // 分配设备内存
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * 100));

    // 启动内核
    myKernel<<<1, 1>>>(d_data);

    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备以确保所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
```

在这个示例中，每个 CUDA API 调用都使用 `CUDA_CHECK` 宏进行了检查。如果调用失败，宏会输出错误信息并退出程序。

### 解释

- `cudaError_t err = call;`: 执行 CUDA 调用并将返回值存储在 `err` 中。
- `if (err != cudaSuccess)`: 检查返回值是否为 `cudaSuccess`，如果不是，说明发生了错误。
- `fprintf(stderr, ...)`: 输出错误信息，包括文件名、行号和错误描述。
- `exit(EXIT_FAILURE)`: 终止程序并返回失败状态。

这种 `CUDA_CHECK` 宏的定义方式可以帮助开发人员快速定位和处理 CUDA 编程中的错误，提高程序的健壮性和可维护性。

以下分别是针对 CUBLAS 和 CUSPARSE 库的检查宏代码示例：

### CUBLAS_CHECK 宏

```cpp
#include <stdio.h>
#include <cublas_api.h>  // CUBLAS头文件

// 定义 CUBLAS_CHECK 宏来检查 CUBLAS 函数的返回值
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error in file '%s' in line %d: %s\\n", \
                __FILE__, __LINE__, cublasGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
```

### 使用示例

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cublas_api.h>

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));  // 创建 CUBLAS 句柄

    // 示例：矩阵乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n = 2;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0.0f};

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));

    CUBLAS_CHECK(cublasDestroy(handle));  // 销毁 CUBLAS 句柄

    printf("Matrix multiplication result:\\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", C[i * n + j]);
        }
        printf("\\n");
    }

    return 0;
}
```

### CUSPARSE_CHECK 宏

```cpp
#include <stdio.h>
#include <cusparse_api.h>  // CUSPARSE头文件

// 定义 CUSPARSE_CHECK 宏来检查 CUSPARSE 函数的返回值
#define CUSPARSE_CHECK(call) \
do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "CUSPARSE error in file '%s' in line %d: %s\\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
```

### 使用示例

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_api.h>

int main() {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));  // 创建 CUSPARSE 句柄

    // 示例：稀疏矩阵向量乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n = 3;
    const int nnz = 2;
    const float values[] = {1.0f, 2.0f};
    const int rowPtr[] = {0, 1, 2, 2};
    const int colIdx[] = {0, 1};
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3] = {0.0f};

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cusparseMatDescr_t descrA;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));  // 创建矩阵描述符
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, idxBase));
    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));

    cusparseSpMV(handle, opA, &alpha, descrA, values, rowPtr, colIdx, CUSPARSE_DATA_FLOAT, CUSPARSE_INDEX_32I,
                 x, CUSPARSE_DATA_FLOAT, &beta, y, CUSPARSE_DATA_FLOAT, CUSPARSE_SPMV_CSR);

    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));  // 销毁矩阵描述符
    CUSPARSE_CHECK(cusparseDestroy(handle));  // 销毁 CUSPARSE 句柄

    printf("Sparse matrix vector multiplication result:\\n");
    for (int i = 0; i < n; ++i) {
        printf("%f\\n", y[i]);
    }

    return 0;
}
```

以上代码中，`CUBLAS_CHECK` 和 `CUSPARSE_CHECK` 宏分别用于检查 CUBLAS 和 CUSPARSE 库函数的返回值。如果函数调用失败，宏会输出错误信息并退出程序。

**解释：**

- `cublasStatus_t` 和 `cusparseStatus_t` 是 CUBLAS 和 CUSPARSE 库函数返回的错误码类型。
- `cublasGetErrorString` 和 `cusparseGetErrorString` 函数用于将错误码转换为可读的错误描述字符串。
- 宏通过检查函数调用的返回值来判断调用是否成功。如果失败，输出错误信息并退出程序，帮助开发者快速定位问题。

需要注意的是，CUBLAS 和 CUSPARSE 库的错误码和函数可能因版本不同而有所变化，建议开发者参考对应版本的官方文档以获取准确的信息。
# CUDASparse库函数
## CUSPARSE 稀疏矩阵乘法函数解析

这个函数 `Cuda_dLnrAlg_mxC_AddScl_csrmxA_X_mxB` 实现了使用 CUSPARSE 库进行稀疏矩阵与稠密矩阵的乘法运算，并支持标量乘法和加法。下面我将详细解释这个函数的各个部分：

### 函数参数说明

```c
void Cuda_dLnrAlg_mxC_AddScl_csrmxA_X_mxB(
    char transa,            // A矩阵的转置选项(N/T/C)
    int nRowA,             // A矩阵的行数
    int nColC,             // 结果矩阵C的列数
    int nColA,             // A矩阵的列数
    double alpha,          // 标量乘数
    const char* matdescra, // 矩阵描述符(4字符)
    double* A,             // A矩阵的非零元素值
    int iRowOff_A,         // A矩阵的行偏移(未使用)
    int jColOff_A,         // A矩阵的列偏移(未使用)
    int* vAllColIdx,       // A矩阵的列索引数组
    int* viRow1stNz,       // A矩阵的行指针数组
    double* B,             // 稠密矩阵B
    int iRowOff_B,         // B矩阵的行偏移
    int jColOff_B,         // B矩阵的列偏移
    int ldb,               // B矩阵的leading dimension
    double beta,           // 结果矩阵C的标量乘数
    double* C,             // 结果矩阵C
    int iRowOff_C,         // C矩阵的行偏移
    int jColOff_C,         // C矩阵的列偏移
    int ldc                // C矩阵的leading dimension
)
```

### 函数执行流程

1. **索引调整** (1-based → 1-based)
   - 函数开始和结束处对列索引和行指针进行了+1/-1操作
   - 这可能是为了适应某种特定的索引约定

2. **初始化CUSPARSE**
   ```c
   cusparseHandle_t handle = NULL;
   CHECK_CUSPARSE(cusparseCreate(&handle));
   ```

3. **创建稀疏矩阵描述符**
   - 根据输入参数 `matdescra` 设置矩阵类型(一般/对称/Hermitian/三角)
   - 设置索引基(0-based或1-based)

4. **设置转置操作**
   - 根据 `transa` 参数确定是否需要转置A矩阵

5. **计算非零元素数量**
   ```c
   int nnz = viRow1stNz[nRowA] - 1; // 1-based
   ```

6. **设备内存分配**
   - 使用 `cudaMallocPitch` 为稠密矩阵B和C分配对齐的内存
   - 为CSR格式的稀疏矩阵A分配值、列索引和行指针数组

7. **数据拷贝到设备**
   - 使用 `cudaMemcpy2D` 拷贝稠密矩阵B和C(考虑leading dimension)
   - 使用 `cudaMemcpy` 拷贝稀疏矩阵A的数据

8. **执行稀疏矩阵乘法**
   ```c
   CHECK_CUSPARSE(cusparseDcsrmm(
       handle, trans,
       nRowA, nColC, nColA,
       nnz, &alpha,
       descrA,
       d_csrVal, d_csrRowPtr, d_csrColInd,
       d_B, culdb,  // A矩阵指针及leading dimension
       &beta,
       d_C, culdc   // C矩阵指针及leading dimension
   ));
   ```
   - 计算: C = alpha * op(A) * B + beta * C
   - `op(A)` 根据 `trans` 参数可能是A、A^T或A^H

9. **结果拷贝回主机**
   ```c
   cudaMemcpy2D(C + iRowOff_C + jColOff_C * ldc, ldc * sizeof(double), 
                d_C, pitch_C, nRowA * sizeof(double), nColC, cudaMemcpyDeviceToHost);
   ```

10. **资源释放**
    - 释放所有分配的GPU内存
    - 销毁CUSPARSE句柄和矩阵描述符

### 关键点说明

1. **索引处理**:
   - 函数开始和结束时的索引调整(+1/-1)表明该函数可能设计用于1-based索引系统
   - 但CUSPARSE默认使用0-based索引，所以需要调整

2. **矩阵描述符**:
   - `matdescra` 是4字符的字符串，描述矩阵属性:
     - 第1个字符: 矩阵类型(G/S/H/T)
     - 第3个字符: 索引基(C/F表示0-based/1-based)

3. **内存分配**:
   - 使用 `cudaMallocPitch` 为稠密矩阵分配内存，确保内存对齐
   - `pitch` 是实际分配的行字节数(可能大于需要的)

4. **cusparseDcsrmm参数**:
   - 这是CUSPARSE的旧版API(建议使用新API `cusparseSpMM`)
   - 执行广义稀疏矩阵-稠密矩阵乘法

5. **子矩阵支持**:
   - 通过 `iRowOff_B`, `jColOff_B` 等参数支持处理大矩阵的子矩阵
   - 但当前实现中A矩阵的偏移参数未被使用

### 改进建议

1. 考虑迁移到CUSPARSE的新API(`cusparseSpMM`)，它更高效且更灵活
2. 可以移除未使用的偏移参数(iRowOff_A, jColOff_A)
3. 索引调整逻辑可以更清晰地表达其目的
4. 添加更详细的错误检查和注释

这个函数提供了一个完整的CSR格式稀疏矩阵与稠密矩阵乘法的实现，支持各种矩阵类型和转置操作，适合集成到更大的数值计算应用中。