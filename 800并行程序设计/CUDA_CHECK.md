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