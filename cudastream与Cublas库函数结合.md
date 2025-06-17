以下是一个使用CUDA Stream和cuBLAS库实现的分块矩阵乘法示例代码。该代码将大矩阵分割成小块，利用多个CUDA流并行处理数据传输和计算，从而隐藏内存传输延迟并提高GPU利用率。

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <stdexcept>

#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
}

#define CHECK_CUBLAS(func) \
{ \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        throw std::runtime_error("cuBLAS error"); \
    } \
}

void blockSgemm(int m, int n, int k, 
                const float* A, const float* B, float* C,
                int blockRow, int blockCol, int blockK,
                int numStreams) {
    // 创建cuBLAS句柄和流
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // 计算分块数量
    int blockRows = (m + blockRow - 1) / blockRow;
    int blockCols = (n + blockCol - 1) / blockCol;
    int blockKs = (k + blockK - 1) / blockK;

    // 为每个流分配设备内存
    std::vector<float*> d_A(numStreams, nullptr);
    std::vector<float*> d_B(numStreams, nullptr);
    std::vector<float*> d_C(numStreams, nullptr);

    size_t A_size = blockRow * blockK * sizeof(float);
    size_t B_size = blockK * blockCol * sizeof(float);
    size_t C_size = blockRow * blockCol * sizeof(float);

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaMalloc(&d_A[i], A_size));
        CHECK_CUDA(cudaMalloc(&d_B[i], B_size));
        CHECK_CUDA(cudaMalloc(&d_C[i], C_size));
    }

    // 分块矩阵乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int br = 0; br < blockRows; ++br) {
        int rowStart = br * blockRow;
        int rowSize = std::min(blockRow, m - rowStart);

        for (int bc = 0; bc < blockCols; ++bc) {
            int colStart = bc * blockCol;
            int colSize = std::min(blockCol, n - colStart);

            // 初始化当前结果块为0
            CHECK_CUDA(cudaMemsetAsync(
                d_C[bc % numStreams], 0, 
                rowSize * colSize * sizeof(float),
                streams[bc % numStreams]
            ));

            for (int bk = 0; bk < blockKs; ++bk) {
                int kStart = bk * blockK;
                int kSize = std::min(blockK, k - kStart);
                int streamId = bk % numStreams;

                // 设置当前流
                CHECK_CUBLAS(cublasSetStream(handle, streams[streamId]));

                // 异步复制A块
                const float* A_block = A + rowStart * k + kStart;
                CHECK_CUDA(cudaMemcpy2DAsync(
                    d_A[streamId], rowSize * sizeof(float),
                    A_block, k * sizeof(float),
                    rowSize * sizeof(float), kSize,
                    cudaMemcpyHostToDevice, streams[streamId]
                ));

                // 异步复制B块
                const float* B_block = B + kStart * n + colStart;
                CHECK_CUDA(cudaMemcpy2DAsync(
                    d_B[streamId], kSize * sizeof(float),
                    B_block, n * sizeof(float),
                    kSize * sizeof(float), colSize,
                    cudaMemcpyHostToDevice, streams[streamId]
                ));

                // 执行分块矩阵乘法
                CHECK_CUBLAS(cublasSgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    colSize, rowSize, kSize,
                    &alpha,
                    d_B[streamId], colSize,
                    d_A[streamId], kSize,
                    &beta,
                    d_C[streamId], colSize
                ));
            }
        }

        // 复制结果块回主机
        for (int bc = 0; bc < blockCols; ++bc) {
            int colStart = bc * blockCol;
            int colSize = std::min(blockCol, n - colStart);
            int streamId = bc % numStreams;

            float* C_block = C + rowStart * n + colStart;
            CHECK_CUDA(cudaMemcpy2DAsync(
                C_block, n * sizeof(float),
                d_C[streamId], colSize * sizeof(float),
                colSize * sizeof(float), rowSize,
                cudaMemcpyDeviceToHost, streams[streamId]
            ));
        }
    }

    // 同步所有流
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // 清理资源
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    // 矩阵尺寸 (示例)
    int m = 1024, n = 1024, k = 1024;
    int blockRow = 256, blockCol = 256, blockK = 256;
    int numStreams = 4;

    // 分配主机内存
    size_t A_size = m * k * sizeof(float);
    size_t B_size = k * n * sizeof(float);
    size_t C_size = m * n * sizeof(float);

    float* h_A = (float*)malloc(A_size);
    float* h_B = (float*)malloc(B_size);
    float* h_C = (float*)malloc(C_size);

    // 初始化矩阵 (这里简化为示例)
    for (int i = 0; i < m * k; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) h_B[i] = 1.0f;

    try {
        blockSgemm(m, n, k, h_A, h_B, h_C, 
                  blockRow, blockCol, blockK, numStreams);
        std::cout << "Matrix multiplication completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // 清理主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### 关键实现说明：

1. **分块策略**：
   - 将大矩阵划分为小块（`blockRow×blockK` 和 `blockK×blockCol`）
   - 结果矩阵块大小为 `blockRow×blockCol`
   - 通过三重循环遍历所有分块组合

2. **流并行化**：
   - 创建多个CUDA流（`numStreams`）
   - 每个流独立处理：
     - 设备内存分配
     - 主机到设备的数据传输
     - cuBLAS矩阵乘法
     - 设备到主机的数据传输

3. **内存管理**：
   - 每个流有自己的设备内存缓冲区
   - 使用`cudaMemcpy2DAsync`处理非连续内存
   - 结果块初始化为0后进行累加

4. **执行流程**：
   - 外层循环：遍历结果矩阵的行块
   - 中层循环：遍历结果矩阵的列块
   - 内层循环：累加K维度的分块乘积
   - 异步操作通过不同流并行执行

5. **边界处理**：
   - 处理矩阵尺寸不能被整除的情况
   - 动态调整每个分块的实际大小

### 性能优化建议：

1. **分块尺寸选择**：
   - 典型值：128×128 或 256×256
   - 需根据GPU架构调整（如A100/V100）
   - 使用`cudaOccupancyMaxPotentialBlockSize`辅助选择

2. **流数量优化**：
   - 通常4-8个流可获得最佳性能
   - 避免创建过多流导致资源竞争

3. **内存复用**：
   - 对大矩阵使用固定内存（`cudaMallocHost`）
   - 考虑使用零拷贝内存或统一内存

4. **高级技巧**：
   - 使用`cublasGemmStridedBatchedEx`批量执行
   - 结合共享内存进一步优化（需编写kernel）
   - 使用Tensor Core（`CUBLAS_COMPUTE_32F_FAST_TF32`）

5. **异步执行**：
   - 主机在等待时执行其他任务
   - 使用事件（`cudaEvent_t`）进行精细控制

此实现充分利用了CUDA流和cuBLAS的异步特性，通过重叠数据传输和计算，显著提升大矩阵乘法的效率。实际应用中需根据具体硬件和矩阵尺寸调整分块参数。