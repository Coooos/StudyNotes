[有关CUBLAS中的矩阵乘法函数 - 爨爨爨好 - 博客园](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)


---


`cublasDgemm` 是 cuBLAS 库中的一个函数，用于执行双精度浮点数（`double` 类型）的矩阵乘法。它实现了以下数学操作：

C=α⋅op(A)⋅op(B)+β⋅C

其中：
- op(A) 和 op(B) 可以是矩阵 A 和 B 的转置或不转置。
- α 和 β 是标量。
- C 是结果矩阵。

``### 函数原型

cpp

复制

```cpp
  cublasSgemm(
          handle,
          CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
          CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
          B_COL,          //矩阵B^T、C^T的行数
          A_ROW,          //矩阵A^T、C^T的列数
          B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
          &a,             //alpha的值
          d_B,            //左矩阵，为B^T
          B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
          d_A,            //右矩阵，为A^T
          A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
          &b,             //beta的值
          d_C,            //结果矩阵C
          B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
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