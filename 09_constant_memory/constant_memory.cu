#include <cuda_runtime.h>
#include <stdio.h>

// TODO 1：声明两个常量内存变量 c_k 和 c_b
__constant__ int c_k;
__constant__ int c_b;

__global__ void transform(int* A, int* B)
{
    // TODO 2：计算全局线程下标
    size_t thread_index = threadIdx.x;
    size_t global_index = blockIdx.x * blockDim.x + thread_index;

    // TODO 3：读取常量内存，完成计算
    B[global_index] = c_k * A[global_index] + c_b;
}

int main()
{
    int A[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int B[8];

    int k = 2;
    int b = 3;

    int* d_A;
    int* d_B;

    cudaMalloc(&d_A, sizeof(A));
    cudaMalloc(&d_B, sizeof(B));

    cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);

    // TODO 4：将 k 和 b 复制到常量内存
    cudaMemcpyToSymbol(c_k, &k, sizeof(int));
    cudaMemcpyToSymbol(c_b, &b, sizeof(int));

    // TODO 5：启动核函数
    transform<<<2, 4>>>(d_A, d_B);

    cudaMemcpy(B, d_B, sizeof(B), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++)
        printf("%d ", B[i]);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}