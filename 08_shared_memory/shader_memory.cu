#include <stdio.h>

__global__ void reverse(int* A, int* B)
{
    __shared__ int buffer[4];

    int local_thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + local_thread_id;

    // 1. 将全局内存中的数据搬运到共享内存
    buffer[local_thread_id] = A[global_id];

    // 2. 线程块内同步
    __syncthreads();

    // 3. 从共享内存反向读取，写入全局内存
    B[global_id] = buffer[3-local_thread_id];
}

int main()
{
    int A[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int B[8];

    int* d_A;
    int* d_B;

    // 分配全局内存
    cudaMalloc(&d_A, sizeof(A));
    cudaMalloc(&d_B, sizeof(B));

    // CPU -> GPU
    cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);

    // GPU 内部计算
    reverse<<<2, 4>>>(d_A, d_B);

    // GPU -> CPU
    cudaMemcpy(B, d_B, sizeof(B), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++)
        printf("%d ", B[i]);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}