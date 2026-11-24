#include <stdio.h>

dim3 gridSize(2,2,2);
dim3 blockSize(2,2,2);

__global__ void d3()
{

    const int block_id = blockIdx.x
                       + blockIdx.y * gridDim.x
                       + blockIdx.z * gridDim.x * gridDim.y;

    const int thread_id = threadIdx.x 
                        + threadIdx.y * blockDim.x
                        + threadIdx.z * blockDim.x * blockDim.y;

    const int global_thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z) + thread_id;

    printf("Hello CUDA! I'm thread %d of block %d (global thread id: %d)\n", thread_id, block_id, global_thread_id);
}

int main()
{
    d3<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
}