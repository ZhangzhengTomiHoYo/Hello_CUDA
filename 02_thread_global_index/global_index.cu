#include <stdio.h>

dim3 grid_size(2,2,1);
dim3 block_size(4,4,1);

__global__ void thread()
{
    const int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int global_thread_id = block_id * (blockDim.x * blockDim.y) + thread_id;
    printf("Hello CUDA! I'm thread %d of block %d (global thread id: %d)\n", thread_id, block_id, global_thread_id);
}

int main()
{
    thread<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
}