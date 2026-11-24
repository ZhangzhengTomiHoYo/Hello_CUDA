#include <stdio.h>

__global__ void hello_cuda()
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int global_thread_id = block_id * blockDim.x + thread_id;
    printf("Hello CUDA! I'm thread %d of block %d (global thread id: %d)\n", thread_id, block_id, global_thread_id);
}

int main()
{
    hello_cuda<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}