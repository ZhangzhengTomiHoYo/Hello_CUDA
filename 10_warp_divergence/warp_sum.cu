__global__ void reduce(int* A, int* result)
{
    __shared__ int buffer[8];

    int tid = threadIdx.x;

    // TODO 1：将全局内存数据搬入共享内存
    buffer[tid] = A[tid];

    // TODO 2：线程同步
    __syncthreads();

    // TODO 3：使用循环进行并行规约
    // stride：4 -> 2 -> 1
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            buffer[tid] = buffer[tid] + buffer[stride + tid];
        }
        __syncthreads();
    }
    
    // TODO 4：由 Thread 0 将最终结果写回全局内存
    if (tid == 0)
    {
        *result = buffer[0];
    }
}