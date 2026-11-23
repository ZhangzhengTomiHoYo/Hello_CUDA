#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello CUDA!");
}

int main()
{
    hello_cuda<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}