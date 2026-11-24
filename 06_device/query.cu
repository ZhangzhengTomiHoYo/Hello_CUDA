#include "../common.cuh"
#include <cstdio>

int main()
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop{};
    ErrorCheck(
        cudaGetDeviceProperties(&prop, device_id),
        __FILE__, __LINE__);

    printf("Device id: %d\n", device_id);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    printf("Global memory: %.2f GiB\n",
           prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Constant memory: %.2f KiB\n",
           prop.totalConstMem / 1024.0);

    printf("Maximum grid size: %d %d %d\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);

    printf("Maximum block dimensions: %d %d %d\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);

    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Shared memory per block: %.2f KiB\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("Shared memory per SM: %.2f KiB\n",
           prop.sharedMemPerMultiprocessor / 1024.0);

    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);

    return 0;
}