#include "../common.cuh"

void initData(float* addr, int stCount)
{
    for (size_t i = 0; i < stCount; i++)
    {
        addr[i] = (float)i;
    }
}

__global__ void addByGPU(float* A, float* B, float* C)
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int global_id = thread_id + block_id * blockDim.x;

    C[global_id] = A[global_id] + B[global_id];
}

int main()
{
    setGPU();

    size_t stCount = 512;                          // 元素数量
    size_t stBytesCount = stCount * sizeof(float); // 每一个元素为float 计算总字节数

    // 分配 CPU内存/主机内存
    float* fpHost_A = nullptr;
    float* fpHost_B = nullptr;
    float* fpHost_C = nullptr;
    fpHost_A = (float*)malloc(stBytesCount);
    fpHost_B = (float*)malloc(stBytesCount);
    fpHost_C = (float*)malloc(stBytesCount);

    if (fpHost_A != nullptr && fpHost_B != nullptr && fpHost_C != nullptr)
    {
        memset(fpHost_A, 0, stBytesCount); // 按字节填充
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
        printf("[INFO] CPU内存分配成功！\n");
    }
    else
    {
        printf("[ERROR] CPU内存分配失败！\n");
    }

    // 分配 GPU显存/设备内存
    float* fpDevice_A = nullptr;
    float* fpDevice_B = nullptr;
    float* fpDevice_C = nullptr;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);

    if (fpDevice_A != nullptr && fpDevice_B != nullptr && fpDevice_C != nullptr)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
        printf("[INFO] GPU显存分配成功！\n");
    }
    else
    {
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        printf("[ERROR] GPU显存分配失败！\n");
    } 
    
    // 初始化CPU内存数据
    initData(fpHost_A, stCount);
    initData(fpHost_B, stCount);

    // CPU数据复制到GPU
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);

    // 调用核函数在GPU进行计算
    dim3 gridSize(stCount / 32);
    dim3 blockSize(32);
    addByGPU<<<gridSize, blockSize>>>(fpDevice_A, fpDevice_B, fpDevice_C);
    // cudaDeviceSynchronize();

    // 含隐式的同步
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);
    
    printf("[INFO] GPU加法结果: %f \n", fpHost_C[2]);

    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);
    cudaDeviceReset();
}