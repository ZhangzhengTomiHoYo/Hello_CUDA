#pragma once
#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("[ERROR] CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void setGPU()
{
    int iDeviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&iDeviceCount);

    if (err != cudaSuccess || iDeviceCount == 0)
    {
        printf("[ERROR] No CUDA Device!");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("[INFO] The count of CUDA device is: %d \n", iDeviceCount);
    }
    
    const int iDev = 0;
    err = cudaSetDevice(iDev);
    if (err != cudaSuccess)
    {
        printf("[ERROR] Failed tp set GPU %d for computing! \n", iDev);
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("[INFO] Success to set GPU %d for computing! \n", iDev);
    }
}
