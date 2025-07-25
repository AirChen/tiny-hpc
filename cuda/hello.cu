#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel()
{
    printf("hello world\n");
}

int main()
{
    kernel<<<10,10>>>();
    cudaDeviceSynchronize();

    return 0;
}