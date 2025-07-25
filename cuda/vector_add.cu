#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 核函数（GPU 上执行）
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 1024;
  int *a, *b, *c;       // 主机端指针
  int *d_a, *d_b, *d_c; // 设备端指针

  // 分配主机内存
  a = (int *)malloc(n * sizeof(int));
  b = (int *)malloc(n * sizeof(int));
  c = (int *)malloc(n * sizeof(int));

  // 初始化数据
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // 分配设备内存
  cudaMalloc(&d_a, n * sizeof(int));
  cudaMalloc(&d_b, n * sizeof(int));
  cudaMalloc(&d_c, n * sizeof(int));

  // 拷贝数据到设备
  cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

  // 启动核函数（1个块，每块256线程）
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

  // 拷贝结果回主机
  cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

  // 验证结果（打印前10个）
  for (int i = 0; i < 10; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  // 释放内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);

  return 0;
}
