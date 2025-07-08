#include <stdio.h>
using namespace std;

const int THREADS_NUM = 1000000;
const int ARRAY_SIZE = 10;
const int BLOCK_WIDTH = 1000;

__global__ void imcrement_naive(int *arr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    idx = idx % ARRAY_SIZE;

    arr[idx] = arr[idx] + 1;
}

__global__ void imcrement_atomic(int *arr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    idx = idx % ARRAY_SIZE;

    atomicAdd(&arr[idx], 1);
}

int main()
{
    int h_arr[ARRAY_SIZE] = {0};
    int *d_arr;

    cudaMalloc((void **)&d_arr, sizeof(int) * ARRAY_SIZE);
    cudaMemset((void *)d_arr, 0, ARRAY_SIZE * sizeof(int));

    cudaMemcpy(d_arr, h_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    imcrement_naive<<<THREADS_NUM / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_arr);

    cudaEventRecord(end);

    cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
    {
        printf("%d \t", h_arr[i]);
        if ((i + 1) % 5 == 0)
            printf("\n");
    }
    printf("\n");
    float miliseconds = 0.0;
    cudaEventElapsedTime(&miliseconds, start, end);

    printf("CUDA: Kernel execution time: %f ms\n\n", miliseconds);

    cudaFree(d_arr);
    return 0;
}