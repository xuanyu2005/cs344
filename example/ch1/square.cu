#include <stdio.h>
#include <chrono>
#include <vector>

using namespace std;

const int N = 1000000;
const int BLOCKS = 8;
typedef float tt;

__global__ void square(float *d_in, float *d_out)
{
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f * f;
}

void square_cpu(float *h_in, float *h_out)
{
    for (int i = 0; i < N; i++)
    {
        h_out[i] = h_in[i] * h_in[i] * h_in[i];
    }
}

int main()
{
    printf("size of array is %d \n", N);
    const int ARRAY_SIZE = N * sizeof(tt);
    tt h_in[N];
    for (int i = 0; i < N; i++)
    {
        h_in[i] = static_cast<tt>(i);
    }
    tt h_out[N];

    float *d_in;
    float *d_out;

    cudaMalloc((void **)&d_in, ARRAY_SIZE);
    cudaMalloc((void **)&d_out, ARRAY_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice);

    // =====================================
    cudaEventRecord(start);

    square<<<BLOCKS, N / BLOCKS>>>(d_in, d_out);

    cudaEventRecord(end);
    // =====================================

    cudaMemcpy(h_out, d_out, ARRAY_SIZE, cudaMemcpyDeviceToHost);

    float miliseconds = 0.0;
    cudaEventElapsedTime(&miliseconds, start, end);

    printf("CUDA: Kernel execution time: %f ms\n\n", miliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_in);
    cudaFree(d_out);

    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f \t", h_out[i]);
    //     if (i % 5 == 0)
    //         printf("\n");
    // }

    // 对比CPU
    auto start_cpu = chrono::high_resolution_clock::now();

    square_cpu(h_in, h_out);

    auto end_cpu = chrono::high_resolution_clock::now();

    auto milisecond_cpu = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu);

    printf("CPU execution time: %f ms\n\n", milisecond_cpu.count() / 1.0);

    return 0;
}