/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void histogram_kernel(const float *const logLuminance,
                                 unsigned int *histogram,
                                 float min_logLum,
                                 int numbins,
                                 int numPix,
                                 float LumRange)
{
   int idx = blockDim.x * blockIdx.x + threadIdx.x;

   if (idx < numPix)
   {
      float Lum = logLuminance[idx];

      unsigned int bin = static_cast<unsigned int>(Lum - min_logLum) / LumRange * (numbins - 1.f);
      atomicAdd(&histogram[bin], 1);
   }
}

// 核函数 1: 部分归约
// 每个线程块将输入数组的一大块归约为一个 min/max 对
__global__ void min_max_partial_reduction_kernel(const float *const input, float2 *partial_results, const size_t numPixels)
{
   // 为这个块的 min/max 值分配共享内存
   extern __shared__ float s_data[];
   float *s_min = s_data;
   float *s_max = &s_data[blockDim.x];

   const unsigned int tid = threadIdx.x;
   const unsigned int global_idx_start = blockIdx.x * blockDim.x * 2;

   // 初始化线程的局部 min/max
   float local_min = 1e20f;  // A large number
   float local_max = -1e20f; // A small number

   // 每个线程处理两个元素以增加效率 (strided loop)
   for (unsigned int i = global_idx_start + tid; i < numPixels; i += blockDim.x * 2)
   {
      float val1 = input[i];
      local_min = fminf(local_min, val1);
      local_max = fmaxf(local_max, val1);

      if (i + blockDim.x < numPixels)
      {
         float val2 = input[i + blockDim.x];
         local_min = fminf(local_min, val2);
         local_max = fmaxf(local_max, val2);
      }
   }

   s_min[tid] = local_min;
   s_max[tid] = local_max;
   __syncthreads();

   // 在共享内存中进行块内归约
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if (tid < s)
      {
         s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
         s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
      }
      __syncthreads();
   }

   // 块的第一个线程将此块的结果写入全局内存
   if (tid == 0)
   {
      partial_results[blockIdx.x].x = s_min[0];
      partial_results[blockIdx.x].y = s_max[0];
   }
}

// 核函数 2: 最终归约
// 使用单个块来归约部分结果，得到最终的 min/max
// __global__ void min_max_final_reduction_kernel(float2 *partial_results, const int num_partials)
// {
//    extern __shared__ float s_data[];
//    float *s_min = s_data;
//    float *s_max = &s_data[blockDim.x];

//    const unsigned int tid = threadIdx.x;

//    // 从全局内存加载部分结果到共享内存
//    if (tid < num_partials)
//    {
//       s_min[tid] = partial_results[tid].x;
//       s_max[tid] = partial_results[tid].y;
//    }
//    else
//    {
//       s_min[tid] = 1e20f;
//       s_max[tid] = -1e20f;
//    }
//    __syncthreads();

//    // 在共享内存中进行最终的归约
//    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//    {
//       if (tid < s)
//       {
//          s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
//          s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
//       }
//       __syncthreads();
//    }

//    // 第一个线程将最终结果写回全局内存的第一个位置
//    if (tid == 0)
//    {
//       partial_results[0].x = s_min[0];
//       partial_results[0].y = s_max[0];
//    }
// }

__global__ void min_max_final_reduction_kernel(float2 *partial_results, const int num_partials)
{
   // 为这个块的 min/max 值分配共享内存
   extern __shared__ float s_data[];
   float *s_min = s_data;
   float *s_max = &s_data[blockDim.x];

   const unsigned int tid = threadIdx.x;
   const unsigned int block_size = blockDim.x;

   // 1. 每个线程初始化自己的局部 min/max
   //    这是最关键的改动。我们不再直接写入共享内存，而是先在寄存器中累积。
   float local_min = 1e20f;  // A large number
   float local_max = -1e20f; // A small number

   // 2. 网格步长循环 (Grid-stride loop)
   //    每个线程从自己的 tid 开始，以 block_size 为步长，遍历所有部分结果。
   for (int i = tid; i < num_partials; i += block_size)
   {
      local_min = fminf(local_min, partial_results[i].x);
      local_max = fmaxf(local_max, partial_results[i].y);
   }

   // 3. 将每个线程的局部结果写入共享内存
   s_min[tid] = local_min;
   s_max[tid] = local_max;
   __syncthreads();

   // 4. 在共享内存中进行最终的块内归约 (这部分代码保持不变)
   for (unsigned int s = block_size / 2; s > 0; s >>= 1)
   {
      if (tid < s)
      {
         s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
         s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
      }
      __syncthreads();
   }

   // 5. 块的第一个线程将最终结果写回全局内存 (这部分代码保持不变)
   if (tid == 0)
   {
      partial_results[0].x = s_min[0];
      partial_results[0].y = s_max[0];
   }
}

// 单块排他性扫描核函数
__global__ void exclusive_scan_kernel(unsigned int *const d_out, const unsigned int *const d_in, const size_t n)
{
   // 动态分配共享内存，并使用新的名字
   extern __shared__ unsigned int s_scan_data[];

   const unsigned int tid = threadIdx.x;

   // 1. 将数据从全局内存加载到共享内存
   if (tid < n)
   {
      s_scan_data[tid] = d_in[tid];
   }
   else
   {
      s_scan_data[tid] = 0; // Padding
   }
   __syncthreads();

   // 2. 在共享内存中执行上扫(reduce)阶段
   for (unsigned int offset = 1; offset < n; offset *= 2)
   {
      unsigned int i = 2 * offset * (tid + 1) - 1;
      if (i < n)
      {
         s_scan_data[i] += s_scan_data[i - offset];
      }
      __syncthreads();
   }

   // 3. 将最后一个元素清零并执行下扫阶段
   if (tid == 0)
   {
      s_scan_data[n - 1] = 0;
   }
   __syncthreads();

   for (unsigned int offset = n / 2; offset > 0; offset /= 2)
   {
      unsigned int i = 2 * offset * (tid + 1) - 1;
      if (i < n)
      {
         unsigned int temp = s_scan_data[i - offset];
         s_scan_data[i - offset] = s_scan_data[i];
         s_scan_data[i] += temp;
      }
      __syncthreads();
   }

   // 4. 将结果从共享内存写回全局内存
   if (tid < n)
   {
      d_out[tid] = s_scan_data[tid];
   }
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   // TODO
   /*Here are the steps you need to implement
     1) find the minimum and maximum value in the input logLuminance channel
        store in min_logLum and max_logLum
     2) subtract them to find the range
     3) generate a histogram of all the values in the logLuminance channel using
        the formula: bin = (lum[i] - lumMin) / lumRange * numBins
     4) Perform an exclusive scan (prefix sum) on the histogram to get
        the cumulative distribution of luminance values (this should go in the
        incoming d_cdf pointer which already has been allocated for you)       */

   const size_t numPixels = numRows * numCols;

   // // --- 步骤 1 & 2: 找到 Min/Max 和 Range  ---

   // // a) 配置归约核函数的执行参数
   const int blockSize = 256;
   // const int gridSize = (numPixels + (blockSize * 2) - 1) / (blockSize * 2);

   // // b) 为部分归约结果分配设备内存
   // float2 *d_partial_results;
   // checkCudaErrors(cudaMalloc(&d_partial_results, gridSize * sizeof(float2)));

   // // c) 启动部分归约核函数
   // size_t sharedMemSize = 2 * blockSize * sizeof(float);
   // min_max_partial_reduction_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_logLuminance, d_partial_results, numPixels);
   // checkCudaErrors(cudaGetLastError());

   // // d) 启动最终归约核函数 (用一个块处理所有部分结果)
   // // 假设 gridSize <= blockSize, 这通常成立
   // min_max_final_reduction_kernel<<<1, blockSize, sharedMemSize>>>(d_partial_results, gridSize);
   // checkCudaErrors(cudaGetLastError());

   // // e) 将最终结果从设备拷贝回主机
   // float2 h_result;
   // checkCudaErrors(cudaMemcpy(&h_result, d_partial_results, sizeof(float2), cudaMemcpyDeviceToHost));

   // min_logLum = h_result.x;
   // max_logLum = h_result.y;

   // // f) 清理部分归约的临时内存
   // checkCudaErrors(cudaFree(d_partial_results));

   // const float lumRange = max_logLum - min_logLum;

   // 在你的主函数中，用以下代码替换整个min/max计算
   thrust::device_ptr<const float> d_logLum_ptr(d_logLuminance);

   // 使用 thrust::minmax_element 一步到位找到min和max
   thrust::pair<thrust::device_ptr<const float>, thrust::device_ptr<const float>> minmax_ptr;
   minmax_ptr = thrust::minmax_element(d_logLum_ptr, d_logLum_ptr + numPixels);

   // 从设备指针获取值
   // thrust::minmax_element返回指向最小/最大元素的指针，我们需要从中拷贝值
   checkCudaErrors(cudaMemcpy(&min_logLum, minmax_ptr.first.get(), sizeof(float), cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaMemcpy(&max_logLum, minmax_ptr.second.get(), sizeof(float), cudaMemcpyDeviceToHost));
   int lumRange = max_logLum - min_logLum;

   // --- 步骤 3: 生成直方图 ---

   // 为直方图分配设备内存并初始化为0
   unsigned int *d_histogram;
   checkCudaErrors(cudaMalloc(&d_histogram, numBins * sizeof(unsigned int)));
   checkCudaErrors(cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int)));

   // 配置并启动直方图核函数
   const int histoGridSize = (numPixels + blockSize - 1) / blockSize;
   histogram_kernel<<<histoGridSize, blockSize>>>(d_logLuminance, d_histogram, min_logLum, lumRange, numPixels, numBins);
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaDeviceSynchronize()); // 确保直方图完成

   // --- 步骤 4: 执行排他性扫描 ---

   // // 假设 numBins <= 1024 (一个块的最大线程数)
   // if (numBins > 1024)
   // {
   //    // 如果 numBins 太大，这个简化的扫描会失败。
   //    // 一个完整的实现需要一个多块扫描算法。
   //    // 对于此作业，此假设是合理的。
   //    printf("Error: numBins > 1024, this simplified scan does not support it.\n");
   //    return;
   // }

   // // 启动单块扫描核函数
   // // 线程块的大小需要是大于或等于 numBins 的最小的2的幂，以便于算法实现。
   // // 这里为简单起见，我们直接使用 numBins，并确保共享内存足够大。
   // size_t scanSharedMemSize = numBins * sizeof(unsigned int);
   // exclusive_scan_kernel<<<1, numBins, scanSharedMemSize>>>(d_cdf, d_histogram, numBins);
   try
   {
      // 将原始设备指针包装成 Thrust 的 device_ptr
      thrust::device_ptr<unsigned int> d_hist_ptr(d_histogram);
      thrust::device_ptr<unsigned int> d_cdf_ptr(d_cdf);

      // 执行排他性扫描 (前缀和)
      // thrust::exclusive_scan(输入开始, 输入结束, 输出开始);
      thrust::exclusive_scan(d_hist_ptr, d_hist_ptr + numBins, d_cdf_ptr);
   }
   catch (const thrust::system_error &e)
   {
      fprintf(stderr, "Thrust error: %s\n", e.what());
      return;
   }
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaDeviceSynchronize());

   // --- 清理 ---
   checkCudaErrors(cudaFree(d_histogram));
}
