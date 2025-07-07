// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"
#define BLOCK_WIDTH 16
#define FILTER_WIDTH 9                             // HW2.cpp 硬编码
#define FILTER_RADIUS (FILTER_WIDTH / 2)           // -> 4
#define TILE_DIM (BLOCK_WIDTH + 2 * FILTER_RADIUS) // -> 16 + 8 = 24

__global__ void gaussian_blur_shared(const unsigned char *const inputChannel,
                                     unsigned char *const outputChannel,
                                     int numRows, int numCols,
                                     const float *const filter, const int filterWidth)
{
  // 声明共享内存
  __shared__ unsigned char tile[TILE_DIM][TILE_DIM];

  // 线程在块内的局部坐标
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 计算块级共享的 tile 在全局内存中的起始坐标
  const int tile_start_x = blockIdx.x * BLOCK_WIDTH - FILTER_RADIUS;
  const int tile_start_y = blockIdx.y * BLOCK_WIDTH - FILTER_RADIUS;

  // --- 1. 从全局内存协同加载数据到共享内存 tile ---
  for (int i = 0; i < TILE_DIM * TILE_DIM; i += blockDim.x * blockDim.y)
  {
    int idx = i + ty * blockDim.x + tx;
    if (idx < TILE_DIM * TILE_DIM)
    {
      int r = idx / TILE_DIM; // tile 内的行
      int c = idx % TILE_DIM; // tile 内的列

      // 计算要加载的全局内存坐标 (基于块的起始地址)
      int g_load_x = tile_start_x + c;
      int g_load_y = tile_start_y + r;

      // 边界钳位 (Clamping)
      if (g_load_x < 0)
        g_load_x = 0;
      if (g_load_x >= numCols)
        g_load_x = numCols - 1;
      if (g_load_y < 0)
        g_load_y = 0;
      if (g_load_y >= numRows)
        g_load_y = numRows - 1;

      tile[r][c] = inputChannel[g_load_y * numCols + g_load_x];
    }
  }

  // --- 2. 同步！确保所有线程都完成了加载 ---
  __syncthreads();

  // --- 3. 从共享内存进行计算 ---
  const int out_x = blockIdx.x * BLOCK_WIDTH + tx;
  const int out_y = blockIdx.y * BLOCK_WIDTH + ty;

  if (out_x < numCols && out_y < numRows)
  {
    float blurred_pix = 0.0f;
    for (int r = -FILTER_RADIUS; r <= FILTER_RADIUS; r++)
    {
      for (int c = -FILTER_RADIUS; c <= FILTER_RADIUS; c++)
      {
        // 读取共享内存中的数据，索引相对于线程在 tile 中的中心位置
        blurred_pix += tile[ty + FILTER_RADIUS + r][tx + FILTER_RADIUS + c] *
                       filter[(r + FILTER_RADIUS) * FILTER_WIDTH + (c + FILTER_RADIUS)];
      }
    }

    outputChannel[out_y * numCols + out_x] = (unsigned char)(blurred_pix);
    // outputChannel[out_y * numCols + out_x] = (unsigned char)(blurred_pix + 0.5f);
  }
}

__global__ void gaussian_blur(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel,
                              int numRows, int numCols,
                              const float *const filter, const int filterWidth)
{
  // TODO
  // 使用全局内存
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= numCols || y >= numRows)
  {
    return;
  }

  float blurred_pix = 0.0;

  int filterR = filterWidth / 2;

  for (int i = -filterR; i <= filterR; i++)
  {
    for (int j = -filterR; j <= filterR; j++)
    {
      int neighbor_x = x + i;
      int neighbor_y = y + j;

      if (neighbor_x < 0)
        neighbor_x = 0;
      if (neighbor_y < 0)
        neighbor_y = 0;
      if (neighbor_x >= numCols)
        neighbor_x = numCols - 1;
      if (neighbor_y >= numRows)
        neighbor_y = numRows - 1;

      int idx_pix = neighbor_y * numCols + neighbor_x;
      int filter_idx = (i + filterR) * filterWidth + (j + filterR);

      blurred_pix += (float)inputChannel[idx_pix] * filter[filter_idx];
    }
  }
  int output_idx = y * numCols + x;
  outputChannel[output_idx] = (unsigned char)(blurred_pix); // 默认直接截断
  // outputChannel[output_idx] = (unsigned char)(blurred_pix + 0.5f); // 四舍五入
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }

  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4 *const inputImageRGBA,
                                 int numRows,
                                 int numCols,
                                 unsigned char *const redChannel,
                                 unsigned char *const greenChannel,
                                 unsigned char *const blueChannel)
{
  // TODO
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= numCols || y >= numRows)
  {
    return;
  }

  int idx_1D = y * numCols + x;

  uchar4 RGB = inputImageRGBA[idx_1D];

  redChannel[idx_1D] = RGB.x;
  greenChannel[idx_1D] = RGB.y;
  blueChannel[idx_1D] = RGB.z;

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
}

// This kernel takes in three color channels and recombines them
// into one image.  The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char *const redChannel,
                                  const unsigned char *const greenChannel,
                                  const unsigned char *const blueChannel,
                                  uchar4 *const outputImageRGBA,
                                  int numRows,
                                  int numCols)
{
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // make sure we don't try and access memory outside the image
  // by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue = blueChannel[thread_1D_pos];

  // Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float *const h_filter, const size_t filterWidth)
{

  // allocate memory for the three different channels
  // original
  checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

  // TODO:
  // Allocate memory for the filter on the GPU
  // Use the pointer d_filter that we have already declared for you
  // You need to allocate memory for the filter with cudaMalloc
  // be sure to use checkCudaErrors like the above examples to
  // be able to tell if anything goes wrong
  // IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc

  checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float) * filterWidth * filterWidth));

  // TODO:
  // Copy the filter on the host (h_filter) to the memory you just allocated
  // on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  // Remember to use checkCudaErrors!

  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 *const h_inputImageRGBA, uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  // TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  // TODO:
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                      (numRows + blockSize.y - 1) / blockSize.y,
                      1);

  // TODO: Launch a kernel for separating the RGBA image into different color channels

  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur_shared<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_shared<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_shared<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  // gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  // gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  // gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup()
{
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
