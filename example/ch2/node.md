### 课程内容
#### 并行处理模式
![alt text](<并行处理模式.png>)

#### GPU并行处理基本单元：SMs（流处理器）
- 每一个SM可以处理多个线程块
- 一个线程块只能在一个SM上处理
- 每一个SMs有独立的线程和共享内存
- GPU内存访问速率：local > share > global

#### CUDA解决线程冲突的方式：原语&同步
##### 原语
假定
```c++
const int THREADS_NUM = 1000000;
const int ARRAY_SIZE = 10;
const int BLOCK_WIDTH = 1000;
```
```c++
__global__ void imcrement_naive(int *arr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    idx = idx % ARRAY_SIZE;

    arr[idx] = arr[idx] + 1; // 会有冲突本质上等于两条命令 int temp = arr[idx] + 1; arr[idx] = temp
}
//输出数组元素远小于1000000
```
修改：
```c++
__global__ void imcrement_atomic(int *arr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    idx = idx % ARRAY_SIZE;

    atomicAdd(&arr[idx], 1);
}
// 输出均为1000000
```
##### 同步
同步一般分隔进程中的加载数据和计算函数两个部分，在作业中函数`gaussian_blur_shared`有所体现。

### 作业 模糊图片
TODO1: 核心操作 output.jpg像素采用input.jpg 周围像素加权平均，
> 为什么不直接求平均？ 直接平均生成的图片没那么平滑
```c++
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
  outputChannel[output_idx] = (unsigned char)(blurred_pix); // 
}
```
模糊图片采用filter卷积操作，其余处理同第一次作业类似。
TODO2： 分隔通道(RGBRGBRGB -> RRRGGGBBB)GPU处理块状数据快得多，减少IO时间
```c++
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
}
```
没啥好说的
TODO3: 处理函数汇总：分通道 - 同步 - 卷积 - 同步 - 合并通道 - 同步
```c++
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
  const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                      (numRows + blockSize.y - 1) / blockSize.y,
                      1);


  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  gaussian_blur_shared<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_shared<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur_shared<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  // gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  // gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  // gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
```
TODO4：最后别忘了释放\*d_\*内存
```c++
void cleanup()
{
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
```
值得一提：在HW1中的第四个维度`w`是透明度，本实验中设为255,（不透明）

### 优化作业代码
找了一张高清图片（原作业图片只有几十k）
在上面的`gaussian_blur`函数处理下耗时： 41.056255 msecs.
下面我们采用共享内存技术优化
```c++
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

  // 计算块级共享的 tile 在全局内存中的起始坐标，定位到filter的左上角
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
```
它与“朴素”版本的最大区别在于，它不是让每个线程都直接从慢速的**全局内存**（Global Memory）中读取其计算所需的全部像素，而是**协同地**将一大块图像（称为 "tile" 或“瓦片”）加载到快速的**共享内存**（Shared Memory）中，然后再从共享内存中进行计算。
上述处理函数运行时间：35.451904 msecs.
重点讲讲加载数据代码
```c++
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
    if (g_load_x < 0) g_load_x = 0;
    if (g_load_x >= numCols) g_load_x = numCols - 1;
    if (g_load_y < 0) g_load_y = 0;
    if (g_load_y >= numRows) g_load_y = numRows - 1;

    tile[r][c] = inputChannel[g_load_y * numCols + g_load_x];
  }
}
```
**块内跨步（Grid-Stride Loop / Block-Stride Loop）**加载模式
将一个比线程数更大的任务（加载整个 `tile`）分配给一个固定数量的线程（一个线程块），通过多次迭代，每次迭代中所有线程同时工作，处理任务的不同部分，直到整个任务完成。