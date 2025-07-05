### 课程内容
#### 重要代码块
文件名以 .cu 结尾
命名规则：host 以 “h_” 开头，device 以 “d_” 开头
``` C++
cudaMalloc((void **)&d_in, ARRAY_SIZE);

cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice); //(destination , src , size , 三种类型 hosttodevice，devicetohost，devicetodevice)

// 内核函数 square 是全局函数
square<<<BLOCKS_NUM,THREADS_NUM_PERBLOCK>>>(d_in, d_out);//<<<>>>中间的通常是dim结构

cudaFree(d_in);
```
重点在根据不同任务设计每个程序块和线程的规模

#### 作业 彩图变黑白
##### TODO1代码
```C++
__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage,
                                  int numRows, int numCols)
{
  // TODO
  // Fill in the kernel to convert from color to greyscale
  // the mapping from components of a uchar4 to RGBA is:
  //  .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  // The output (greyImage) at each pixel should be the result of
  // applying the formula: output = .299f * R + .587f * G + .114f * B;
  // Note: We will be ignoring the alpha channel for this conversion

  // First create a mapping from the 2D block and grid locations
  // to an absolute 2D location in the image, then use that to
  // calculate a 1D offset

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < numRows && col < numCols)
  {
    int pix_id = col * numRows + row;

    uchar4 temp = rgbaImage[pix_id];

    float I = 0.299 * temp.x + 0.587 * temp.y + 0.114 * temp.z;

    greyImage[pix_id] = (unsigned char)I;
  }
}
```
图片在内存中用一维数组表示，前两行计算图片全局的行列，后面判断超出图片范围不做处理
CUDA 提供了内置变量来确定一个线程在整个网格（Grid）中的位置：
*   `blockIdx.x`, `blockIdx.y`: 线程块（Block）在网格（Grid）中的二维索引。（可以是三维）
*   `threadIdx.x`, `threadIdx.y`: 线程（Thread）在其所属线程块中的二维索引。
*   `blockDim.x`, `blockDim.y`: 一个线程块的维度（即每个块里有多少线程）。
##### TODO2代码
```C++
const dim3 blockSize(16, 16, 1); // TODO

  // 向上取整技巧： cell（x/y） = (x + y - 1) / y
  int gridX = (numRows + blockSize.x - 1) / blockSize.x;
  int gridY = (numCols + blockSize.y - 1) / blockSize.y;
  const dim3 gridSize(gridX, gridY, 1); // TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
```
`blockSize` 定义了每个线程块中有多少个线程。
*   对于图像处理这样的二维问题，使用二维的 `blockSize` 是最自然和高效的。
*   一个线程块中的线程总数通常是 32 的倍数（因为 GPU 的一个 warp 包含 32 个线程）。
*   常见的高效尺寸是 `16x16` (256 个线程) 或 `32x32` (1024 个线程，这是很多设备支持的上限),选择 `16x16`，这是一个非常安全和通用的选择。

```c++
const dim3 blockSize(16, 16, 1);
```
这里的 `1` 表示 z 维度只有一个线程，因为我们处理的是 2D 图像。
`gridSize` 定义了整个网格中有多少个线程块。我们的目标是启动足够多的线程块，以覆盖整个图像。
*   `gridSize.x` (x方向的块数) = `ceil(图像总列数 / 每个块的列数)`
*   `gridSize.y` (y方向的块数) = `ceil(图像总行数 / 每个块的行数)`
```c++
const dim3 gridSize(gridX, gridY, 1);
```

##### 编译执行
```
mkdir build
cd build
cmake ..
make
```
此时会生成一个HW1的可执行文件
```
./HW1 JPG_PATH
```
完成