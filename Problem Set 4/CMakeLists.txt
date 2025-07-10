############################################################################
# <summary> 现代化的 CMakeLists.txt 用于 CUDA 和 OpenCV. </summary>
# <date>    2024-05-21 (最终完整版) </date>
############################################################################

# 1. 指定 CMake 最低版本
cmake_minimum_required(VERSION 3.18)

# 2. 声明项目名称和语言
project(Homework3 LANGUAGES CXX C CUDA)

# 3. 显式地寻找所有依赖包
#    我们不仅需要 CUDA 工具包，还需要 OpenCV
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)      # <--- 这是新增的一行

# 4. 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 5. 收集源文件
file(GLOB hdr *.hpp *.h)
file(GLOB cu *.cu)
set(HW4_files main.cpp reference_calc.cpp compare.cpp loadSaveImage.cpp)

# 6. 创建可执行文件
add_executable(HW4
    ${HW4_files}
    ${cu}
)

# 7. 将我们的目标链接到所有需要的库
#    我们不仅要链接 CUDA 工具包，还要链接 OpenCV
target_link_libraries(HW4 PRIVATE 
    CUDA::toolkit
    ${OpenCV_LIBS}      # <--- 这是新增的一行
)