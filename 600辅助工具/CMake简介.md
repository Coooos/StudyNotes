# 📘 CMake 精华笔记

## 一、CMake 简介
- 跨平台构建工具，可生成 Makefile 或其他编译系统文件
- 比手动编写 Makefile 更简单，尤其适合大型项目

---

## 二、安装 CMake
```bash
sudo apt install cmake
cmake -version  # 查看版本
```

---

## 三、基本使用：CMakeLists.txt
### 基本结构：
```cmake
cmake_minimum_required(VERSION 2.8)
project(demo)
set(CMAKE_BUILD_TYPE "Debug")
add_executable(main main.cpp)
```

### 编译流程：
```bash
cmake .   # 生成 Makefile
make      # 编译生成可执行文件
make clean # 清理
```

---

## 四、多源文件编译
### 方法一：手动列出
```cmake
add_executable(main main.cpp test.cpp)
```

### 方法二：自动收集（可能包含不需要的文件）
```cmake
aux_source_directory(. SRC_LIST)
add_executable(main ${SRC_LIST})
```

### 方法三：手动指定文件列表
```cmake
set(SRC_LIST
    ./main.cpp
    ./test.cpp
)
add_executable(main ${SRC_LIST})
```

---

## 五、标准项目结构示例
```
project/
├── bin/          # 可执行文件
├── build/        # 编译中间文件
├── include/      # 头文件
├── src/          # 源文件
└── CMakeLists.txt
```

### CMakeLists.txt 示例：
```cmake
cmake_minimum_required(VERSION 2.8)
project(math)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(src SRC_LIST)
add_executable(main main.cpp ${SRC_LIST})
```

---

## 六、编译静态库/动态库
### 库目录结构：
```
project/
├── lib/          # 库文件
├── src/          # 库源码
├── test/         # 测试代码
└── CMakeLists.txt
```

### src/CMakeLists.txt（生成库）：
```cmake
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(sum SHARED sum.cpp)   # 动态库
add_library(minor STATIC minor.cpp) # 静态库
```

### test/CMakeLists.txt（链接库）：
```cmake
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(../include)
link_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable(main main.cpp)
target_link_libraries(main sum minor)
```

---

## 七、常用预定义变量
| 变量名 | 说明 |
|--------|------|
| `PROJECT_NAME` | 项目名称 |
| `PROJECT_SOURCE_DIR` | 工程根目录 |
| `PROJECT_BINARY_DIR` | 编译目录 |
| `EXECUTABLE_OUTPUT_PATH` | 可执行文件输出路径 |
| `LIBRARY_OUTPUT_PATH` | 库文件输出路径 |
| `CMAKE_BUILD_TYPE` | Debug / Release |
| `CMAKE_CXX_FLAGS` | C++ 编译选项 |
| `CMAKE_CXX_FLAGS_DEBUG` | Debug 模式编译选项 |

---

## 八、常用命令总结
- `cmake_minimum_required(VERSION x.x)`
- `project(name)`
- `set(var value)`
- `add_executable(target sources)`
- `add_library(libname SHARED/STATIC sources)`
- `target_link_libraries(target libs)`
- `include_directories(dir)`
- `link_directories(dir)`
- `add_subdirectory(dir)`



# 🛠️ 简易CMake项目示例

## 项目结构
```
simple_project/
├── CMakeLists.txt
├── include/
│   └── utils.h
├── src/
│   ├── utils.cpp
│   └── main.cpp
└── build/          # 编译目录（可选）
```

## 文件内容

### 1. 头文件
**include/utils.h**
```cpp
#pragma once

int add(int a, int b);
int multiply(int a, int b);
void print_message(const char* message);
```

### 2. 源文件
**src/utils.cpp**
```cpp
#include <iostream>
#include "../include/utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

void print_message(const char* message) {
    std::cout << "Message: " << message << std::endl;
}
```

**src/main.cpp**
```cpp
#include <iostream>
#include "utils.h"

int main() {
    std::cout << "Simple CMake Project" << std::endl;
    
    int a = 10, b = 5;
    std::cout << a << " + " << b << " = " << add(a, b) << std::endl;
    std::cout << a << " * " << b << " = " << multiply(a, b) << std::endl;
    
    print_message("Hello from CMake!");
    
    return 0;
}
```

### 3. CMakeLists.txt（简化版）
```cmake
# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(SimpleProject VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 添加头文件目录
include_directories(include)

# 收集所有源文件
file(GLOB SOURCES "src/*.cpp")

# 创建可执行文件
add_executable(simple_app ${SOURCES})
```

## 构建步骤

### 方法一：使用build目录（推荐）
```bash
# 进入项目目录
cd simple_project

# 创建build目录
mkdir build
cd build

# 生成Makefile
cmake ..

# 编译项目
make

# 运行程序
./simple_app
```

### 方法二：直接在当前目录构建
```bash
cd simple_project
cmake .
make
./simple_app
```

## 预期输出
```
Simple CMake Project
10 + 5 = 15
10 * 5 = 50
Message: Hello from CMake!
```

## 进阶功能（可选添加）

### 1. 添加调试信息
在CMakeLists.txt中添加：
```cmake
# 设置编译模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# 调试模式添加-g标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
else()
    add_compile_options(-O2)
endif()
```

### 2. 添加安装规则
在CMakeLists.txt末尾添加：
```cmake
# 安装可执行文件到系统目录
install(TARGETS simple_app
    RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
```

然后可以运行：
```bash
sudo make install  # 安装到系统
```

### 3. 清理构建文件
```bash
make clean        # 清理编译文件
rm -rf build/*    # 彻底清理build目录（如果使用build目录）
```

## 最简版本（极简主义）

如果你想要更简单的版本：

**CMakeLists.txt（最简版）**
```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleApp)

# 直接指定所有源文件
add_executable(simple_app
    src/main.cpp
    src/utils.cpp
)

# 添加头文件路径
target_include_directories(simple_app PRIVATE include)
```

这个简单项目包含了CMake的基本要素：
- 多文件编译
- 头文件包含
- 可执行文件生成
- 简单的构建流程

适合初学者学习和快速上手！