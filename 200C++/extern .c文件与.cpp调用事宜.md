## 1.C++与C编译区别

在C++中常在头文件见到extern "C"修饰函数，那有什么作用呢？ 是用于C++链接在C语言模块中定义的函数。

C++虽然兼容C，但C++文件中函数编译后生成的符号与C语言生成的不同。因为C++支持函数重载，C++函数编译后生成的符时带有函数参数类型的信息，而C则没有。

例如`int add(int a, int b)`函数经过C++编译器生成.o文件后，`add`会变成形如`add_int_int`之类的, 而C的话则会是形如`_add`, 就是说：相同的函数，在C和C++中，编译后生成的符号不同。

这就导致一个问题：如果C++中使用C语言实现的函数，在编译链接的时候，会出错，提示找不到对应的符号。此时`extern "C"`就起作用了：告诉链接器去寻找`_add`这类的C语言符号，而不是经过C++修饰的符号。

## 2.C++调用C函数

C++调用C函数的例子: 引用C的头文件时，需要加`extern "C"`

1. `//add.h`
2. `#ifndef ADD_H`
3. `#define ADD_H`
4. `int add(int x,int y);`
5. `#endif`

6. `//add.c`
7. `#include "add.h"`

8. `int add(int x,int y) {`
9.     `return x+y;`
10. `}`

11. `//add.cpp`
12. `#include <iostream>`
13. `#include "add.h"`
14. `using namespace std;`
15. `int main() {`
16.     `add(2,3);`
17.     `return 0;`
18. `}`

编译：

1. `//Generate add.o file`
2. `gcc -c add.c`

链接：

1. `g++ add.cpp add.o -o main`

没有添加extern "C" 报错：

1. `> g++ add.cpp add.o -o main`                                   
2. `add.o：在函数‘main’中：`
3. ``add.cpp:(.text+0x0): `main'被多次定义``
4. `/tmp/ccH65yQF.o:add.cpp:(.text+0x0)：第一次在此定义`
5. `/tmp/ccH65yQF.o：在函数‘main’中：`
6. `add.cpp:(.text+0xf)：对‘add(int, int)’未定义的引用`
7. `add.o：在函数‘main’中：`
8. `add.cpp:(.text+0xf)：对‘add(int, int)’未定义的引用`
9. `collect2: error: ld returned 1 exit status`

添加extern "C"后：

`add.cpp`

1. `#include <iostream>`
2. `using namespace std;`
3. `extern "C" {`
4.     `#include "add.h"`
5. `}`
6. `int main() {`
7.     `add(2,3);`
8.     `return 0;`
9. `}`

编译的时候一定要注意，先通过gcc生成中间文件add.o。

1. `gcc -c add.c` 

然后编译：

1. `g++ add.cpp add.o -o main`

上述案例源代码见：

- [add.h](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c++/add.h)
    
- [add.c](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c++/add.c)
    
- [add.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c++/add.cpp)
    

## 2.C中调用C++函数

`extern "C"`在C中是语法错误，需要放在C++头文件中。

1. `// add.h`
2. `#ifndef ADD_H`
3. `#define ADD_H`
4. `extern "C" {`
5.     `int add(int x,int y);`
6. `}`
7. `#endif`

8. `// add.cpp`
9. `#include "add.h"`

10. `int add(int x,int y) {`
11.     `return x+y;`
12. `}`

13. `// add.c`
14. `extern int add(int x,int y);`
15. `int main() {`
16.     `add(2,3);`
17.     `return 0;`
18. `}`

编译：

1. `g++ -c add.cpp`

链接：

1. `gcc add.c add.o -o main`

上述案例源代码见：

- [add.h](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c/add.h)
    
- [add.c](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c/add.c)
    
- [add.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/extern/extern_c/add.cpp)
    

综上，总结出使用方法，在C语言的头文件中，对其外部函数只能指定为extern类型，C语言中不支持extern "C"声明，在.c文件中包含了extern "C"时会出现编译语法错误。所以使用extern "C"全部都放在于cpp程序相关文件或其头文件中。

总结出如下形式：

（1）C++调用C函数：

1. `//xx.h`
2. `extern int add(...)`

3. `//xx.c`
4. `int add(){`

5. `}`

6. `//xx.cpp`
7. `extern "C" {`
8.     `#include "xx.h"`
9. `}`

（2）C调用C++函数

 复制代码

1. `//xx.h`
2. `extern "C"{`
3.     `int add();`
4. `}`
5. `//xx.cpp`
6. `int add(){`

7. `}`
8. `//xx.c`
9. `extern int add();`