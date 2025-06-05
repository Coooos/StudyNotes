---
date: 
aliases: 
tags:
  - study
---
前提知识点参考：[[并行程序设计基础]][[]]


## 1预备知识

OpenMP是一个针对共享内存并行编程的API，其中MP表示“多处理”，具有使用简单，代码变动小等优点。使用需包含头文件：#include<omp.h>

OpenMP 提供“基于指令”的共享内存API 。

在C和C++中，预处理器指令以 #pragma 开头。

通常，我们把字符#放在第一列，并且像其他预处理器指令一样，移动指令的剩余部分，使它和剩下的代码对齐。

与所有的预处理器指令一样，pragma 的默认长度是一行，因此如果有一个pragma 在一行中放不下，那么新行需要被“转义”:前面加一个反斜杠"\"。

OpenMP 的pragma 总是以#pragma omp

每个OpenMP指令后是一个结构块（用大括号括起来）

## 2编译制导

编译制导指令以#pragma omp 开始，后边跟具体的功能指令，格式如：#pragma omp 指令[子句[,子句] …]。

- parallel：用在一个结构块之前，表示这段代码将被多个线程并行执行；
    
- parallel for：parallel和for指令的结合，也是用在for循环语句之前，表示for循环体的代码将被多个线程并行执行，它同时具有并行域的产生和任务分担两个功能；
    
- sections：用在可被并行执行的代码段之前，用于实现多个结构块语句的任务分担，可并行执行的代码段各自用section指令标出（注意区分sections和section）；
    
- parallel sections：parallel和sections两个语句的结合，类似于parallel for；
    
- single：用在并行域内，表示一段只被单个线程执行的代码；
    
- critical：用在一段代码临界区之前，保证每次只有一个OpenMP线程进入；
    
- flush：保证各个OpenMP线程的数据影像的一致性；
    
- barrier：用于并行域内代码的线程同步，线程执行到barrier时要停下等待，直到所有线程都执行到barrier时才继续往下执行；
    
- atomic：用于指定一个数据操作需要原子性地完成；
    
- master：用于指定一段代码由主线程执行；
    
- threadprivate：用于指定一个或多个变量是线程专用，后面会解释线程专有和私有的区别。
    

相应的OpenMP子句：

- private：指定一个或多个变量在每个线程中都有它自己的私有副本；
    
- shared：指定一个或多个变量为多个线程间的共享变量；
    
- firstprivate：指定一个或多个变量在每个线程都有它自己的私有副本，并且私有变量要在进入并行域或任务分担域时，继承主线程中的同名变量的值作为初值；
    
- lastprivate：是用来指定将线程中的一个或多个私有变量的值在并行处理结束后复制到主线程中的同名变量中，负责拷贝的线程是for或sections任务分担中的最后一个线程；
    
- reduction：用来指定一个或多个变量是私有的，并且在并行处理结束后这些变量要执行指定的归约运算，并将结果返回给主线程同名变量；
    
- nowait：指出并发线程可以忽略其他制导指令暗含的路障同步；
    
- num_threads：指定并行域内的线程的数目；
    
- schedule：指定for任务分担中的任务分配调度类型；
    
- ordered：用来指定for任务分担域内指定代码段需要按照串行循环次序执行；
    
- copyprivate：配合single指令，将指定线程的专有变量广播到并行域内其他线程的同名变量中；
    
- copyin：用来指定一个threadprivate类型的变量需要用主线程同名变量进行初始化；
    
- default：用来指定并行域内的变量的使用方式，缺省是shared。
    

## 3 OPENMP实现

线程被同一个进程派生(fork)，这些线程共享启动它们的进程的大部分资源（例如，对标准输入和标准输出的访问），但每个线程有它自己的栈和程序计数器。

当一个线程完成了执行，它就又合并(join) 到启动它的进程中。

在OpenMP 语法中，执行并行块的线程集合（原始的线程和新的线程）称为线程组(team) ，原始的线程称为主线程(master) ，额外的线程称为从线程(slave) 。

#pragma omp parallel num_threads(thread_cnunt)

检查预处理宏_OpenMP是否定义

### 梯形积分法

```
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Trap(double a, double b, int n, double* global_result_p);

int main(int argc, char* argv[]) {
   
   double  a, b;                 /* Left and right endpoints      */
   int     n;                    /* Total number of trapezoids    */
   int     thread_count;
   double  global_result = 0.0;  /* Store result in global_result */
	
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n");
   scanf("%lf %lf %d", &a, &b, &n);
#  pragma omp parallel num_threads(thread_count) Trap(a, b, n, &global_result);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n",
      a, b, global_result);
   return 0;
}  /* main */

void Trap(double a, double b, int n, double* global_result_p) {
   double  h, x, my_result;
   double  local_a, local_b;
   int  i, local_n;
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   h = (b-a)/n; 										/*梯形的长度*/
   local_n = n/thread_count;							/*给每个线程分配到的梯形数*/  
   local_a = a + my_rank*local_n*h; 					/*区间的左右端点*/
   local_b = local_a + local_n*h; 
   my_result = (f(local_a) + f(local_b))/2.0; 
   for (i = 1; i <= local_n-1; i++) {
     x = local_a + i*h;
     my_result += f(x);
   }
   my_result = my_result*h; 

#  pragma omp critical *global_result_p += my_result;
}  /* Trap */
```



### 变量作用域

一个能够被线程组中的所有线程访问的变量拥有共享作用域，而一个只能被单个线程访问的变量拥有私有作用域。

在parallel 块之前被声明的变量的缺省作用域是共享的

OpenMP 有两种锁：简单(simple)锁和嵌套(nested)锁。

- 简单锁在被释放前只能获得一次，一个嵌套锁在被释放前可以被同一个线程获得多次。
    
- 定义简单锁的函数：
    

critical 指令、atomic 指令、锁的比较

- 一般而言， atomic 指令是实现互斥访问最快的方法。
    
- 程序中有多个不同的由atomic 指令保护的临界区，则应当使用命名的critical 指令或者锁。
    
- 使用critical 指令保护临界区与使用锁保护临界区在性能上没有太大的差别。
    
- 锁机制适用于需要互斥的是某个数据结构而不是代码块的情况。



### 规约子句



归约操作符(reduction operator) 是一个二元操作（例如：加法和减法），归约就是将相同的归约操作符重复地应用到操作数序列来得到一个结果的计算。

所有操作的中间结果存储在同一个变量里：归约变量(reduction variable)

reduction 子句的语法是：

reduction(<operator>: <variable list>)

在C 语言中， operator 可能是操作符＋、＊、－、＆ 、|、^、＆＆、||中的任意一个，但使用减法操作会有一点问题，因为减法不满足交换律和结合律。


 ###   parallel for 指令

- 在 parallel for 指令之后的结构化块必须是for循环，它不会并行化while 或do-while 循环。
    
- 只能并行化那些可以确定迭代次数的for循环
    
- 在一个已经被 parallel for 指令并行化的for 循环中，线程间的缺省划分方式是由系统决定的。OpenMP 编译器不检查被parallel for 指令并行化的循环所包含的迭代间的依赖关系，而是由程序员来识别这些依赖关系。
    
- 循环依赖(loop-carried dependence)：值在一个迭代中计算，其结果在之后的迭代中使用。
    
- 数据依赖
    
- OpenMP 提供了一个子句default, 该子句显式地要求我们这样做。如果我们添加子句default(none)到 parallel 或 parallel for 指令中，那么编译器将要求我们明确在这个块中使用的每个变量和已经在块之外声明的变量的作用域。（在一个块中声明的变量都是私有的，因为它们会被分配给线程的栈。）
    

