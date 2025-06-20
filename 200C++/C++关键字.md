[基础进阶 - const那些事 - 《C++那些事（Stories About C Plus Plus）》 - 书栈网 · BookStack](https://www.bookstack.cn/read/CPlusPlusThings/97534d50f44dc7a3.md)
# const

`const` 关键字用于声明常量，它指定了一个变量在初始化后不能被修改的特性。`const` 可以用于变量、成员函数和指针。

### 1. 常量变量（Const Variables）

常量变量一旦被初始化，就不能被修改。

```cpp
const int MAX_SIZE = 100; // 声明一个常量

int main() {
    // MAX_SIZE = 200; // 错误：常量不能被修改
    std::cout << "Max size: " << MAX_SIZE << std::endl;
    return 0;
}
```

### 2. 常量成员函数（Const Member Functions）

常量成员函数承诺不会修改对象的成员变量。在常量对象上调用常量成员函数是合法的，但在非常量对象上调用常量成员函数是不合法的。

```cpp
class MyClass {
public:
    int getValue() const { // 声明一个常量成员函数
        // some code
    }
};

int main() {
    const MyClass obj;
    int value = obj.getValue(); // 合法：在常量对象上调用常量成员函数
    return 0;
}
```

### 3. 指向常量的指针（Pointers to Constants）

指向常量的指针表示指针所指向的值是常量，**不能通过指针修改该值。

```cpp
int main() {
    const int* ptr; // 指向常量的指针
    int value = 5;
    ptr = &value;
    // *ptr = 10; // 错误：不能通过指针修改常量值
    return 0;
}
```

### 4. 常量指针（Constant Pointers）

常量指针表示指针本身是常量，**不能改变指针指向的地址。

```cpp
int main() {
    int value = 5;
    int* const ptr = &value; // 常量指针
    *ptr = 10; // 合法：可以通过常量指针修改指针指向的值
    // ptr = nullptr; // 错误：不能改变常量指针的指向
    return 0;
}
```
### 5.const 与 #define 的区别
**编译器处理方式不同**
define --> 在预处理阶段进行替换
const --> 在编译阶段确定其值

**类型检查**
define --> 无类型，不进行类型安全检查，可能会产生错误
const --> 有数据类型，编译时会检查

内存空间**
define --> 不分配内存，给出的是立即数，用了多少次就进行多少次替换，在内存中会有多个拷贝，消耗内存大
const --> 在静态存储区分配空间，程序运行中在内存中只有一次拷贝

在编译时，编译器通常不为const分配空间而是保存在符号表中，使得成为编译期间的常量，没有存储与读内存的操作，效率高。
宏替换只做替换，不做表达式计算。
# static
`static` 关键字用于声明**静态成员变量、静态成员函数和局部静态变量**，其作用取决于它所修饰的实体。

### 1. 静态成员变量（Static Member Variables）

静态成员变量是属于类的，而不是属于类的各个实例的。它的特点是所有类的实例共享同一份静态成员变量。静态成员变量可以通过类名直接访问，也可以通过对象访问。

```cpp
class MyClass {
public:
    static int count; // 声明静态成员变量
};

int MyClass::count = 0; // 初始化静态成员变量

int main() {
    MyClass obj1;
    MyClass obj2;
    
    obj1.count = 5; // 通过对象访问静态成员变量
    MyClass::count = 10; // 通过类名直接访问静态成员变量
    
    return 0;
}
```

### 2. 静态成员函数（Static Member Functions）

静态成员函数是属于类的函数，它不依赖于任何特定的对象。因此，它可以直接通过类名来调用，而不需要创建类的实例。

```cpp
class MyClass {
public:
    static void myStaticFunction() {
        // 静态成员函数的实现
    }
};

int main() {
    MyClass::myStaticFunction(); // 调用静态成员函数
    return 0;
}
```

### 3. 局部静态变量（Local Static Variables）

局部静态变量是在函数内部声明的静态变量，它具有静态生存期，即它在程序运行期间只初始化一次，并且在函数调用结束后仍然存在于内存中。

```cpp
void myFunction() {
    static int count = 0; // 声明局部静态变量
    count++;
    std::cout << "Count: " << count << std::endl;
}

int main() {
    myFunction();
    myFunction();
    myFunction();
    return 0;
}
```
### **1.作用范围**：

- `static` 关键字可以用于修饰变量、函数和类，它具有不同的作用效果。
- `const` 关键字用于声明常量，可以修饰变量、成员函数和指针。

### **2.作用时间**：

- `static` 关键字表示静态的特性，它的作用是在编译时期确定，程序运行过程中不会改变。
- `const` 关键字表示常量的特性，它的值在初始化后不能被修改。

### **3.内存分配**：

- `static` 修饰的静态变量在程序的静态存储区分配内存，它的生命周期贯穿整个程序的执行过程。
- `const` 修饰的常量可以是静态变量、全局变量、局部变量或者类的成员变量，其内存分配取决于其声明的位置和作用域。

### **4.用途**：

- `static` 用于实现静态成员变量、静态成员函数、局部静态变量等，通常用于共享数据和实现单例模式。
- `const` 用于声明常量，以及在成员函数中表明函数不会修改对象的状态，从而提高代码的安全性和可读性。



# volatile

对于 define 来说，宏定义实际上是在预编译阶段进⾏处理，没有类型，也就没有类型检查，仅仅做的是遇到宏定 义进⾏字符串的展开，遇到多少次就展开多少次，⽽且这个简单的展开过程中，很容易出现边界效应，达不到预期 的效果。因为 define 宏定义仅仅是展开，因此运⾏时系统并不为宏定义分配内存，但是从汇编 的⻆度来讲， define 却以⽴即数的⽅式保留了多份数据的拷⻉。 对于 const 来说，const 是在编译期间进⾏处理的，const 有类型，也有类型检查，程序运⾏时系统会为 const 常 分配内存，⽽且从汇编的⻆度讲，const 常在出现的地⽅保留的是真正数据的内存地址，只保留了⼀份数据的 拷⻉，省去了不必要的内存空间。⽽且，有时编译器不会为普通的 const 常分配内存，⽽是直接将 const 常添 加到符号表中，省去了读取和写⼊内存的操作，效率更⾼。



static 作⽤：控制变的存储⽅式和可⻅性。 作⽤⼀：修饰局部变：⼀般情况下，对于局部变在程序中是存放在栈区的，并且局部的⽣命周期在包含语句块 执⾏结束时便结束了。但是如果⽤ static 关键字修饰的话，该变便会存放在静态数据区，其⽣命周期会⼀直延续 到整个程序执⾏结束。但是要注意的是，虽然⽤ static 对局部变进⾏修饰之后，其⽣命周期以及存储空间发⽣了 变化，但其作⽤域并没有改变，作⽤域还是限制在其语句块。 作⽤⼆：修饰全部变：对于⼀个全局变，它既可以在本⽂件中被访问到，也可以在同⼀个⼯程中其它源⽂件被 访问(添加 extern进⾏声明即可)。⽤ static 对全局变进⾏修饰改变了其作⽤域范围，由原来的整个⼯程可⻅变成 了本⽂件可⻅。 作⽤三：修饰函数：⽤ static 修饰函数，情况和修饰全局变类似，也是改变了函数的作⽤域。 作⽤四：修饰类：如果 C++ 中对类中的某个函数⽤ static 修饰，则表示该函数属于⼀个类⽽不是属于此类的任何 特定对象；如果对类中的某个变进⾏ static 修饰，则表示该变以及所有的对象所有，存储空间中只存在⼀个副 本，可以通过；类和对象去调⽤。 （补充：静态⾮常数据成员，其只能在类外定义和初始化，在类内仅是声明⽽已。） 作⽤五：类成员/类函数声明 static 函数体内 static 变的作⽤范围为该函数体，不同于 auto 变，该变的内存只被分配⼀次，因此其值在下 次调⽤时仍维持上次的值； 在模块内的 static 全局变可以被模块内所⽤函数访问，但不能被模块外其它函数访问； 在模块内的 static 函数只可被这⼀模块内的其它函数调⽤，这个函数的使⽤范围被限制在声明它的模块内； 在类中的 static 成员变属于整个类所拥有，对类的所有对象只有⼀份拷⻉； 在类中的 static 成员函数属于整个类所拥有，这个函数不接收 this 指针，因⽽只能访问类的 static 成员变 。 static 类对象必须要在类外进⾏初始化，static 修饰的变先于对象存在，所以 static 修饰的变要在类外初 始化； 由于 static 修饰的类成员属于类，不属于对象，因此 static 类成员函数是没有 this 指针，this 指针是指向本 对象的指针，正因为没有 this 指针，所以 static 类成员函数不能访问⾮ static 的类成员，只能访问 static修饰 的类成员； static 成员函数不能被 virtual 修饰，static 成员不属于任何对象或实例，所以加上 virtual 没有任何实际意 义；静态成员函数没有 this 指针，虚函数的实现是为每⼀个对象分配⼀个 vptr 指针，⽽ vptr 是通过 this 指 针调⽤的，所以不能为 virtual；虚函数的调⽤关系，this->vptr->ctable->virtual function。 const 关键字：含义及实现机制 const 修饰基本类型数据类型：基本数据类型，修饰符 const 可以⽤在类型说明符前，也可以⽤在类型说明符后， 其结果是⼀样的。在使⽤这些常的时候，只要不改变这些常的值即可。 const 修饰指针变和引⽤变：如果 const 位于⼩星星的左侧，则 const 就是⽤来修饰指针所指向的变，即指 针指向为常；如果 const 位于⼩星星的右侧，则 const 就是修饰指针本身，即指针本身是常。 const 应⽤到函数中：作为参数的 const 修饰符：调⽤函数的时候，⽤相应的变初始化 const 常，则在函数体 中，按照 const 所修饰的部分进⾏常化，保护了原对象的属性。 [注意]：参数 const 通常⽤于参数为指针或引⽤ 的情况; 作为函数返回值的 const 修饰符：声明了返回值后，const 按照"修饰原则"进⾏修饰，起到相应的保护作 ⽤。 const 在类中的⽤法：const 成员变，只在某个对象⽣命周期内是常，⽽对于整个类⽽⾔是可以改变的。因为 类可以创建多个对象，不同的对象其 const 数据成员值可以不同。所以不能在类的声明中初始化 const 数据成员， 因为类的对象在没有创建时候，编译器不知道 const 数据成员的值是什么。const 数据成员的初始化只能在类的构 造函数的初始化列表中进⾏。const 成员函数：const 成员函数的主要⽬的是防⽌成员函数修改对象的内容。要注 意，const 关键字和 static 关键字对于成员函数来说是不能同时使⽤的，因为 static 关键字修饰静态成员函数不含 有 this 指针，即不能实例化，const 成员函数⼜必须具体到某⼀个函数。 const 修饰类对象，定义常对象：常对象只能调⽤常函数，别的成员函数都不能调⽤。 补充：const 成员函数中如果实在想修改某个变，可以使⽤ mutable 进⾏修饰。成员变中如果想建⽴在整个类 中都恒定的常，应该⽤类中的枚举常来实现或者 static const。 C ++ 中的 const类成员函数（⽤法和意义） 常对象可以调⽤类中的 const 成员函数，但不能调⽤⾮ const 成员函数； （原因：对象调⽤成员函数时，在形 参列表的最前⾯加⼀个形参 this，但这是隐式的。this 指针是默认指向调⽤函数的当前对象的，所以，很⾃然， this 是⼀个常指针 test * const，因为不可以修改 this 指针代表的地址。但当成员函数的参数列表（即⼩括号） 后加了 const 关键字（void print() const;），此成员函数为常成员函数，此时它的隐式this形参为 const test * const，即不可以通过 this 指针来改变指向对象的值。 ⾮常对象可以调⽤类中的 const 成员函数，也可以调⽤⾮ const 成员函数。


---

# inline
编译器工作时，以 .c 文件为单位逐个编译 .o 文件，每个 .c 文件的编译是独立的，如果 当前 .c 文件中要用到外部函数，那么就在编译时预留一个符号。等到所有 .o 文件生成后，链接时才给这些符号指定地址（链接脚本决定地址），所以这个 . c 文件编译时只会看到外部函数的声明而无法知道它的函数体。而内联函数声明时，加关键字 inline 修饰。调用到它的地方直接将汇编代码展开，而不需要通过符号（函数名）地址跳转。
## 内联函数

**在C语言中，如果一些函数被频繁调用，不断地有函数入栈，即函数栈，会造成栈空间或栈内存的大量消耗。

为了解决这个问题，特别的引入了inline修饰符，表示为内联函数。

栈空间就是指放置程序代码的局部数据也就是函数内数据的内存空间，在系统下，栈空间是有限的，假如频繁大量的使用就会造成因栈空间不足所造成的程序出错的问题，函数的死循环递归调用的最终结果就是导致栈内存空间枯竭。

## 代码用例

```c
#include <stdio.h>  
 
//函数定义为inline即:内联函数  
inline char* dbtest(int a) 
{  
	return (i % 2 > 0) ? "奇" : "偶";  
}   
  
int main()  
{  
	int i = 0;  
	for (i=1; i < 100; i++) 
	{  
		printf("i:%d    奇偶性:%s /n", i, dbtest(i));      
	}  
}
```

**其实在内部的工作就是在每个for循环的内部任何调用dbtest(i)的地方都换成了(i%2>0)?“奇”:“偶”，这样就避免了频繁调用函数对栈内存重复开辟所带来的消耗。**

## 内联函数编程风格

##### 1、关键字inline 必须与函数定义体放在一起才能使函数成为内联，仅将inline 放在函数声明前面不起任何作用。

如下风格的函数Foo 不能成为内联函数：

```c
inline void Foo(int x, int y); // inline 仅与函数声明放在一起
void Foo(int x, int y)
{
}
```

而如下风格的函数Foo 则成为内联函数：

```c
void Foo(int x, int y);
inline void Foo(int x, int y) // inline 与函数定义体放在一起
{
}
```

所以说，inline 是一种 **“用于实现的关键字**” ，**而不是一种“用于声明的关键字”**。一般地，用户可以阅读函数的声明，但是看不到函数的定义。尽管在大多数教科书中内联函数的声明、定义体前面都加了inline 关键字，但我认为inline 不应该出现在函数的声明中。这个细节虽然不会影响函数的功能，但是体现了高质量C++/C 程序设计风格的一个基本原则：**声明与定义不可混为一谈，用户没有必要、也不应该知道函数是否需要内联**。

##### 2、inline的使用是有所限制的

inline只适合**函数体内代码简单**的函数数使用，不能包含复杂的结构控制语句例如while、switch，并且内联函数本身不能是**直接递归函数**(自己内部还调用自己的函数)。

##### 3、慎用内联

内联能提高函数的执行效率，为什么不把所有的函数都定义成内联函数？如果所有的函数都是内联函数，还用得着“内联”这个关键字吗？

**内联是以代码膨胀（复制）为代价**，仅仅省去了函数调用的开销，从而提高函数的执行效率。**如果执行函数体内代码的时间，相比于函数调用的开销较大，那么效率的收获会很少**。另一方面，**每一处内联函数的调用都要复制代码，将使程序的总代码量增大，消耗更多的内存空间。**

以下情况不宜使用内联：

（1）如果函数体内的**代码比较长**，使用内联将导致内存消耗代价较高。

（2）如果函数体内**出现循环**，那么执行函数体内代码的时间要比函数调用的开销大。

一个好的编译器将会根据函数的定义体，自动地取消不值得的内联（这进一步说明了inline 不应该出现在函数的声明中）。

##### 4、将内联函数放在头文件里实现是合适的,省却你为每个文件实现一次的麻烦

而所以声明跟定义要一致,其实是指,如果在每个文件里都实现一次该内联函数的话,那么,最好保证每个定义都是一样的,否则,将会引起未定义的行为。

# extern
利用关键字extern，可以在一个文件中引用另一个文件中定义的变量或者函数，下面就结合具体的实例，分类说明一下。

## 一、引用同一个文件中的变量

```c
#include<stdio.h>

int func();

int main()
{
    func(); //1
    printf("%d",num); //2
    return 0;
}

int num = 3;

int func()
{
    printf("%d\n",num);
}

```

如果按照这个顺序，变量 num在main函数的后边进行声明和初始化的话，那么在main函数中是不能直接引用num这个变量的，因为当编译器编译到这一句话的时候，找不到num这个变量的声明，但是在func函数中是可以正常使用，因为func对num的调用是发生在num的声明和初始化之后。

如果我不想改变num的声明的位置，但是想在main函数中直接使用num这个变量，怎么办呢？可以使用extern这个关键字。像下面这一段代码，利用extern关键字先声明一下num变量，告诉编译器num这个变量是存在的，但是不是在这之前声明的，你到别的地方找找吧，果然，这样就可以顺利通过编译啦。但是你要是想欺骗编译器也是不行的，比如你声明了extern int num；但是在后面却没有真正的给出num变量的声明，那么编译器去别的地方找了，但是没找到还是不行的。

下面的程序就是利用extern关键字，使用在后边定义的变量。

```c
#include<stdio.h>

int func();

int main()
{
    func(); //1
    extern int num;
    printf("%d",num); //2
    return 0;
}

int num = 3;

int func()
{
    printf("%d\n",num);
}

```

## 二、引用另一个文件中的变量

如果extern这个关键字就这点功能，那么这个关键字就显得多余了，因为上边的程序可以通过将num变量在main函数的上边声明，使得在main函数中也可以使用。  
extern这个关键字的真正的作用是引用不在同一个文件中的变量或者函数。  
**main.c**

```c
#include<stdio.h>

int main()
{
    extern int num;
    printf("%d",num);
    return 0;
}
```

**b.c**

```c
#include<stdio.h>

int num = 5;

void func()
{
    printf("fun in a.c");
}

```

例如，这里b.c中定义了一个变量num，如果main.c中想要引用这个变量，那么可以使用extern这个关键字，注意这里能成功引用的原因是，num这个关键字在b.c中是一个全局变量，也就是说只有当一个变量是一个全局变量时，extern变量才会起作用。

另外，extern关键字只需要指明类型和变量名就行了，不能再重新赋值，初始化需要在原文件所在处进行，如果不进行初始化的话，全局变量会被编译器自动初始化为0。

像这种写法是不行的。  
`extern int num=4;`  
但是在声明之后就可以使用变量名进行修改了。

```c
#include<stdio.h>

int main()
{
    extern int num;
    num=1;
    printf("%d",num);
    return 0;
}

```

使用include将另一个文件全部包含进去可以引用另一个文件中的变量，但是这样做的结果就是，被包含的文件中的所有的变量和方法都可以被这个文件使用，这样就变得不安全，如果只是希望一个文件使用另一个文件中的某个变量还是使用extern关键字更好。

## 引用另一个文件中的函数

extern除了引用另一个文件中的变量外，还可以引用另一个文件中的函数，引用方法和引用变量相似。  
mian.c

```c
#include<stdio.h>

int main()
{
    extern void func();
    func();
    return 0;
}
```

b.c

```c
#include<stdio.h>

const int num=5;
void func()
{
    printf("fun in a.c");
}
```

这里main函数中引用了b.c中的函数func。因为所有的函数都是全局的，所以对函数的extern用法和对全局变量的修饰基本相同，需要注意的就是，需要指明返回值的类型和参数。
## 区别
| 特性         | `inline`                               | `extern`                  |
| ---------- | -------------------------------------- | ------------------------- |
| **用途**     | 建议编译器将函数调用替换为函数主体代码，以优化性能。             | 声明变量或函数在其他文件中定义，用于跨文件链接。  |
| **作用对象**   | 函数。                                    | 变量或函数。                    |
| **使用位置**   | 函数定义。                                  | 变量或函数声明。                  |
| **对链接的影响** | 编译器可能将多个 `inline` 函数的定义合并为一个，避免重复符号错误。 | 告诉编译器变量或函数在其他地方定义，避免链接错误。 |
| **性能影响**   | 可能提高性能，减少函数调用开销。                       | 无直接影响，主要用于链接阶段。           |

---

# this


# explicit
- explicit 修饰构造函数时，可以防止隐式转换和复制初始化
- explicit 修饰转换函数时，可以防止隐式转换，但按语境转换除外

# ：：
- 全局作用域符（::name）：用于类型名称（类、类成员、成员函数、变量等）前，表示作用域为全局命名空间
- 类作用域符（class::name）：用于表示指定类型的作用域范围是具体某个类的
- 命名空间作用域符（namespace::name）:用于表示指定类型的作用域范围是具体某个命名空间的
# decltype
decltype的语法是:

1. `decltype (expression)`

这里的括号是必不可少的,decltype的作用是“查询表达式的类型”，因此，上面语句的效果是，返回 expression 表达式的类型。注意，decltype 仅仅“查询”表达式的类型，并不会对表达式进行“求值”。

### 1.1 推导出表达式类型

1. `int i = 4;`
2. `decltype(i) a; //推导结果为int。a的类型为int。`

### 1.2 与using/typedef合用，用于定义类型。

1. `using size_t = decltype(sizeof(0));//sizeof(a)的返回值为size_t类型`
2. `using ptrdiff_t = decltype((int*)0 - (int*)0);`
3. `using nullptr_t = decltype(nullptr);`
4. `vector<int >vec;`
5. `typedef decltype(vec.begin()) vectype;`
6. `for (vectype i = vec.begin; i != vec.end(); i++)`
7. `{`
8. `//...`
9. `}`

这样和auto一样，也提高了代码的可读性。

### 1.3 重用匿名类型

在C++中，我们有时候会遇上一些匿名类型，如:

1. `struct` 
2. `{`
3.     `int d ;`
4.     `doubel b;`
5. `}anon_s;`

而借助decltype，我们可以重新使用这个匿名的结构体：

1. `decltype(anon_s) as ;//定义了一个上面匿名的结构体`

### 1.4 泛型编程中结合auto，用于追踪函数的返回值类型

这也是decltype最大的用途了。

1. `template <typename T>`
2. `auto multiply(T x, T y)->decltype(x*y)`
3. `{`
4.     `return x*y;`
5. `}`

完整代码见：[decltype.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/decltype/decltype.cpp)

## 2.判别规则

对于decltype(e)而言，其判别结果受以下条件的影响：

如果e是一个没有带括号的标记符表达式或者类成员访问表达式，那么的decltype（e）就是e所命名的实体的类型。此外，如果e是一个被重载的函数，则会导致编译错误。否则 ，假设e的类型是T，如果e是一个将亡值，那么decltype（e）为T&&否则，假设e的类型是T，如果e是一个左值，那么decltype（e）为T&。否则，假设e的类型是T，则decltype（e）为T。

标记符指的是除去关键字、字面量等编译器需要使用的标记之外的程序员自己定义的标记，而单个标记符对应的表达式即为标记符表达式。例如：

1. `int arr[4]`

则arr为一个标记符表达式，而arr[3]+0不是。

举例如下：

1. `int i = 4;`
2. `int arr[5] = { 0 };`
3. `int *ptr = arr;`
4. `struct S{ double d; }s ;`
5. `void Overloaded(int);`
6. `void Overloaded(char);//重载的函数`
7. `int && RvalRef();`
8. `const bool Func(int);`

9. `//规则一：推导为其类型`
10. `decltype (arr) var1; //int 标记符表达式`

11. `decltype (ptr) var2;//int *  标记符表达式`

12. `decltype(s.d) var3;//doubel 成员访问表达式`

13. `//decltype(Overloaded) var4;//重载函数。编译错误。`

14. `//规则二：将亡值。推导为类型的右值引用。`

15. `decltype (RvalRef()) var5 = 1;`

16. `//规则三：左值，推导为类型的引用。`

17. `decltype ((i))var6 = i;     //int&`

18. `decltype (true ? i : i) var7 = i; //int&  条件表达式返回左值。`

19. `decltype (++i) var8 = i; //int&  ++i返回i的左值。`

20. `decltype(arr[5]) var9 = i;//int&. []操作返回左值`

21. `decltype(*ptr)var10 = i;//int& *操作返回左值`

22. `decltype("hello")var11 = "hello"; //const char(&)[9]  字符串字面常量为左值，且为const左值。`

23. `//规则四：以上都不是，则推导为本类型`

24. `decltype(1) var12;//const int`

25. `decltype(Func(1)) var13=true;//const bool`

26. `decltype(i++) var14 = i;//int i++返回右值`

学习参考：[https://www.cnblogs.com/QG-whz/p/4952980.html](https://www.cnblogs.com/QG-whz/p/4952980.html)
# namespace
https://blog.csdn.net/qq_40416052/article/details/82528676?fromshare=blogdetail&sharetype=blogdetail&sharerId=82528676&sharerefer=PC&sharesource=cos03&sharefrom=from_link

# 宏
## 1.宏中包含特殊符号

分为几种：`#`，`##`，`\`

### 1.1 字符串化操作符（#）

**在一个宏中的参数前面使用一个#,预处理器会把这个参数转换为一个字符数组**，换言之就是：**#是“字符串化”的意思，出现在宏定义中的#是把跟在后面的参数转换成一个字符串**。

**注意：其只能用于有传入参数的宏定义中，且必须置于宏定义体中的参数名前。**

例如：

1. `#define exp(s) printf("test s is:%s\n",s)`
2. `#define exp1(s) printf("test s is:%s\n",#s)`
3. `#define exp2(s) #s` 
4. `int main() {`
5.     `exp("hello");`
6.     `exp1(hello);`

7.     `string str = exp2(   bac );`
8.     `cout<<str<<" "<<str.size()<<endl;`
9.     `/**`
10.      `* 忽略传入参数名前面和后面的空格。`
11.      `*/`
12.     `string str1 = exp2( asda  bac );`
13.     `/**`
14.      `* 当传入参数名间存在空格时，编译器将会自动连接各个子字符串，`
15.      `* 用每个子字符串之间以一个空格连接，忽略剩余空格。`
16.      `*/`
17.     `cout<<str1<<" "<<str1.size()<<endl;`
18.     `return 0;`
19. `}`

上述代码给出了基本的使用与空格处理规则，空格处理规则如下：

- 忽略传入参数名前面和后面的空格。

1. `string str = exp2(   bac );`
2. `cout<<str<<" "<<str.size()<<endl;`

输出：

1. `bac 3`

- 当传入参数名间存在空格时，编译器将会自动连接各个子字符串，用每个子字符串之间以一个空格连接，忽略剩余空格。

1. `string str1 = exp2( asda  bac );`
2. `cout<<str1<<" "<<str1.size()<<endl;`

输出：

1. `asda bac 8`

### 1.2 符号连接操作符（##）

**“##”是一种分隔连接方式，它的作用是先分隔，然后进行强制连接。将宏定义的多个形参转换成一个实际参数名。**

注意事项：

**（1）当用##连接形参时，##前后的空格可有可无。**

**（2）连接后的实际参数名，必须为实际存在的参数名或是编译器已知的宏定义。**

**（3）如果##后的参数本身也是一个宏的话，##会阻止这个宏的展开。**

示例：

1. `#define expA(s) printf("前缀加上后的字符串为:%s\n",gc_##s)  //gc_s必须存在`
2. `// 注意事项2`
3. `#define expB(s) printf("前缀加上后的字符串为:%s\n",gc_  ##  s)  //gc_s必须存在`
4. `// 注意事项1`
5. `#define gc_hello1 "I am gc_hello1"`
6. `int main() {`
7.     `// 注意事项1`
8.     `const char * gc_hello = "I am gc_hello";`
9.     `expA(hello);`
10.     `expB(hello1);`
11. `}`

### 1.3 续行操作符（\）

**当定义的宏不能用一行表达完整时，可以用”\”表示下一行继续此宏的定义。**

**注意 \ 前留空格。**

1. `#define MAX(a,b) ((a)>(b) ? (a) \`
2.    `:(b))`  
3. `int main() {`
4.     `int max_val = MAX(3,6);`
5.     `cout<<max_val<<endl;`
6. `}`

上述代码见：[sig_examp.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/macro/sig_examp.cpp)

## 2.do{…}while(0)的使用

### 2.1 避免语义曲解

例如：

1. `#define fun() f1();f2();`
2. `if(a>0)`
3.     `fun()`

这个宏被展开后就是：

1. `if(a>0)`
2.     `f1();`
3.     `f2();`

本意是a>0执行f1 f2，而实际是f2每次都会执行，所以就错误了。

为了解决这种问题，在写代码的时候，通常可以采用`{}`块。

如：

1. `#define fun() {f1();f2();}`
2. `if(a>0)`
3.     `fun();`
4. `// 宏展开`
5. `if(a>0)`
6. `{`
7.     `f1();`
8.     `f2();`
9. `};`

但是会发现上述宏展开后多了一个分号，实际语法不太对。(虽然编译运行没问题，正常没分号)。

### 2.2避免使用goto控制流

在一些函数中，我们可能需要在return语句之前做一些清理工作，比如释放在函数开始处由malloc申请的内存空间，使用goto总是一种简单的方法：

1. `int f() {`
2.     `int *p = (int *)malloc(sizeof(int));`
3.     `*p = 10;` 
4.     `cout<<*p<<endl;`
5. `#ifndef DEBUG`
6.     `int error=1;`
7. `#endif`
8.     `if(error)`
9.         `goto END;`
10.     `// dosomething`
11. `END:`
12.     `cout<<"free"<<endl;`
13.     `free(p);`
14.     `return 0;`
15. `}`

但由于goto不符合软件工程的结构化，而且有可能使得代码难懂，所以很多人都不倡导使用，这个时候我们可以使用do{…}while(0)来做同样的事情：

1. `int ff() {`
2.     `int *p = (int *)malloc(sizeof(int));`
3.     `*p = 10;` 
4.     `cout<<*p<<endl;`
5.     `do{` 
6. `#ifndef DEBUG`
7.         `int error=1;`
8. `#endif`
9.         `if(error)`
10.             `break;`
11.         `//dosomething`
12.     `}while(0);`
13.     `cout<<"free"<<endl;`
14.     `free(p);`
15.     `return 0;`
16. `}`

这里将函数主体部分使用do{…}while(0)包含起来，使用break来代替goto，后续的清理工作在while之后，现在既能达到同样的效果，而且代码的可读性、可维护性都要比上面的goto代码好的多了。

### 2.3 避免由宏引起的警告

内核中由于不同架构的限制，很多时候会用到空宏，。在编译的时候，这些空宏会给出warning，为了避免这样的warning，我们可以使用do{…}while(0)来定义空宏：

1. `#define EMPTYMICRO do{}while(0)`

### 2.4 定义单一的函数块来完成复杂的操作

如果你有一个复杂的函数，变量很多，而且你不想要增加新的函数，可以使用do{…}while(0)，将你的代码写在里面，里面可以定义变量而不用考虑变量名会同函数之前或者之后的重复。这种情况应该是指一个变量多处使用（但每处的意义还不同），我们可以在每个do-while中缩小作用域，比如：

1. `int fc()`
2. `{`
3.     `int k1 = 10;`
4.     `cout<<k1<<endl;`
5.     `do{`
6.         `int k1 = 100;`
7.         `cout<<k1<<endl;`
8.     `}while(0);`
9.     `cout<<k1<<endl;`
10. `}`

上述代码见：[do_while.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/macro/do_while.cpp)

学习文章：[https://www.cnblogs.com/lizhenghn/p/3674430.html](https://www.cnblogs.com/lizhenghn/p/3674430.html)