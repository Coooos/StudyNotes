## 0.概述

友元提供了一种 普通函数或者类成员函数 访问另一个类中的私有或保护成员 的机制。也就是说有两种形式的友元：

（1）友元函数：普通函数对一个访问某个类中的私有或保护成员。

（2）友元类：类A中的成员函数访问类B中的私有或保护成员

优点：提高了程序的运行效率。

缺点：破坏了类的封装性和数据的透明性。

总结：- 能访问私有成员- 破坏封装性- 友元关系不可传递- 友元关系的单向性- 友元声明的形式及数量不受限制

## 1.友元函数

在类声明的任何区域中声明，而定义则在类的外部。

1. `friend <类型><友元函数名>(<参数表>);`

注意，友元函数只是一个普通函数，并不是该类的类成员函数，它可以在任何地方调用，友元函数中通过对象名来访问该类的私有或保护成员。

具体代码见：[friend_func.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/friend/friend_func.cpp)

1. `#include <iostream>`

2. `using namespace std;`

3. `class A`
4. `{`
5. `public:`
6.     `A(int _a):a(_a){};`
7.     `friend int geta(A &ca);  ///< 友元函数`
8. `private:`
9.     `int a;`
10. `};`

11. `int geta(A &ca)` 
12. `{`
13.     `return ca.a;`
14. `}`

15. `int main()`
16. `{`
17.     `A a(3);`    
18.     `cout<<geta(a)<<endl;`

19.     `return 0;`
20. `}`

## 2.友元类

友元类的声明在该类的声明中，而实现在该类外。

1. `friend class <友元类名>;`

类B是类A的友元，那么类B可以直接访问A的私有成员。

具体代码见：[friend_class.cpp](https://github.com/Light-City/CPlusPlusThings/tree/master/basic_content/friend/friend_class.cpp)

1. `#include <iostream>`

2. `using namespace std;`

3. `class A`
4. `{`
5. `public:`
6.     `A(int _a):a(_a){};`
7.     `friend class B;`
8. `private:`
9.     `int a;`
10. `};`

11. `class B`
12. `{`
13. `public:`
14.     `int getb(A ca) {`
15.         `return  ca.a;` 
16.     `};`
17. `};`

18. `int main()` 
19. `{`
20.     `A a(3);`
21.     `B b;`
22.     `cout<<b.getb(a)<<endl;`
23.     `return 0;`
24. `}`

## 3.注意

- 友元关系没有继承性假如类B是类A的友元，类C继承于类A，那么友元类B是没办法直接访问类C的私有或保护成员。
    
- 友元关系没有传递性假如类B是类A的友元，类C是类B的友元，那么友元类C是没办法直接访问类A的私有或保护成员，也就是不存在“友元的友元”这种关系。