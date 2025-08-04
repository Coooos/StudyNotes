[C++11中的lambda，std::function以及std:bind](https://paul.pub/cpp-lambda-function-bind/)

Lambda表达式是C++11引入的一种匿名函数对象，它允许你在代码中直接定义一个函数，而无需显式地声明一个函数名。Lambda表达式在处理简单函数时非常方便，尤其是在需要传递函数作为参数（如标准库中的算法）时。

**使用lambda表达式，可以让我们省却定义函数的麻烦，以inline的方式写出代码，这样的代码通常更简洁。  并且，由于阅读代码时不用寻找函数定义，这样的代码也更易读。

### **语法**

Lambda表达式的语法如下：
```cpp
[捕获列表](参数列表) -> 返回类型 { 函数体 }
```

#### **组成部分**

1. **捕获列表（Capture List）**：

- 捕获列表定义了Lambda表达式可以访问的外部变量。

- 可以通过值捕获（`[=]`）或引用捕获（`[&]`）来访问外部变量。

- 也可以显式指定捕获的变量，例如 `[x, &y]` 表示按值捕获 `x`，按引用捕获 `y`。

2. **参数列表（Parameter List）**：
- 参数列表定义了Lambda表达式接受的参数类型和名称。

- 与普通函数类似，参数列表可以为空。

1. **返回类型（Return Type）**：
- 返回类型是可选的。如果Lambda表达式有一个非空的返回类型，可以显式指定。

- 如果未指定返回类型，编译器会根据函数体中的返回语句推断返回类型。    

1. **函数体（Function Body）**：

- 函数体包含Lambda表达式的执行逻辑。

- 可以包含任意的C++语句，包括返回语句。



### 2. **捕获列表**
捕获列表用于将外部变量捕获到Lambda表达式中。捕获方式有两种：
- **值捕获**：通过值捕获变量，捕获的变量在Lambda表达式中是副本。
- **引用捕获**：通过引用捕获变量，捕获的变量在Lambda表达式中是引用。

捕获列表可以使用以下语法：
- **`[var]`**：通过值捕获变量`var`。
- **`[&var]`**：通过引用捕获变量`var`。
- **`[this]`**：捕获当前对象的指针。
- **`[=]`**：通过值捕获所有外部变量。
- **`[&]`**：通过引用捕获所有外部变量。

### **3.示例**

以下是一些Lambda表达式的示例：

#### **1. 简单的Lambda表达式**
```cpp
#include <iostream>

int main() {
    auto lambda = []() {
        std::cout << "Hello, Lambda!" << std::endl;
    };

    lambda(); // 调用Lambda表达式
    return 0;
}
```

#### **2. 带参数和返回值的Lambda表达式**

```cpp
#include <iostream>

int main() {
    auto add = [](int a, int b) -> int {
        return a + b;
    };

    std::cout << "Result: " << add(3, 5) << std::endl;
    return 0;
}
```

#### **3. 捕获外部变量**

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;

    auto sum = [x, &y]() -> int {
        return x + y;
    };

    std::cout << "Sum: " << sum() << std::endl;
    return 0;
}
```

#### **4. 捕获列表的变种**

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;

    // 按值捕获所有外部变量
    auto sum_by_value = [=]() -> int {
        return x + y;
    };

    // 按引用捕获所有外部变量
    auto sum_by_reference = [&]() -> int {
        return x + y;
    };

    std::cout << "Sum by value: " << sum_by_value() << std::endl;
    std::cout << "Sum by reference: " << sum_by_reference() << std::endl;
    return 0;
}
```


### 4. **Lambda表达式的特性**
- **匿名性**：Lambda表达式没有函数名，因此不能递归调用。
- **类型推导**：编译器可以自动推导Lambda表达式的参数类型和返回类型。
- **捕获列表**：通过捕获列表，Lambda表达式可以访问外部变量。
- **可调用对象**：Lambda表达式是一个可调用对象，可以像普通函数一样使用。
### **5.应用场景**

Lambda表达式在C++中非常有用，特别是在以下场景中：

1. **算法和容器操作**：
    
    - 用于 `std::for_each`、`std::transform`、`std::sort` 等算法中，作为回调函数。
        
        ```cpp
    #include <iostream>
    #include <vector>
    #include <algorithm>
    
    int main() {
        std::vector<int> vec = {1, 2, 3, 4, 5};
    
        std::for_each(vec.begin(), vec.end(), [](int x) {
            std::cout << x << " ";
        });
    
        return 0;
    }
    ```
    
2. **并发编程**：
    
    - 用于 `std::thread`、`std::async` 等并发API中，作为任务函数。
        
    
    ```cpp
    #include <iostream>
    #include <thread>
    
    int main() {
        int x = 42;
    
        std::thread t([x]() {
            std::cout << "Value in thread: " << x << std::endl;
        });
    
        t.join();
        return 0;
    }
    ```
    
3. **事件处理和回调**：
    
    - 用于事件驱动编程中，作为事件处理函数。
        
    ```cpp
    #include <iostream>
    #include <functional>
    
    void on_event(std::function<void()> callback) {
        callback();
    }
    
    int main() {
        on_event([]() {
            std::cout << "Event occurred!" << std::endl;
        });
    
        return 0;
    }
    ```
    

### 6. **注意事项**
- **捕获列表的生命周期**：捕获的变量必须在Lambda表达式使用期间有效，否则可能导致未定义行为。
- **异常安全**：Lambda表达式中的代码需要考虑异常安全，确保不会抛出未处理的异常。
- **性能**：Lambda表达式通常不会带来额外的性能开销，但过多的捕获可能导致额外的内存分配。

### 7. **总结**
Lambda表达式是C++11引入的一种强大的特性，它允许你在代码中直接定义匿名函数对象。通过捕获列表，Lambda表达式可以访问外部变量，使得代码更加简洁和灵活。Lambda表达式在标准库算法、线程创建和事件处理等场景中非常有用。