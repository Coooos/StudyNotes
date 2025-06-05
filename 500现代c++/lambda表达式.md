[C++11中的lambda，std::function以及std:bind](https://paul.pub/cpp-lambda-function-bind/)

Lambda表达式是C++11引入的一种匿名函数对象，它允许你在代码中直接定义一个函数，而无需显式地声明一个函数名。Lambda表达式在处理简单函数时非常方便，尤其是在需要传递函数作为参数（如标准库中的算法）时。

**使用lambda表达式，可以让我们省却定义函数的麻烦，以inline的方式写出代码，这样的代码通常更简洁。  并且，由于阅读代码时不用寻找函数定义，这样的代码也更易读。

### 1. **基本语法**
Lambda表达式的基本语法如下：
```cpp
[capture](parameters) -> return_type { body }
```
- **`[capture]`**：捕获列表，用于捕获外部变量。
- **`(parameters)`**：参数列表，与普通函数的参数列表类似。
- **`-> return_type`**：返回类型，可选。如果省略，编译器会自动推导返回类型。
- **`{ body }`**：函数体，与普通函数的函数体类似。

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

### 3. **示例代码**
以下是一些Lambda表达式的示例：

#### 3.1 简单的Lambda表达式
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 使用Lambda表达式作为参数
    std::for_each(vec.begin(), vec.end(), [](int x) {
        std::cout << x << " ";
    });
    std::cout << std::endl;

    return 0;
}
```

#### 3.2 带捕获列表的Lambda表达式
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int factor = 2;
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 通过值捕获外部变量
    std::for_each(vec.begin(), vec.end(), [factor](int x) {
        std::cout << x * factor << " ";
    });
    std::cout << std::endl;

    // 通过引用捕获外部变量
    std::for_each(vec.begin(), vec.end(), [&factor](int x) {
        std::cout << x * factor << " ";
    });
    std::cout << std::endl;

    return 0;
}
```

#### 3.3 带返回值的Lambda表达式
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 使用Lambda表达式计算平方
    std::for_each(vec.begin(), vec.end(), [](int x) {
        std::cout << x * x << " ";
    });
    std::cout << std::endl;

    return 0;
}
```

#### 3.4 带捕获列表和返回值的Lambda表达式
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int factor = 2;
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 通过值捕获外部变量并返回计算结果
    std::for_each(vec.begin(), vec.end(), [factor](int x) -> int {
        return x * factor;
    });

    return 0;
}
```

### 4. **Lambda表达式的特性**
- **匿名性**：Lambda表达式没有函数名，因此不能递归调用。
- **类型推导**：编译器可以自动推导Lambda表达式的参数类型和返回类型。
- **捕获列表**：通过捕获列表，Lambda表达式可以访问外部变量。
- **可调用对象**：Lambda表达式是一个可调用对象，可以像普通函数一样使用。

### 5. **Lambda表达式的应用场景**
- **标准库算法**：如 `std::for_each`、`std::sort` 等，Lambda表达式可以作为参数传递。
- **线程创建**：如 `std::thread`，Lambda表达式可以作为线程函数。
- **事件处理**：在事件驱动编程中，Lambda表达式可以作为事件处理器。

### 6. **注意事项**
- **捕获列表的生命周期**：捕获的变量必须在Lambda表达式使用期间有效，否则可能导致未定义行为。
- **异常安全**：Lambda表达式中的代码需要考虑异常安全，确保不会抛出未处理的异常。
- **性能**：Lambda表达式通常不会带来额外的性能开销，但过多的捕获可能导致额外的内存分配。

### 7. **总结**
Lambda表达式是C++11引入的一种强大的特性，它允许你在代码中直接定义匿名函数对象。通过捕获列表，Lambda表达式可以访问外部变量，使得代码更加简洁和灵活。Lambda表达式在标准库算法、线程创建和事件处理等场景中非常有用。