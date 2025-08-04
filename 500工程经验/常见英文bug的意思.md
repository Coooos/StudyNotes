### **一、语法错误（Syntax Errors）**
| 英文提示                                             | 中文解释   | 常见原因及修复方法                   |
| ------------------------------------------------ | ------ | --------------------------- |
| `SyntaxError: invalid syntax`                    | 无效语法   | 缺少冒号、括号不匹配、拼写错误（如`pront()`） |
| `IndentationError: unexpected indent`            | 意外的缩进  | 代码缩进不一致（混用空格和Tab）           |
| `SyntaxError: EOL while scanning string literal` | 字符串未闭合 | 字符串缺少引号（如 `print("hello)`）  |

---

### **二、变量与类型错误（Variables & Types）**
| 英文提示                                                     | 中文解释     | 示例及修复方法                                          |
| -------------------------------------------------------- | -------- | ------------------------------------------------ |
| `NameError: name 'x' is not defined`                     | 变量未定义    | 拼写错误或变量未声明（检查变量名）                                |
| `TypeError: can only concatenate str (not "int") to str` | 类型不匹配    | 字符串和数字直接相加（如 `"age: " + 18` → 改用 `f"age: {18}"`） |
| `TypeError: 'NoneType' object is not subscriptable`      | 对象不可下标访问 | 对`None`值进行索引（如 `lst = None; lst[0]`）             |

---

### **三、数据结构操作错误**
| 英文提示                                                     | 中文解释      | 示例及修复方法                                    |
| -------------------------------------------------------- | --------- | ------------------------------------------ |
| `IndexError: list index out of range`                    | 列表索引越界    | 访问不存在的索引（如 `lst=[1]; lst[2]`）              |
| `KeyError: 'name'`                                       | 字典键不存在    | 访问不存在的键（用 `dict.get('name')` 避免）           |
| `AttributeError: 'list' object has no attribute 'split'` | 对象无此属性/方法 | 错误调用方法（如 `lst.split()` → 应为 `str.split()`） |

---

### **四、函数与模块错误**
| 英文提示                                                | 中文解释   | 示例及修复方法                              |
| --------------------------------------------------- | ------ | ------------------------------------ |
| `ImportError: No module named 'numpy'`              | 模块未安装  | 先运行 `pip install numpy`              |
| `TypeError: missing 1 required positional argument` | 缺少参数   | 函数调用时参数不足（如 `def foo(x):` → `foo()`） |
| `RecursionError: maximum recursion depth exceeded`  | 递归深度超限 | 递归函数未设置终止条件                          |

---

### **五、文件与系统错误**
| 英文提示                                                     | 中文解释  | 示例及修复方法                                     |
| -------------------------------------------------------- | ----- | ------------------------------------------- |
| `FileNotFoundError: [Errno 2] No such file or directory` | 文件不存在 | 检查文件路径（用 `os.path.exists()` 验证）             |
| `PermissionError: [Errno 13] Permission denied`          | 权限不足  | 文件不可写或目录无权限（检查文件属性）                         |
| `UnicodeDecodeError: 'utf-8' codec can't decode byte`    | 编码错误  | 文件非UTF-8编码（如用 `open(file, encoding='gbk')`） |

---

### **六、网络与API错误**
| 英文提示                                              | 中文解释     | 示例及修复方法                                   |
| ------------------------------------------------- | -------- | ----------------------------------------- |
| `ConnectionError: Failed to establish connection` | 连接失败     | 检查网络或URL（如 `requests.get("错误的URL")`）      |
| `TimeoutError: The request timed out`             | 请求超时     | 增加超时时间或重试                                 |
| `JSONDecodeError: Expecting value`                | JSON解析失败 | 返回数据非JSON格式（先用 `print(response.text)` 检查） |

---

### **七、其他高频通用错误**
| 英文提示                                                 | 中文解释  | 示例及修复方法                       |
| ---------------------------------------------------- | ----- | ----------------------------- |
| `ZeroDivisionError: division by zero`                | 除数为零  | 检查分母是否为0（如 `x = 1/0`）         |
| `ValueError: invalid literal for int() with base 10` | 值转换失败 | 字符串转数字时含非数字字符（如 `int("12a")`） |
| `MemoryError: Out of memory`                         | 内存不足  | 优化代码或使用生成器减少内存占用              |
