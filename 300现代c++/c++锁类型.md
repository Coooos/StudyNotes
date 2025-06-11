[C++所有锁的讲解、使用场景、相应的C++代码示例_c++ 锁-CSDN博客](https://blog.csdn.net/weixin_44046545/article/details/138551385)
[「硬核科普」C++11锁机制三兄弟大比拼：mutex、lock_guard与unique_lock - 江小康 - 博客园](https://www.cnblogs.com/xiaokang-coding/p/18895297)

在 C++ 中，锁（Lock）是用于同步线程访问共享资源的重要机制，以防止数据竞争和保证线程安全。C++ 标准库提供了多种锁类型，每种锁类型都有其特定的用途和性能特点。以下是对 C++ 中常见锁类型的简单介绍：

### 1. **互斥锁（`std::mutex`）**
互斥锁是最基本的锁类型，用于保护共享资源，确保同一时间只有一个线程可以访问该资源。

- **特点**：
  - 不可递归：同一个线程不能多次锁定同一个互斥锁。
  - 不可移动：`std::mutex` 不支持移动语义，不能被复制或移动。
  - 阻塞式：如果线程尝试锁定一个已经被其他线程锁定的互斥锁，该线程会被阻塞，直到互斥锁被释放。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <mutex>

  std::mutex mtx;

  void print_block(int n, char c) {
      std::unique_lock<std::mutex> lock(mtx);
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
  }

  int main() {
      std::thread t1(print_block, 50, '*');
      std::thread t2(print_block, 50, '$');

      t1.join();
      t2.join();

      return 0;
  }
  ```

### 2. **递归互斥锁（`std::recursive_mutex`）**
递归互斥锁允许同一个线程多次锁定同一个锁，但每次锁定后必须相应地解锁。

- **特点**：
  - 可递归：同一个线程可以多次锁定同一个递归互斥锁。
  - 解锁次数必须与锁定次数匹配：每次锁定后必须解锁一次。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <mutex>

  std::recursive_mutex mtx;

  void print_block(int n, char c) {
      std::unique_lock<std::recursive_mutex> lock(mtx);
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
  }

  void print_block_twice(int n, char c) {
      std::unique_lock<std::recursive_mutex> lock(mtx);
      print_block(n, c);
      print_block(n, c);
  }

  int main() {
      std::thread t1(print_block_twice, 50, '*');
      std::thread t2(print_block_twice, 50, '$');

      t1.join();
      t2.join();

      return 0;
  }
  ```

### 3. **共享互斥锁（`std::shared_mutex`）**
共享互斥锁允许多个线程同时读取共享资源，但写入操作时必须独占访问。

- **特点**：
  - 支持共享锁和独占锁：多个线程可以同时持有共享锁，但只有一个线程可以持有独占锁。
  - 适用于读多写少的场景。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <shared_mutex>

  std::shared_mutex mtx;

  void print_block(int n, char c) {
      std::shared_lock<std::shared_mutex> lock(mtx);
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
  }

  void modify_block(int n, char c) {
      std::unique_lock<std::shared_mutex> lock(mtx);
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
  }

  int main() {
      std::thread t1(print_block, 50, '*');
      std::thread t2(print_block, 50, '$');
      std::thread t3(modify_block, 50, '#');

      t1.join();
      t2.join();
      t3.join();

      return 0;
  }
  ```

### 4. **读写锁（`std::shared_timed_mutex`）**
读写锁是共享互斥锁的扩展，支持超时机制，允许线程在尝试获取锁时等待一定时间。

- **特点**：
  - 支持共享锁和独占锁。
  - 支持超时机制：线程可以在尝试获取锁时设置超时时间。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <shared_mutex>
  #include <chrono>

  std::shared_timed_mutex mtx;

  void print_block(int n, char c) {
      std::shared_lock<std::shared_timed_mutex> lock(mtx, std::defer_lock);
      if (lock.try_lock_for(std::chrono::seconds(1))) {
          for (int i = 0; i < n; ++i) {
              std::cout << c;
          }
          std::cout << '\n';
      } else {
          std::cout << "Failed to acquire lock for " << c << '\n';
      }
  }

  void modify_block(int n, char c) {
      std::unique_lock<std::shared_timed_mutex> lock(mtx, std::defer_lock);
      if (lock.try_lock_for(std::chrono::seconds(1))) {
          for (int i = 0; i < n; ++i) {
              std::cout << c;
          }
          std::cout << '\n';
      } else {
          std::cout << "Failed to acquire lock for " << c << '\n';
      }
  }

  int main() {
      std::thread t1(print_block, 50, '*');
      std::thread t2(print_block, 50, '$');
      std::thread t3(modify_block, 50, '#');

      t1.join();
      t2.join();
      t3.join();

      return 0;
  }
  ```

### 5. **自旋锁（`std::spinlock`）**
自旋锁是一种轻量级锁，线程在尝试获取锁时会不断自旋（忙等待），而不是阻塞。

- **特点**：
  - 非阻塞：线程不会被阻塞，而是不断尝试获取锁。
  - 适用于锁持有时间非常短的场景。
  - 不支持递归。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <atomic>

  std::atomic_flag lock = ATOMIC_FLAG_INIT;

  void print_block(int n, char c) {
      while (lock.test_and_set(std::memory_order_acquire)) {
          // 自旋等待
      }
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
      lock.clear(std::memory_order_release);
  }

  int main() {
      std::thread t1(print_block, 50, '*');
      std::thread t2(print_block, 50, '$');

      t1.join();
      t2.join();

      return 0;
  }
  ```

### 6. **条件变量（`std::condition_variable`）**
条件变量用于线程间的同步，允许线程在某个条件不满足时等待，直到条件满足时被唤醒。

- **特点**：
  - 与互斥锁配合使用：线程在等待条件变量时必须持有互斥锁。
  - 支持通知机制：线程可以通过 `notify_one` 或 `notify_all` 唤醒等待的线程。

- **使用示例**：
  ```cpp
  #include <iostream>
  #include <thread>
  #include <mutex>
  #include <condition_variable>

  std::mutex mtx;
  std::condition_variable cv;
  bool ready = false;

  void print_block(int n, char c) {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [] { return ready; });
      for (int i = 0; i < n; ++i) {
          std::cout << c;
      }
      std::cout << '\n';
  }

  void set_ready() {
      {
          std::lock_guard<std::mutex> lock(mtx);
          ready = true;
      }
      cv.notify_all();
  }

  int main() {
      std::thread t1(print_block, 50, '*');
      std::thread t2(print_block, 50, '$');
      std::thread t3(set_ready);

      t1.join();
      t2.join();
      t3.join();

      return 0;
  }
  ```

