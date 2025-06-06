
先做实验lab，学习数据库的课程。再考虑项目

| 学习任务    |     |
| ------- | --- |
| 基础语法学习  |     |
| 数据结构与算法 | 刷题  |
| os      | lab |
| 计算机网络   |     |
|         |     |
| 项目      |     |
|         |     |
|         |     |
|         |     |
lab+八股+项目+leetcode 速成实习
# 1 阶段1-基础语法学习

**学习编程语言，敲代码才能熟练。电子书或者视频可见的代码内容能一键复制，极大地提高了效率，而纸质书则非常并不方便

**第一遍: 看黑马的视频课程**  
由于自己一开始不懂, 直接看的黑马的视频课程, 但其缺少常见新特性和并发编程的知识, 仅仅适合新人入门, 距离你找工作差得很远

**第二遍: 刷CS106L**  
后来发现了[csdiy](https://csdiy.wiki/)这个自学指南, 就刷这里推荐的公开课, 这里我刷的是`CS106L`, 同时建议把杜克大学的C语言的课程[Introductory C Programming Specialization](https://www.coursera.org/specializations/c-programming)也刷了

**第三遍: 阅读电子书学习现代C++特性**  
这一阶段, 我学习的目标主要是涉及`C++`的新特性, 包括智能指针、右值引用、模板元编程这一类

**推荐的学习方法**  
其实我列举出我的3遍`C++`学习路线, 不是为了说明`C++`真的要学三遍, 而是为了说明方法很重要, 我学3遍的原因还不是因为没有找到合适的途径. 现在要推荐的话, 我如下推荐:

1. 电子教程 or 刷视频快速学习`C++`基础知识, 视频推荐[B站](https://www.bilibili.com/video/BV1o8411x7K3/?spm_id_from=333.1387.upload.video_card.click&vd_source=213e02ac84a7c6ef251b9f5d6691e21c), 电子书推荐(零基础C++)[[https://www.yuque.com/lianlianfengchen-cvvh2/zack]](https://www.yuque.com/lianlianfengchen-cvvh2/zack]), 其中这份电子书不仅仅是基础, 包括后续的高阶知识, 强烈推荐
2. 刷视频课 or 电子书重点学习`C++`的新特性, 这里的电子书包括: [现代C++教程](https://changkun.de/modern-cpp/zh-cn/00-preface/), 视频课推荐之前提到的[CS106L](https://web.stanford.edu/class/cs106l/)

# 2 阶段2-基础四大件学习

无论是什么语言，基础四大件都是学习不可缺少的，我基础四大件基本上都是经典教材+刷公开课or训练营的方式学习的。因此我虽然不是计算机科班，但我的计算机基础知识其实还挺扎实的：

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#2-1-%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E4%B8%8E%E7%AE%97%E6%B3%95 "2.1 数据结构与算法")2.1 数据结构与算法

这个没什么好说的， 坚持每天刷`LeetCode`就行了。我到去年秋招时刷了差不多300道题，足够应付正常难度的笔试了. 不过有一些要点需要关注:

1. 如果没有时间， 就优先刷`Hot100`
2. 刷题时尽量选择ACM模式, 贴近笔试的实际场景
3. `LeetCode`缺少图论的题, 可从其他途径补充图论的算法题

刷题重要知识点梳理:

1. 字符串
    1. `KMP`算法
    2. 双指针
2. 树
    1. 不同遍历方法
    2. 反转、重构
    3. 树与数组的转化
3. 图
    1. 最短路径算法
    2. 查并集
    3. 最小生成树
4. 回溯
5. 贪心
6. 动态规划

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#2-2-%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F "2.2 操作系统")2.2 操作系统

操作系统曾是我主要技术栈之一，我的学习路线是：

**第一步： 经典教材阅读**  
这里我强烈推荐《操作系统导论》

![操作系统导论](https://tonixwd.github.io/images/Notes/OS-three.png)

> 这里还有一个本教程推荐: 《操作系统真相还原》

**第二步: 刷 MIT6.S081**  
这里我刷了知名课程[MIT6.S081](https://pdos.csail.mit.edu/6.S081/2020/schedule.html)并做完了`xv6`的课程`lab`, 收获很多。其实熟读完《操作系统导论》后， `MIT6.S081`的课程知识很多都是重复的，这里的重点学习内容是`xv6`的课程作业, 通过代码来实践与加强对操作系统的理解。

> 这里不一定刷`MIT6.S081`这一课程, 南京大学蒋炎岩老师的[操作系统：设计与实现](https://jyywiki.cn/OS/2023/index.html)或者UCB的[CS162](https://cs162.org/)也很不错

**第三步: 参加OS训练营**  
之后我又参与了清华大学举办的开源操作系统训练营, 并获得了优秀学员, 这里是用`Rust`完成各项作业. 这个训练营分为三个阶段:

1. 阶段一: `rustlings`熟悉`Rust`语法和基本编程概念
2. 阶段二: `Rust`开发一个简单的操作系统`rCore`
3. 阶段三: 按照不同方向进行进一步的线上实习(也同样是上课做项目), 包括
    1. 内核进阶
    2. 虚拟化
    3. 组件化的`ArceOS`
    4. Linux内核驱动开发项目: [Rust for Linux](https://github.com/rust-for-linux/)

关于这个训练营，我还有一个专栏记录自己的学习过程：[清华大学开源操作系统训练营](https://www.zhihu.com/column/c_1738250998752292864)

可以看到, 学习操作系统的过程中, 我基础打得非常扎实, 因为我当时是把操作系统和驱动开发作为未来的求职方向之一，不过如果你对深入学习操作系统没有兴趣， 仅仅是为了满足对操作系统的基础学习的话，这里完成我的第一步， 读一下《操作系统导论》就可以了。

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#2-3-%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%BD%91%E7%BB%9C "2.3 计算机网络")2.3 计算机网络

计算机网络也是我基础四大件之一，且这直接对应于互联网后端这一岗位，非常重要。我学习路线是：

1. 阅读《计算机网络： 自顶向下方法》
2. 刷CS144

![计算机网络-自顶向下方法](https://tonixwd.github.io/images/Notes/CSNetWorking.png)

> 这里说明一下, 中科大的计算机网络课程也是使用的《计算机网络： 自顶向下方法》这本教材, 且更精炼一些, 如果时间不够或者读教材困难, 可以选择直接刷[中科大的计算机网络视频课](https://www.bilibili.com/video/BV1JV411t7ow/?spm_id_from=333.337.search-card.all.click)

同样, 时间不多的同学可以只看一下《计算机网络： 自顶向下方法》就可以了, 不需要再去刷公开课的`Lab`, 反正后面你大概率会通过C++网络编程的学习巩固这些知识

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#2-4-%E6%95%B0%E6%8D%AE%E5%BA%93 "2.4 数据库")2.4 数据库

数据库同样是我花了很多时间和精力去学习的，我学习路线是：  
**步骤1： 阅读《MySQL是怎样运行的》**  
阅读这本书，你会对数据库有一个基本的认识，并且大致了解其运行机制和实现原理。

**步骤2: 刷CMU 15445**  
我是直接看了`CMU 15445`所有的视频课程, 并自己独立完成了`CMU 15445`的课程作业`bustub`。独立做完这个课程项目后，你的数据库方面的知识已经能够比绝大多数非数据库领域的校招生强了。

> 其实我是先做的`CMU 15445`再看的《MySQL是怎样运行的》, 后者只要是为了应付秋招, 毕竟`CMU 15445`是宏观介绍所有的数据库基本知识, 而《MySQL是怎样运行的》聚焦于`MySQL`(国内最热门的数据库)

**步骤3: 刷MIT6.824**  
严格来说，数据库不等同于分布式系统，不过实际上他们结合很紧密，我这里就放在一起说了。`MIT6.824`是经典的分布式系统课程，基本上也是数据库领域校招生必学的课程。这里我同样完成了`MIT6.824`全套的`Lab`, 同样记录在[专栏: MIT6.8540(6.824)合集](https://www.zhihu.com/column/c_1725883332322226177)和[Github](https://github.com/ToniXWD/MIT6.5840)

**步骤4: 刷Talent Plan的TinyKV**  
`Talent Plan`是国内数据库公司`PingCap`举办的线上训练营, 其中的`TinyKV`项目是`MIT6.824`的完善版本, 实现了基于`Raft`的分布式KV数据库

数据库领域也是我之前学习的主攻方向之一, 最后秋招去了业务后端岗还挺遗憾的。如果对这方面感兴趣，也可以看看我自己原创的`LSM Tree KV存储引擎`项目：[Toni-LSM](https://github.com/ToniXWD/toni-lsm)

> 虽然这里的`MIT6.824`和`TinyKV`都是`Go`的项目而不是`C++`, 但他们的知识都是互通的, 且`C++`选手也会辅修`Go`的

既然这篇文章是介绍`C++`的, 那么首先我从个人的角度回答一下, `C++`和`Java`应该学哪个这个热点问题。我水的`C++`群都在唱衰`C++`, 我水的`Java`群都在Diss `Java`。就我个人经验而言，可以如下做一个总结：

**C++的优势/Java的劣势**

1. `C++`可以做底层开发, 相对而言可替代性比较低; 而`Java`确实人太多太卷，相对而言可替代性高。
2. `C++`就业领域更广, 除后端外还有`嵌入式`, `操作系统`, `数据库`, `存储`, `音视频`, `编译器`, `量化开发`, `AI infra`, `HPC`等; 而`Java`的除后端外应用领域相对较少.
3. `C++`的八股相对少, 基本上就是语言本身的知识; `Java`的八股更庞杂, 除了语言本身外, 还有`Spring`全家桶以及各种中间件

**C++的劣势/Java的优势**

1. `C++`学习难度大, 语法复杂, 学习曲线陡峭; 而`Java`相对简单, 学习曲线平缓. 可以认为, `C++`的八股`少而难`, `Java`大八股`庞而杂`
2. `C++`不同领域的知识点差异大, 不同领域不容易互转, 例如`音视频`和`AI infra`的专业领域知识完全不同, 很难互转; `Java`领域局限且统一
3. `C++`就业时的门槛确实相对较高

不过有一点需要澄清, 尽管`C++`和`Java`的语言选择确实一定程度上影响求职面试, 但其权重远不如你四大件和笔试的影响大, 求职最重要的还是你个人的能力和项目经验, 语言只是一个工具而已。

# [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#1-%E9%98%B6%E6%AE%B51-%E5%9F%BA%E7%A1%80%E8%AF%AD%E6%B3%95%E5%AD%A6%E4%B9%A0 "1 阶段1-基础语法学习")1 阶段1-基础语法学习

我从研0阶段开始学习`C++`的基础语法, 由于一开始没有领路人全靠自学, `C++`语言本身我前后至少学了3遍:

**第一遍: 看黑马的视频课程**  
由于自己一开始不懂, 直接看的黑马的视频课程, 但其缺少常见新特性和并发编程的知识, 仅仅适合新人入门, 距离你找工作差得很远

**第二遍: 刷CS106L**  
后来发现了[csdiy](https://csdiy.wiki/)这个自学指南, 就刷这里推荐的公开课, 这里我刷的是`CS106L`, 同时建议把杜克大学的C语言的课程[Introductory C Programming Specialization](https://www.coursera.org/specializations/c-programming)也刷了

**第三遍: 阅读电子书学习现代C++特性**  
这一阶段, 我学习的目标主要是涉及`C++`的新特性, 包括智能指针、右值引用、模板元编程这一类

**推荐的学习方法**  
其实我列举出我的3遍`C++`学习路线, 不是为了说明`C++`真的要学三遍, 而是为了说明方法很重要, 我学3遍的原因还不是因为没有找到合适的途径. 现在要推荐的话, 我如下推荐:

1. 电子教程 or 刷视频快速学习`C++`基础知识, 视频推荐[B站](https://www.bilibili.com/video/BV1o8411x7K3/?spm_id_from=333.1387.upload.video_card.click&vd_source=213e02ac84a7c6ef251b9f5d6691e21c), 电子书推荐(零基础C++)[[https://www.yuque.com/lianlianfengchen-cvvh2/zack]](https://www.yuque.com/lianlianfengchen-cvvh2/zack]), 其中这份电子书不仅仅是基础, 包括后续的高阶知识, 强烈推荐
2. 刷视频课 or 电子书重点学习`C++`的新特性, 这里的电子书包括: [现代C++教程](https://changkun.de/modern-cpp/zh-cn/00-preface/), 视频课推荐之前提到的[CS106L](https://web.stanford.edu/class/cs106l/)

> 顺带提一句, 我个人十分不推荐使用`C++ Primer Plus`这些经典书籍进行学习, 因为他们的可交互性太差了。学习编程语言，敲代码才能熟练。电子书或者视频可见的代码内容能一键复制，极大地提高了效率，而纸质书则非常并不方便

# [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#3-%E9%98%B6%E6%AE%B53-%E9%A1%B9%E7%9B%AE%E5%AE%9E%E8%B7%B5 "3 阶段3-项目实践")3 阶段3-项目实践

在完成了四大件的学习后, 基本上就可以开始做项目了。这里的项目可以直接用之前学习四大件时候的公开课项目来做, 也可以自己找网上的项目。

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#3-1-TinyWebServer-%E7%83%82%E5%A4%A7%E8%A1%97-%E5%81%9A%E8%BF%87 "3.1 TinyWebServer(烂大街/做过)")3.1 TinyWebServer(烂大街/做过)

[TinyWebServer](https://github.com/qinguoyi/TinyWebServer)虽然已经烂大街了, 但是项目本身的内容还是非常值得学习的。同时，这个项目基本上就是《Linux高性能服务器编程》这本书附带项目的进阶版本， 因此强烈建议阅读这本书。

![HPLSP](https://tonixwd.github.io/images/Notes/HPLSP.png)

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#3-2-%E5%8D%8F%E7%A8%8B%E5%BA%93-%E5%81%9A%E8%BF%87 "3.2 协程库(做过)")3.2 协程库(做过)

网上的协程库项目非常多，尤其是在`C++ 20`以前, 协程库是热门轮子项目。这里最推荐的是：[sylar](https://github.com/sylar-yin/sylar), 其本身是一个C++高性能分布式服务器框架, 协程只是其中的一个模块, 该项目有全套的视频课程, 可以参考。

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#3-3-RPC%E6%A1%86%E6%9E%B6-%E7%9C%8B%E8%BF%87%E6%B2%A1%E5%81%9A%E8%BF%87 "3.3 RPC框架(看过没做过)")3.3 RPC框架(看过没做过)

`RPC`框架也是非常热门的轮子项目, 这里推荐一个开源的`RPC`框架：[tinyrpc](https://github.com/Gooddbird/tinyrpc), 同样是文档齐全易于学习。（不过我只是看过没做过，毕竟人的精力是有限的）

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#3-4-%E5%8D%95%E6%9C%BAKV%E5%AD%98%E5%82%A8%E5%BC%95%E6%93%8E-%E4%B8%AA%E4%BA%BA%E5%B9%BF%E5%91%8A "3.4 单机KV存储引擎(个人广告)")3.4 单机KV存储引擎(个人广告)

这一小节是给自己打个广告, 我自己基于`LSM Tree`开发了一个KV存储引擎, 且兼容`Redis`的`Resp`协议, 换句话说可以代替`redis-server`作为`redis`的后端(相信大家知道这个创新性了), 欢迎大家来学习: [Toni-LSM](https://github.com/ToniXWD/toni-lsm)


# 4 阶段4-常用工具学习(秋招/实现面试必考)

在完成上面所有的学习路线后, 就可以开始准备秋招或者实习了, 但还需要额外学习一些常用工具, 也就是组件中间件什么的:

## 4.1 Docker/K8S

我学习`Docker`是基于这个视频课程: [https://kodekloud.com/courses/docker-for-the-absolute-beginner/](https://kodekloud.com/courses/docker-for-the-absolute-beginner/)

`K8S`相对`Docker`来说更复杂, 但属于是锦上添花的技术栈, 这里推荐在线教程 [Kubernetes 练习手册](https://k8s-tutorials.pages.dev/)

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#4-2-Redis "4.2 Redis")4.2 Redis

`Redis`的学习分为2部分:

**第一部分: Redis的使用**  
这一部分直接到Bilibili上搜一份视频课程就可以了, 你英文好也可以看官方文档

**第二部分: Redis的原理**  
这部分如果有时间, 推荐阅读书籍《Redis设计与实现》

![redis-book](https://tonixwd.github.io/images/Notes/redis-book.png)

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#4-3-Nginx "4.3 Nginx")4.3 Nginx

和`Redis`类似, 基础的知识找个视频教程就可以了, 进阶学习我推荐《深入理解Nginx》

![Nginx-book](https://tonixwd.github.io/images/Notes/Nginx-book.png)

## [](https://tonixwd.github.io/2025/04/23/%E9%9A%8F%E7%AC%94/25%E5%B1%8A%E7%9A%84C++%E5%BC%80%E5%8F%91%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB/#4-4-Linux%E7%8E%AF%E5%A2%83%E5%B7%A5%E5%85%B7%E9%93%BE "4.4 Linux环境工具链")4.4 Linux环境工具链

这里的工具链包括`Git/Cmake/Make/GDB`等, 每一个都不难, 只需要搜索对应的视频教程学习就好, 这里不展开介绍了