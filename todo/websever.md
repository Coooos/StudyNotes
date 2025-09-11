
[从零开始实现C++ TinyWebServer 全过程记录_tinywebserver要做多久-CSDN博客](https://blog.csdn.net/weixin_51322383/article/details/130464403)

[从零开始自制实现C++ High-Performance WebServer 全流程记录_c++ webserver项目 速成-CSDN博客](https://love6.blog.csdn.net/article/details/123754194)

[基于Linux下的vscode c/c++开发环境搭建详细教程_linux配置vscode c++环境-CSDN博客](https://blog.csdn.net/icacxygh001/article/details/120981354)

[建议立刻将 WSL + VSCode 作为你的最强生产力环境，起飞吧_wsl vscode-CSDN博客](https://blog.csdn.net/yanbober/article/details/138245581)

---

**本项目基于本站施磊老师的课程进行coding

---
# 环境搭建

[在wsl上配置vscode和c++环境 - 片刻的自由 - 博客园](https://www.cnblogs.com/7d1-z/p/18462132#:~:text=%E6%A3%80%E6%9F%A5%E6%98%AF%E5%90%A6%E5%AE%89%E8%A3%85%E4%BA%86g%2B%2B%E5%92%8Cgcc%20%E6%B2%A1%E6%9C%89%E5%88%99%E5%AE%89%E8%A3%85%E3%80%82%20%E5%86%99%E4%B8%80%E6%AE%B5cpp%E7%A8%8B%E5%BA%8F%E6%B5%8B%E8%AF%95%E4%B8%8B%20cout%3C%3C123%20%3C%3Cendl%3B%20%E7%BC%96%E8%AF%91%E3%80%81%E8%BF%90%E8%A1%8C,%E5%9C%A8windows%E6%9C%AC%E5%9C%B0%E7%9A%84vscode%E4%B8%8A%EF%BC%8C%E5%85%88%E5%AE%89%E8%A3%85%E6%8F%92%E4%BB%B6%E2%80%9CRemote%20-%20SSH%E2%80%9D%EF%BC%8C%E9%80%9A%E8%BF%87%E5%B7%A6%E4%B8%8B%E8%A7%92%E2%80%9C%E6%89%93%E5%BC%80%E8%BF%9C%E7%A8%8B%E7%AA%97%E5%8F%A3%E2%80%9D%E8%BF%9E%E6%8E%A5wsl%E7%9A%84linux%E4%B8%8A%EF%BC%8C%E5%B9%B6%E8%87%AA%E5%8A%A8%E5%AE%89%E8%A3%85vscode%E3%80%82%20%E5%88%9B%E5%BB%BAcpp%E6%96%87%E4%BB%B6%E6%97%B6%EF%BC%8C%E6%A0%B9%E6%8D%AE%E6%8E%A8%E8%8D%90%E5%AE%89%E8%A3%85%E6%8F%92%E4%BB%B6%E2%80%9CC%2FC%2B%2B%20Extension%20Pack%E2%80%9D%E5%88%B0linux%E3%80%82)

linux 装 gcc g++ gbd


linux 安装 boost muodu网络库

telnet 127.0.0.1 6000

{"msgid":1,"id":1,"password":"123456"}

mysql -u root -p

Kuang200354

USE chat

select * from User




|  1 | kjm  | 123456   | offline

[在 Ubuntu 上安装 Muduo 网络库的详细指南_ubuntu安装muduo-CSDN博客](https://blog.csdn.net/m0_74795952/article/details/144631342)






## 什么是webserve
**简单来说：想象Web服务器是一个餐厅的服务员。** 顾客（浏览器）告诉服务员（Web服务器）想要什么（请求一个URL）。服务员去厨房（文件系统或应用服务器）取食物（静态文件）或让厨师现做（动态内容），然后把做好的食物（HTTP响应）端给顾客。服务员负责整个点餐和上菜的过程。

这就是Web服务器的基础作用——让互联网上的信息能够被请求和传递。

一个WebServer如何和用户进行通信，就需要在浏览器输入域名或者IP地址:端口号，浏览器会先将你的域名解析成响应的IP地址(DNS)或者直接根据你的IP:Port向对应的Web服务器发送一个HTTP请求。这个过程需要TCP协议的三次握手建立和目标Web服务器的连接，然后HTTP协议生成针对该Web服务器的HTTP请求报文，这个报文里面包括了请求行、请求头部、空行和请求数据四个部分（后面再细讲），通过TCP/IP协议发送到Web服务器上。


### **核心功能：**

- **监听请求：** 持续监听特定的网络端口（通常是HTTP的80端口或HTTPS的443端口），等待客户端（浏览器）发来的请求。
- - **处理请求：** 接收客户端发来的HTTP/HTTPS请求（例如请求访问某个网页 `GET /index.html`）。
- 对于**静态内容**（如HTML文件、图片、CSS样式表、JavaScript文件）：服务器直接从其文件系统中找到对应的文件。
- 对于**动态内容**（如根据用户数据生成的页面）：服务器会将请求传递给其他程序（如PHP、Python、Node.js应用、Java Servlet等）处理，并接收其生成的结果。
- **发送响应：** 将请求的内容（文件内容或应用程序生成的结果）打包成HTTP/HTTPS响应消息，并发送回客户端浏览器。响应中通常包含状态码（如200表示成功，404表示未找到）、响应头和实际内容（HTML、图片等）。
- **管理连接：** 处理与客户端的网络连接（建立、维护、关闭）。

