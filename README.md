

### 1.运行环境安装及配置
  \*提交代码均在 **Python 3.6、Pytorch 1.1** 环境下成功运行.

  \*安装 Anaconda 并配置 Python 3.6
  
  \*在[Anaconda](http://jianshu.com)官方网站下载 Windows 版本软件
  
   [Anaconda 2019.03 for Windows Installer](https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe)
  
  \*下载完成后，打开软件选择默认设置完成安装.
  
  
  \*配置 Pytorch 
  ```
  #打开 Anaconda Prompt，创建 Pytorch 环境
    conda create -n pytorch python=3.6
  #pytorch为环境名称，python版本为3.6
  
  #切换到pytorch环境
    activate pytorch
  
  #安装cpu版pytorch，版本为1.1
    conda install pytorch-cpu torchvision-cpu -c pytorch
  
  #验证是否安装成功，输入python进入编辑器
    import torch
    torch.__version__
  #输出版本信息即为配置成功.
  ```

  \*编辑器可使用 Pycharm 或者 Vscode,配置编辑器的运行环境百度一下你就知道.
  
### 2.Python及Pytorch基础学习  
  \*依据实际需要，主要学习了解以下基础：  
      1)Python 文件读写操作：读写'txt','json','csv'文件；  
      2)Python 函数用法；  
      3)Python 类的用法；  
      4)Python list的用法及操作；  
      5)Python numpy包的使用；  
      ......  
      
  \*Pytorch 快速入门：  
      可根据以下教程顺序快速学习Pytorch相关的使用方法与构建深度网络的基本框架：  
      [PyTorch 中文手册](https://github.com/zergtant/pytorch-handbook.git)  
      
### 3.机器学习相关原理：  
  \*主要了解经典、常用的模型：  
      1)[卷积神经网络(CNN)](https://www.cnblogs.com/skyfsm/p/6790245.html)  
      2) 循环神经网络(RNN)：  
         [长短时记忆(LSTM)](https://www.cnblogs.com/wangduo/p/6773601.html?utm_source=itdadao&utm_medium=referral)  
         [门控循环单元(GRU)](https://www.cnblogs.com/jiangxinyang/p/9376021.html)  
         ......  
  \*图像处理主要使用CNN,RNN主要用在自然语言处理方面；  
  
### 4.图像分类比赛：  
  \*开始一个不熟悉的任务，从入门到放弃的主要思路：  
      1)了解‘图像分类’已有的方法和模型：使用Google学术搜索‘图像分类综述’，论文越新越好；  
      2)根据综述提到的方法，从最新的模型开始，统计所有相关的模型方法；  
      3)按时间顺序从近到远去了解模型原理与思路，查找相关实现代码来复现；一般越新的模型效果越好。  
