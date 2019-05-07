

### 1.运行环境
  提交代码均在 **Python 3.6、Pytorch 1.1** 环境下成功运行.

#### 1.1运行环境安装与配置
  1.1.1 安装 Anaconda 并配置 Python 3.6
  
  #在[Anaconda](http://jianshu.com)官方网站下载 Windows 版本软件
  
   [Anaconda 2019.03 for Windows Installer](https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe)
  
  #下载完成后，打开软件选择默认设置完成安装.
  
  1.1.2 配置 Pytorch 
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
#### 1.2 编辑器
  #可使用 Pycharm 或者 Vscode
  #配置编辑器运行环境百度一下你就知道.
  
