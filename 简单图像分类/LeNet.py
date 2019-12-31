# coding: utf-8
# @derrors 2019-12-31
#
# LeNet 网络结构：输入->卷积层->激活->池化->卷积->激活->池化->卷积->激活->池化
#                    ->全连接->激活->全连接->激活->全连接->激活->输出
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100


# 是否使用 GPU 进行处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_loader(path):
    return Image.open(path).convert('RGB')


# 数据预处理
def load_data(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR100('../data/CIFAR100', train=True, transform=transform_train, download=True)
    testset = CIFAR100('../data/CIFAR100', train=False, transform=transform_test, download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset, testset, trainloader, testloader


# 定义网络结构
class LeNet(nn.Module):
    # 构造函数
    def __init__(self, classes_num=200):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            # 卷积层 in_channels、out_channels、kernel_size、stride、padding
            #          输入深度     输出深度     卷积核尺寸  滑动步长  补0填充
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 120, 5, stride=1, padding=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            # 全连接层 in_features、out_features 分别表示输入特征数、输出特征数
            nn.Linear(in_features=8 * 8 * 120, out_features=120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes_num),
            nn.Sigmoid()
        )
    # 前向传播过程

    def forward(self, x):
        out = self.conv(x)
        # reshape 操作
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # 参数设置
    learning_rate = 0.1
    batch_size = 256
    epoches = 30

    # 构建网络模型
    lenet = LeNet().to(device)
    trainset, trainloader = load_data(batch_size)
    # 损失函数：交叉熵
    criterian = nn.CrossEntropyLoss(reduction='sum')
    # 优化方法：随机梯度下降 lenet.parameters() 返回 LeNet 网络中可学习的参数  lr为学习率
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # 训练网络
    print('Training Starting-------------------------')
    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (train_img, train_label) in trainloader:
            train_img, train_label = train_img.to(device), train_label.to(device)
            # 在训练之前，必须先清零梯度缓存
            optimizer.zero_grad()
            train_output = lenet(train_img)
            # 计算误差
            loss = criterian(train_output, train_label)
            loss.backward()
            # 参数更新
            optimizer.step()
            # 计算训练过程的损失大小及准确率
            running_loss += loss.item()
            _, predict = torch.max(train_output, 1)
            correct_num = (predict == train_label).sum().item()
            running_acc += correct_num
        running_loss /= len(trainset)
        running_acc /= len(trainset)
        print('[%d/%d] Training_Loss: %.4f, Training_Accuracy: %.2f %% ' % (i + 1, epoches, running_loss, running_acc * 100))
