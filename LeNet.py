# coding: utf-8                                                                   
# @derrors 2019-05-07

# LeNet 网络结构：输入->卷积层->激活->池化->卷积->激活->池化->卷积->激活->池化         
#                    ->全连接->激活->全连接->激活->全连接->激活->输出               

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# 使用 GPU 进行处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
def load_data(batch_size):
    # 数据处理器，可进行数据预处理操作
    trans_img = transforms.Compose([transforms.ToTensor()])
    # 下载图像数据
    trainset = CIFAR10('./data/CIFAR10', train=True, transform=trans_img, download=True)
    testset = CIFAR10('./data/CIFAR10', train=False, transform=trans_img, download=True)
    # 数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)

    return trainset, testset, trainloader, testloader

# 定义网络结构
class LeNet(nn.Module):
    # 构造函数
    def __init__(self):
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
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # 全连接层 in_features、out_features 分别表示输入特征数、输出特征数
            nn.Linear(in_features=8*8*120, out_features=120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Sigmoid(),
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
    learning_rate = 0.003
    batch_size = 128
    epoches = 30

    # 构建网络模型
    lenet = LeNet().to(device)
    trainset, testset, trainloader, testloader = load_data(batch_size)
    # 损失函数：交叉熵
    criterian = nn.CrossEntropyLoss(reduction='sum')
    # 优化方法：随机梯度下降 lenet.parameters() 返回 LeNet 网络中可学习的参数  lr为学习率
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)

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
        print('[%d/%d] Loss: %.4f, Accuracy: %.2f %%' % (i+1, epoches, running_loss, running_acc*100))

    # 测试网络
    print('Tseting Starting-------------------------')
    correct = 0
    total = 0
    # 测试时，无需进行梯度计算与参数更新
    with torch.no_grad():
        for (test_img, test_label) in testloader:
            test_img, test_label = test_img.to(device), test_label.to(device)
            test_output = lenet(test_img)
            _, predicted = torch.max(test_output.data, 1)
            correct += (predicted == test_label).sum().item()
            total += test_label.size(0)
        print('Accuracy of test images: %.2f %%' % (100 * correct / total))