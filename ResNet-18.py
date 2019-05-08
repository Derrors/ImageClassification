# coding: utf-8
# @derrors 2019-05-08
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


# 是否使用 GPU 进行处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, classes_num=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, classes_num)

    def make_layer(self, residual_block, channels, blocks_num, stride):
        strides = [stride] + [1] * (blocks_num - 1)
        layers = []
        for stride in strides:
            layers.append(residual_block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def ResNet18():
    return ResNet(ResidualBlock)

def load_data(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR10('./data/CIFAR10', train=True, transform=transform_train, download=True)
    testset = CIFAR10('./data/CIFAR10', train=False, transform=transform_test, download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset, testset, trainloader, testloader


if __name__ == "__main__":
    epoches = 100
    batch_size = 128
    learning_rate = 0.1

    trainset, testset, trainloader, testloader = load_data(batch_size)
    resnet = ResNet18().to(device)
    criterian = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

     # 训练网络
    print('Training Starting-------------------------')
    for i in range(epoches):
        train_loss = 0.0
        train_acc = 0.0
        for (train_input, train_label) in trainloader:
            train_input, train_label = train_input.to(device), train_label.to(device)
            # 在训练之前，必须先清零梯度缓存
            optimizer.zero_grad()
            train_output = resnet(train_input)
            # 计算误差
            loss = criterian(train_output, train_label)
            loss.backward()
            # 参数更新
            optimizer.step()
            # 计算训练过程的损失大小及准确率
            train_loss += loss.item()
            _, predict = torch.max(train_output, 1)
            correct_num = (predict == train_label).sum().item()
            train_acc += correct_num
        train_loss /= len(trainset)
        train_acc /= len(trainset)
        print('[%d/%d] Loss: %.4f, Accuracy: %.2f %%' % (i+1, epoches, train_loss, train_acc*100))

    # 测试网络
    print('Tseting Starting-------------------------')
    correct = 0
    total = 0
    # 测试时，无需进行梯度计算与参数更新
    with torch.no_grad():
        for (test_input, test_label) in testloader:
            test_input, test_label = test_input.to(device), test_label.to(device)
            test_output = resnet(test_input)
            _, predicted = torch.max(test_output.data, 1)
            correct += (predicted == test_label).sum().item()
            total += test_label.size(0)
        print('Accuracy of test images: %.2f %%' % (100 * correct / total))
