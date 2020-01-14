# coding: utf-8
# @derrors 2019-05-08
#
# ResNet18 模型结构：输入->卷积、BN、激活->残差层1->残差层2->残差层3->残差层4->池化->全连接->输出
#

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel
from torchvision.datasets import CIFAR10
from tqdm import tqdm

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 这里采用 conv3x3 -> BN -> ReLU -> conv3x3 -> BN 结构
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 直接传递
        self.shortcut = nn.Sequential()
        # 输出维度减半时， shortcut 也引应转换为相应的维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    # 一个残差块的前向传递

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet18 网络
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, classes_num=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 每个 layer 表示一个残差层
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, classes_num)
    # 创建残差层，每个残差层包含 blocks_num 个残差块

    def make_layer(self, residual_block, channels, blocks_num, stride):
        strides = [stride] + [1] * (blocks_num - 1)     # 形式为 [1, 1], [2, 1] 等
        layers = []
        for stride in strides:
            layers.append(residual_block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)                   # *表示将输入迭代器拆成一个个元素，即顺序链接各层

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

    trainset = CIFAR10('./data/CIFAR10', train=True, transform=transform_train, download=True)
    testset = CIFAR10('./data/CIFAR10', train=False, transform=transform_test, download=True)

    train_sampler = DistributedSampler(trainset)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    return trainset, testset, trainloader, testloader


if __name__ == "__main__":
    # 设置参数
    epoches = 50
    batch_size = 256
    learning_rate = 0.001
    best = 0
    # 加载数据、构建模型、选择损失函数和优化器
    trainset, testset, trainloader, testloader = load_data(batch_size)

    resnet = ResNet18().to(device)
    resnet = DistributedDataParallel(resnet, device_ids=[local_rank], output_device=local_rank)

    criterian = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=5e-4)

    print('Training Starting-------------------------')
    for i in range(epoches):
        # 训练模型
        train_loss = 0.0
        train_acc = 0.0
        resnet.train()
        for (train_input, train_label) in tqdm(trainloader):
            train_input = train_input.to(device)
            train_label = train_label.to(device)

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

        # 测试模型
        correct = 0
        total = 0
        resnet.eval()
        # 测试时，无需进行梯度计算与参数更新
        with torch.no_grad():
            for (test_input, test_label) in testloader:
                test_input = test_input.to(device)
                test_label = test_label.to(device)

                test_output = resnet(test_input)
                _, predicted = torch.max(test_output.data, 1)
                correct += (predicted == test_label).sum().item()
                total += test_label.size(0)
                test_acc = correct / total
            if test_acc > best:
                torch.save(resnet.state_dict(), './result/resnet.pth')

        print('[%d/%d] Training_Loss: %.4f, Training_Accuracy: %.2f %%  Testing_Accuracy: %.2f %%'
              % (i + 1, epoches, train_loss, train_acc * 100, test_acc * 100))
