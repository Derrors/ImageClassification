
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import dpn

# 是否使用 GPU 进行处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置参数
epoches = 100
batch_size = 1024
learning_rate = 0.1
train_path = './train.txt'
test_path = './test.txt'
output_path = './output/'


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        fh = open(path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.imgs)


# 数据预处理
def load_data(batch_size, data_path, shuffle):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_set = MyDataset(path=data_path, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    return data_set, data_loader


def model_train():
    # 加载数据、构建模型、选择损失函数和优化器
    trainset, trainloader = load_data(batch_size, train_path, shuffle=True)
    # 创建 DPN 模型
    model = dpn.dpn131(200).to(device)
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    print('Training Starting-------------------------')
    for i in range(epoches):
        # 训练模型
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for (train_input, train_label) in trainloader:
            train_input, train_label = train_input.to(device), train_label.to(device)
            # 在训练之前，必须先清零梯度缓存
            optimizer.zero_grad()
            train_output = model(train_input)
            # 计算误差
            loss = criterion(train_output, train_label)
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
        print('[%d/%d] Training_Loss: %.6f, Training_Accuracy: %.4f %% ' % (i+1, epoches, train_loss, train_acc*100))

        if train_acc >= 0.999:

            model_path = output_path + 'dpn1024_' + str(i+1) + '.pth'
            torch.save(model.state_dict(), model_path)

            csv_path = output_path + 'dpn1024_' + str(i+1) + '.csv'
            model_test(model, csv_path)
    
    print('Training_Succeed!')
    return 


def model_test(Model, csv_path=None):
    model = Model

    # 测试模型
    model.eval()
    images = []
    labels = []
    _, testloader = load_data(batch_size, test_path, shuffle=False)

    # 测试时，无需进行梯度计算与参数更新
    with torch.no_grad():
        for (test_input, test_label) in testloader:
            test_input, test_label = test_input.to(device), test_label.to(device)
            test_output = model(test_input)
            _, predicted = torch.max(test_output.data, 1)
            labels.extend(predicted)

    output = open(csv_path, 'a', newline='')
    csv_write = csv.writer(output, dialect='excel')
    csv_write.writerow(['ImageName', 'CategoryId'])

    with open(test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            tmp2 = tmp[0].split('/')
            images.append(tmp2[2])
    for i in range(len(images)):
        csv_write.writerow([images[i], labels[i].item() + 1])
    
    print('Testing_Succeed!')
    return


if __name__ == '__main__':
    model_train()

