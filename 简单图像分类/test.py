# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchvision import transforms
from ResNet import ResNet18
from PIL import Image

dic = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

img = Image.open('dog.jpg')

trans = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

img_tensor = trans(img).unsqueeze(0)


model = ResNet18()
model.load_state_dict(torch.load('./result/resnet.pth', map_location=torch.device('cpu')))

model.eval()

output = model(img_tensor).detach().numpy()
result = np.argmax(output)
print(dic[result])
