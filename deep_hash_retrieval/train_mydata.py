from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import AlexNet
import resnet50_hash
from PIL import Image
# import cv2
batch_size = 16
num_classes = 10
learning_rate = 0.001
epoch = 200
pre_epoch = 0

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomRotation(60),
    transforms.ToTensor(),

    transforms.Normalize((.4, .4, .4), (.08, .08, .08))
])
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.4, .4, .4), (.08, .08, .08))
])

train_dir = './mydataset/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

val_dir = './mydataset/test'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)


print(train_datasets.classes)
net = resnet50_hash.resnet50(False,10)


flag_pre_train = True
if flag_pre_train == True:
    model_dict = net.state_dict()
    state_dict = torch.load('./weight/40xxxxx.pth')
    
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
   
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print("load pretrain model success!")

if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


epochs = 200

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print("learning rate", param_group['lr'])


def train():
    for epoch in range(0,epochs):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        adjust_learning_rate(optimizer)
       
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in val_dataloader:
                net.eval()
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                _,outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('test classification accuracy of gg_model is：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
            print('Saving model......')
           
    print("Training Finished, TotalEPOCH=%d" % epoch)

def run():
    img_class = ["1","10","2","3","4","5","6","7","8","9"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50_hash.resnet50(False,10)
    state_dict = torch.load('./weight/150xxxxx.pth')
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(state_dict)
    model.eval() 

    img_path = './9.jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    img = trans(img)
    img = Variable(img).cuda()
    img = img.unsqueeze(0)
    output = model(img)
    prob = F.softmax(output, dim=1)
    _, predicted = torch.max(output.data, 1)
    print(prob,predicted.item())
    print("predict is ",img_class[predicted.item()])
    plt.show()

if __name__ == "__main__":
    train()