import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import glob
import random
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import myres50
import read_mat_file
import matplotlib.pyplot as plt


num_epoch=5000
train_dir='../data/train/'
batch_size=4
ln = 0.0001

flag = True
model = myres50.resnet50(num_classes=50)

if flag == True:
    model_dict = model.state_dict()
    state_dict = torch.load('resnet50-19c8e357.pth')
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



if torch.cuda.is_available():
    model=model.cuda()
    print("gpu")


window_dict = read_mat_file.get_car_windows()
print (window_dict)
def get_car_window(car_img):

    return window_img


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2,keepdim = True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2)
                                      +(label) * torch.pow(torch.clamp(self.margin
                                                                       - euclidean_distance, min=0.0), 2))

        return loss_contrastive


train_transforms = transforms.Compose([

    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomRotation(6),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.2, .2, .2))
])


class net_dataset(Dataset):
        def __init__(self,img_folder,imageFolderDataset,transform=None,should_invert=True):
            self.transform = transform
            self.should_invert = should_invert
            self.img_folder=img_folder
            self.imageFolderDataset=imageFolderDataset

        def __getitem__(self, item):
            self.img_folder_list=glob.glob(self.img_folder+'*')
            img0_path=self.img_folder_list[random.randint(0,len(self.img_folder_list)-1)]
            
            img0_class=img0_path.split('/')[-1]
           
            img_list=glob.glob(img0_path+'/*')
         
            img0_way=img_list[random.randint(0,len(img_list)-1)]
            
            should_get_same_img=random.randint(0,1)
         
            if should_get_same_img:
                img1_way=img_list[random.randint(0,len(img_list)-1)]
                img1_class=img0_class
            else:
                while True:
                    img1_path=self.img_folder_list[random.randint(0,len(self.img_folder_list)-1)]
                    img1_class=img1_path.split('/')[-1]
                    if img1_class != img0_class:
                        img_list=glob.glob(img1_path+'/*')
                        img1_way=img_list[random.randint(0,len(img_list)-1)]
                        break

            img0_name = img0_way.split('/')[-1]
            img0_rect = window_dict[img0_name]
            img1_name = img1_way.split('/')[-1]
            img1_rect = window_dict[img1_name]
            

            img0=Image.open(img0_way)
            img1=Image.open(img1_way)
            window0 = img0.crop(img0_rect)
            window1 = img1.crop(img1_rect)
            img0=window0.resize((400,200))
            img1=window1.resize((400,200))

         


            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
         

            return img0, img1, torch.from_numpy(np.array([int(img1_class != img0_class)], dtype=np.float32))

        def __len__(self):
            print("data-len",len(self.imageFolderDataset.imgs))
            return len(self.imageFolderDataset.imgs)

def adjust_learning_rate(optimizer, epoch):
    lr = ln * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate",lr)



def train():
    print("data_floder")
    folder_dataset=dset.ImageFolder(root=train_dir)
    print("folder_dataset",folder_dataset)
    print("siamese_dataset")
    siamese_dataset = net_dataset(img_folder=train_dir,imageFolderDataset=folder_dataset,
                                  transform=train_transforms,
                                  should_invert=False)

    print('train_data')
    train_data=DataLoader(siamese_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_data))


    criterion=ContrastiveLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
   
    loss_all=[]
    correct_nums_i = 0
    for epoch in range(num_epoch):
        correct_nums = 0

        for i,data in enumerate(train_data,0):

        
            img0,img1,label=data
            img0, img1, label=Variable(img0).cuda(),Variable(img1).cuda(),Variable(label).cuda()
            optimizer.zero_grad()
            out1=model(img0)
            out2=model(img1)
            loss=criterion(out1,out2,label)
            loss.backward()
            optimizer.step()
            loss_all.append(loss.item())



            euclidean_distance = F.pairwise_distance(out1, out2)

            predict_label = euclidean_distance > 1.3
            label = label.view(4)
            label_uint8 = torch.tensor(label, dtype=torch.uint8)
            correct_nums += (predict_label.cpu()==label_uint8).sum().item()
            correct_nums_i += (predict_label.cpu()==label_uint8).sum().item()

            if i %60==0:

                print("epoch:",epoch)
                print(sum(loss_all[:])/60)
                print("predict_batch_size precison", correct_nums_i / 240)
                correct_nums_i = 0
                f = open("loss.txt","a+")
                f.write(str(sum(loss_all[:])/60)+'\n')
                f.close()
           
        f = open("loss.txt", "a+")
        f.write("precison  "+str(correct_nums/len(train_data)) + '\n')
        f.close()
        print("precion:",correct_nums/len(train_data))
        if epoch%1==0:
            torch.save(model.state_dict(),'../weights/'+str(epoch)+'.pth')


if __name__== "__main__":
    train()