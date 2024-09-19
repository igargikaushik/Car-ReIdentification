import torch
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import argparse
import os
import resnet50_hash

train_binary = torch.load('./result/train_binary')
train_data = torch.load('./result/train_data')
parser = argparse.ArgumentParser(description='Image Search')
parser.add_argument('--pretrained', type=str, default=92, metavar='pretrained_model',
                    help='')
parser.add_argument('--querypath', type=str, default='./4404000000002940408492.jpg', metavar='',
                    help='')
args = parser.parse_args()

transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.4, .4, .4), (.08, .08, .08))
    ])

query_pic = Image.open(args.querypath)
query_pic = transform_test(query_pic)

net = resnet50_hash.resnet50(False, 10)
net.load_state_dict(torch.load('./weight/40xxxxx.pth'))

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    query_pic = query_pic.cuda().unsqueeze(0)
net.eval()
outputs, _ = net(query_pic)
print("48feature",outputs[0])
query_binary = (outputs[0] > 0.5).cpu().numpy()

trn_binary = train_binary.cpu().numpy()

print("train_binary",trn_binary,trn_binary.shape)

query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length


def eucldist_generator(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return sum((x - y)**2 for x, y in zip(coords1, coords2))**0.5
img_list = open('./mydataset/train.txt').readlines()


def coarse_research():
    sort_indices = np.argsort(query_result)
    for i in sort_indices:
        print(img_list[i])
def fine_research():
   
    query_result_eu = {}
    for i, j in enumerate(train_data):
        if query_result[i] == 0:
            query_result_eu[i] = (eucldist_generator(j, outputs[0]).data)

    query_result_list = list(query_result_eu.items())
    query_result_list.sort(key=lambda x: x[1], reverse=False)
    print(query_result_list)
    for i in query_result_list:
        print(img_list[i[0]])
if __name__ == '__main__':
    fine_research()