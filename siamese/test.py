import Siamese
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
from train_data import SiameseNetwork
import siamese_window
import myres50
print("test")
test_dir='../data/test/'
batch_size = 1
model = myres50.resnet50(num_classes=50)
if torch.cuda.is_available():
    model=model.cuda()
    print("gpu")
