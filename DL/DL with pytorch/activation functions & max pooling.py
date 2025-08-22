import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
Z=conv(image)
relu = nn.ReLU()
relu(Z)

image1=torch.zeros(1,1,4,4)
image1[0,0,0,:]=torch.tensor([1.0,2.0,3.0,-4.0])
image1[0,0,1,:]=torch.tensor([0.0,2.0,-3.0,0.0])
image1[0,0,2,:]=torch.tensor([0.0,2.0,3.0,1.0])

max1=torch.nn.MaxPool2d(2,stride=1)
max1(image1)

max1=torch.nn.MaxPool2d(2)
max1(image1)
