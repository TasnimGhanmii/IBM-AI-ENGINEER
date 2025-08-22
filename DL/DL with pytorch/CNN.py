import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
z=conv(image)
print("z:",z)
print("shape:",z.shape[2:4])

K=2
conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=K)
M=4
image1=torch.ones(1,1,M,M)
z1=conv1(image1)
print("z1:",z1)
print("shape:",z1.shape[2:4])

conv3 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=2)
z3=conv3(image1)
print("z3:",z3)
print("shape:",z3.shape[2:4])

conv4 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3)
z4=conv4(image1)
print("z4:",z4)
print("z4:",z4.shape[2:4])

conv5 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3,padding=1)
z5=conv5(image1)
print("z5:",z5)
print("z5:",z4.shape[2:4])

