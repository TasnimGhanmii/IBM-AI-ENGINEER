import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#data
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())

#softmax classifier
class SoftMax(nn.Module):
    def __init__(self, input_size, ouput_size):
        super(SoftMax,self).__init__()
        self.linear=nn.Linear(input_size,ouput_size)

    def forward(self,x):
        z=self.linear(x)
        return z

