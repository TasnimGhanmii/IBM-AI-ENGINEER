import torch 
from torch import nn
from torch.nn import Linear

b=torch.tensor(-1.0,requires_grad=True)
w=torch.tensor(2.0,requires_grad=True)

def forward(x):
    yhat=b+2*x
    return yhat

x=torch.tensor([[1.0]])
print("prediction: ",forward(x))

#multiple inputs
x=torch.tensor([[1.0],[2.0]])
print("prediction: ",forward(x))

#using class Linear
torch.manual_seed(1)
lr=Linear(in_features=1,out_features=1,bias=True)
print("Parameters w and b: ", list(lr.parameters()))

print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())
print("weight:",lr.weight)
print("bias:",lr.bias)

x = torch.tensor([[1.0]])
z = torch.tensor([[1.0],[2.0]])
yhat = lr(x)
zhat = lr(z)
print("The prediction: ", yhat)
print("The prediction: ", zhat)


#using custom modules
class LR(nn.Module):

    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
        out=self.linear(x)
        return out

lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)  

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)

print("Python dictionary: ", lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())


