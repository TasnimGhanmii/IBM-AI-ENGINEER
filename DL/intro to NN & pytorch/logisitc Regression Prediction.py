import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 


torch.manual_seed(2)
z = torch.arange(-100, 100, 0.1).view(-1, 1)
sig = nn.Sigmoid()
yhat = sig(z)

plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())


#model build
x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
yhat = model(x)
print("The prediction: ", yhat)
yhat = model(X)
print("The prediction result: \n", yhat)

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
yhat = model(x)
print("The prediction: ", yhat)
#prediction multiple samples
yhat = model(X)
print("The prediction: ", yhat)

#using custom models
class logistic_regression(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

model = logistic_regression(1)   
x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
yhat = model(x)
print("The prediction result: \n", yhat)
yhat = model(X)
print("The prediction result: \n", yhat)

model = logistic_regression(2)

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
yhat = model(x)
print("The prediction result: \n", yhat)
yhat = model(X)
print("The prediction result: \n", yhat)