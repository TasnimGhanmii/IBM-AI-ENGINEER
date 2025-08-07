from torch import nn
import torch
torch.manual_seed(1)

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)

def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat


x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
yhat = forward(X)
print("The result: ", yhat)

#using Linear class
model = nn.Linear(2, 1)
yhat = model(x)
print("The result: ", yhat)

# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)


#using cutom model
class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat
    

model = linear_regression(2, 1)

yhat = model(x)
print("The result: ", yhat)

yhat = model(X)
print("The result: ", yhat)