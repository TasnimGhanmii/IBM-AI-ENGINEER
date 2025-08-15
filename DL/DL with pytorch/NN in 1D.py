import torch
import torch.nn as nn
from torch import sigmoid

#n PyTorch, gradients are accumulated by default. This means that if you call backward() multiple times, 
# the gradients will be added together. To avoid this, you need to reset the gradients to zero before each forward pass.

torch.manual_seed(0)

#network
class NN(nn.Module):
    def __init__(self, D_in,H, D_out):
        super(NN,self).__init__()
        #hidden layer
        #creates the first linear layer (fully connected layer) that maps the input features to the hidden layer.
        self.linear1 = nn.Linear(D_in, H)
        #creates the second linear layer that maps the hidden layer to the output layer
        self.linear2 = nn.Linear(H, D_out)
       

    # Prediction
    def forward(self, x):
        l1 = self.linear1(x)
        a1 = sigmoid(self.l1)
        l2=self.linear2(self.a1)
        yhat = sigmoid(self.linear2(self.a1))
        return yhat


#train    
def train(Y, X, model, optimizer, criterion, epochs=1000):
    cost = []
    total=0
    for epoch in range(epochs):
        total=0
        for y, x in zip(Y, X):
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #cumulative loss 
            total+=loss.item() 
        cost.append(total)
        if epoch % 300 == 0:    
            model(X)
    return cost 

#data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out



# Train the model
# size of input 
D_in = 1
# size of hidden layer 
H = 2
# number of outputs 
D_out = 1
# learning rate 
learning_rate = 0.1
# create the model 
model = NN(D_in, H, D_out)
#optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#train the model usein
cost_cross = train(Y, X, model, optimizer, criterion_cross, epochs=1000)
#predict
model(X)



