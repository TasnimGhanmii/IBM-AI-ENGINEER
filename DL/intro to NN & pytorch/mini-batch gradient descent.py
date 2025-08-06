import torch 
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

X=torch.arange(-3,3,0.1).view(-1,1)
f=X-1
Y=f+0.1*torch.randn(X.size)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
lr = 0.1

LOSS_BGD = []

def forward(x):
    return w*x+b

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

#BGD
def train_model_BGD(epochs):
    for epoch in range(epochs):
        yhat=forward(X)
        loss=criterion(yhat,Y)
        LOSS_BGD.append(loss)
        loss.backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

train_model_BGD(10)



class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * X - 1
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    # Get length
    def __len__(self):
        return self.len
    
dataset=Data()
trainloader=DataLoader(dataset=dataset,batch_size=1)

#SGD
LOSS_SGD = []
lr = 0.1
def train_model_SGD(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        LOSS_SGD.append(criterion(forward(X), Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_SGD(10)


#mini-batch 
trainloader=DataLoader(dataset=dataset,batch_size=3)
def train_model_Mini5(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        #train loader 
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()