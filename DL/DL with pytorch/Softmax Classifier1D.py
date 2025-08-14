import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

torch.manual_seed(0)

#synthetic data
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        #creates a tensor of zeros with the same number of elements as the number of rows in self.x
        self.y = torch.zeros(self.x.shape[0])
        #assigns the value 1 to the elements in self.y where the condition is True
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        #converts to integer true=>1 & false=>0
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
dataset=Data()

#softmax classifier
model = nn.Sequential(nn.Linear(1, 3))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
trainloader = DataLoader(dataset = dataset, batch_size = 5)

LOSS=[]
def train_model(epochs):
    for epoch in range(epochs):
        for x,y in trainloader:
            optimizer.zero_grad()
            yhat=model(x)
            loss=criterion(yhat,y)
            LOSS.append(loss)
            loss.backward()
            optimizer.step()

train_model(300)

z =  model(dataset.x)
_, yhat = z.max(1)
print("The prediction:", yhat)

correct = (dataset.y == yhat).sum().item()
accuracy = correct / len(dataset)
print("The accuracy: ", accuracy)

Softmax_fn=nn.Softmax(dim=-1)
Probability =Softmax_fn(z)
for i in range(3):
    print("probability of class {} isg given by  {}".format(i, Probability[0,i]) )
