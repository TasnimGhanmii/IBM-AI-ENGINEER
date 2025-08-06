import torch 
from torch.utils.data import Dataset,DataLoader
from torch import nn, optim

torch.manual_seed(1)

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 1 * self.x - 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
     # Getter
    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
dataset=Data()
loader=DataLoader(dataset,batch_size=1)

class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

criterion = nn.MSELoss()
model = linear_regression(1,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train_model_SGD(iter):
    for epoch in range(iter):
        for x,y in loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


train_model_SGD(10)

