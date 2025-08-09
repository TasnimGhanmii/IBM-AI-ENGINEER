import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

torch.manual_seed(0)

class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
data_set = Data()

class Logistic(nn.Module):
    def __init__(self,n_inputs):
        super(Logistic,self).__init__()
        self.linear
