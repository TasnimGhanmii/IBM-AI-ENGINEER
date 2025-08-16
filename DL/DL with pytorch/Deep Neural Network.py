import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

torch.manual_seed(2)

#model
class NNSigmoid(nn.Module):
    def __init__(self,D_in,H1,H2,D_out):
        super(NNSigmoid,self).__init__()
        self.linear1=nn.Linear(D_in,H1)
        self.linear2=nn.Linear(H1,H2)
        self.linear3=nn.Linear(H2,D_out)

    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        x=self.linear3(x)
        return x 
    

class NNTanh(nn.Module):
    def __init__(self,D_in,H1,H2,D_out):
        super(NNTanh,self).__init__()
        self.linear1=nn.Linear(D_in,H1)
        self.linear2=nn.Linear(H1,H2)
        self.linear3=nn.Linear(H2,D_out)

    def forward(self,x):
        x=torch.tanh(self.linear1(x))
        x=torch.tanh(self.linear2(x))
        x=self.linear3(x)
        return x 
    
class NNRelu(nn.Module):
    def __init__(self,D_in,H1,H2,D_out):
        super(NNRelu,self).__init__()
        self.linear1=nn.Linear(D_in,H1)
        self.linear2=nn.Linear(H1,H2)
        self.linear3=nn.Linear(H2,D_out)

    def forward(self,x):
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.linear3(x)
        return x 
    
# Train the model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}  
    
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
        
        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
    
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff
    
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

input_dim = 28 * 28
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10
cust_epochs = 10

learning_rate = 0.01
model = NNSigmoid(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

learning_rate = 0.01
model_Tanh = NNTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
training_results_tanch = train(model_Tanh, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

learning_rate = 0.01
modelRelu = NNRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

# Compare the training loss
plt.plot(training_results_tanch['training_loss'], label='tanh')
plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

# Compare the validation loss
plt.plot(training_results_tanch['validation_accuracy'], label = 'tanh')
plt.plot(training_results['validation_accuracy'], label = 'sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label = 'relu') 
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')   
plt.legend()