import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#Use torch activation functions when you need a simple, stateless function.
#Use torch.nn activation functions when you need a stateful layer that can be integrated into a neural network model
torch.manual_seed(2)
z = torch.arange(-10, 10, 0.1,).view(-1, 1)

#sigmoid
sig=nn.Sigmoid()
yhat=sig(z)
#detatching the tensor from the computational graph by creating an independent clone=>the ops performed in the clone wouldn't affect the original tensor  
plt.plot(z.detach().numpy(),yhat.detach().numpy())
plt.xlabel('z')
plt.ylabel('yhat')
#built in fct
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

#tanh
TANH = nn.Tanh()
yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()
#or
yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

#Relu
RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())
#or
yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


# Plot the results to compare the activation functions
x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()