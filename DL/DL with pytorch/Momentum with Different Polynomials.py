import torch
import torch.nn as nn
import torch.optim as optim

# Set seed for reproducibility
torch.manual_seed(0)

# Define the model with one parameter (linear layer without bias)
class one_param(nn.Module):
    def __init__(self, input_size, output_size):
        super(one_param, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

# Instantiate the model
w = one_param(1, 1)

# Custom transformation: y = (w * x)^3
def cubic(yhat):
    return yhat ** 3

# Define optimizer and loss function
optimizer = optim.SGD(w.parameters(), lr=0.01, momentum=0)
criterion = nn.MSELoss()

true_weight = 2.0
x_train = torch.linspace(-10, 10, 100).reshape(-1, 1)  
y_true = (true_weight * x_train) ** 3  

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    yhat_linear = w(x_train)          
    yhat_cubic = cubic(yhat_linear)   
    loss = criterion(yhat_cubic, y_true)  
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
# Final learned weight
learned_weight = w.linear.weight.item()
print(f"Learned weight: {learned_weight:.4f}")
print(f"True weight: {true_weight}")

# Prediction example
with torch.no_grad():
    x_test = torch.tensor([[2.0]])
    pred_linear = w(x_test)
    pred_cubic = cubic(pred_linear)
    print(f"Input: {x_test.item()}, w*x: {pred_linear.item():.4f}, (w*x)^3: {pred_cubic.item():.4f}")