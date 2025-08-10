import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import torch.optim as optim

# Data preprocessing
data = pd.read_csv('league_of_legends_data_large.csv')
X = data.drop('win', axis=1)
y = data['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use transform instead of fit_transform for the test set

X_train = torch.tensor(X_train, dtype=torch.float32)  # Use float32 for numerical stability
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for BCELoss
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Model build
class LogisticRegression(nn.Module):
    def __init__(self, n_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Model training
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate accuracy
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    test_accuracy = (y_pred_class == y_test).float().mean().item() * 100
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')

# Optimization & evaluation
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate accuracy
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    test_accuracy = (y_pred_class == y_test).float().mean().item() * 100
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')

# Visualization
# Confusion Matrix
y_pred_test_class = y_pred_class.numpy()
conf_matrix = confusion_matrix(y_test.numpy(), y_pred_test_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test.numpy(), y_pred_test_class))

# ROC Curve
y_pred_test = y_pred.numpy()
fpr, tpr, _ = roc_curve(y_test.numpy(), y_pred_test)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Model saving
# Save the model
torch.save(model.state_dict(), 'logistic_regression_model.pth')

# Load the model
model_loaded = LogisticRegression(input_dim)
model_loaded.load_state_dict(torch.load('logistic_regression_model.pth'))

# Ensure the loaded model is in evaluation mode
model_loaded.eval()

# Evaluate the loaded model
with torch.no_grad():
    y_pred_test = model_loaded(X_test)
    y_pred_test_class = (y_pred_test >= 0.5).float()

accuracy = accuracy_score(y_test.numpy(), y_pred_test_class.numpy())
print(f'Test Accuracy of the loaded model: {accuracy * 100:.2f}%')

# Hyperparameter tuning
learning_rates = [0.01, 0.05, 0.1]
best_accuracy = 0
best_lr = None

# Train and evaluate the model for each learning rate
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    
    # Reinitialize the model and optimizer
    input_size = X_train.shape[1]
    model = LogisticRegression(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)  # L2 regularization with weight_decay=0.01
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate the model on the test dataset
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test_class = (y_pred_test >= 0.5).float()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test.numpy(), y_pred_test_class.numpy())
    print(f'Test Accuracy with learning rate {lr}: {accuracy * 100:.2f}%')
    
    # Track the best learning rate
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lr = lr

print(f"Best learning rate: {best_lr} with test accuracy: {best_accuracy * 100:.2f}%")

# Feature importance
# Extract the weights of the linear layer
weights = model.linear.weight.data.numpy().flatten()

# Create a DataFrame for feature importance
feature_names = data.drop('win', axis=1).columns  # Assuming `data` is the original DataFrame
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': weights})

# Sort the features based on the absolute value of their importance
feature_importance = feature_importance.sort_values(by='Importance', key=abs, ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Logistic Regression Model')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()

# Print the sorted feature importances
print(feature_importance)