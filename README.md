# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.




## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.



### STEP 3: 
Define the RNN model architecture.



### STEP 4: 
Summarize, compile with loss and optimizer



### STEP 5: 
Train the model with loss tracking.



### STEP 6:
Predict on test data, plot actual vs. predicted prices.





## PROGRAM

### Name: THIRUMALAI K

### Register Number: 212224240176

```python
# -------------------------------------------------------------
# 🎯 Stock Price Prediction using RNN (Complete Project)
# Developed by: [THIRUMALAI K]
# -------------------------------------------------------------

# ✅ Step 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ✅ Step 2: Load & Preprocess Dataset
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

# Convert to tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ✅ Prepare DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ✅ Step 3: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # Take the output from last timestep

# ✅ Step 4: Initialize Model
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ Optional: Print Summary
!pip install -q torchinfo
from torchinfo import summary
summary(model, input_size=(64, 60, 1))

# ✅ Step 5: Setup Loss + Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ Step 6: Train the Model
epochs = 35
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb)
        loss = criterion(output, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# ✅ Step 7: Plot Training Loss
print('Name: THIRUMALAI K')
print('Register Number: 212224240176')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ✅ Step 8: Predict on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse scale predictions
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# ✅ Plot Predicted vs Actual Prices
print('Name: THIRUMALAI K')
print('Register Number: 212224240176')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.tight_layout()
plt.show()

# ✅ Final Prediction Output
print(f'Last Predicted Price: {predicted_prices[-1][0]:.2f}')
print(f'Actual Price:          {actual_prices[-1][0]:.2f}')


```

### OUTPUT

## Training Loss Over Epochs Plot

![alt text](image.png)

## True Stock Price, Predicted Stock Price vs time
![alt text](image-1.png)
### Predictions
![alt text](image-2.png)

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
