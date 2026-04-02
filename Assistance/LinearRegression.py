import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

import sklearn.model_selection as selection
import matplotlib.pyplot as plt

# Model the loss function
mse = nn.MSELoss()
def loss_function(predictions, actual):
    return mse(predictions, actual)

# Helper function to plot loss curve
def plot_loss(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)

# Wireup the trainable tensors
class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        w = th.tensor(np.random.random(size=n_features), dtype=th.float32)
        c = th.tensor(np.random.random(), dtype=th.float32)
        self.w = nn.Parameter(w)
        self.c = nn.Parameter(c)
    
    def predict(self, X):
        return  self.w @ th.transpose(X, 0, 1) + self.c
        #tensors stored as collection of rows
        # 1 X n (M X n)T + c)
        #1xn (n x M) + c
        # 1 x M
    
    # instead of saying model.predict(X)
    # this allows us to say model(X)
    def forward(self, X):
        return self.predict(X)
    
    def loss(self, X, y):
        predictions = self.predict(X)
        return loss_function(predictions, y)

# load data that we plan to use
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)
nrows, ncols = X.shape # ncols is the number of features

X_train, X_test, y_train, y_test = selection.train_test_split(X, y, test_size=0.2, shuffle=True)


# Convert Trainind data to tensors
X_train = th.tensor(X_train, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)

# Set up the usual gradient descent loop
model = LinearRegression(ncols)
lr = 0.3
optimizer = optim.RMSprop(model.parameters(), lr=lr) # Modified SGD
num_iters = 2000
loss_curve = []

for i in range(num_iters):
    optimizer.zero_grad()
    loss_value = model.loss(X_train, y_train)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimizer.step()

# Plot the loss curve
plot_loss(loss_curve)
w = model.w.detach()
c = model.c.detach()
print(w, c)

#Your code here
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

import sklearn.model_selection as selection
import matplotlib.pyplot as plt

# Model the loss function
mse = nn.MSELoss()
def loss_function(predictions, actual):
    return mse(predictions, actual)

# Helper function to plot loss curve
def plot_loss(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)

# Wireup the trainable tensors
class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        w = th.tensor(np.random.random(size=n_features), dtype=th.float32)
        c = th.tensor(np.random.random(), dtype=th.float32)
        self.w = nn.Parameter(w)
        self.c = nn.Parameter(c)

    def predict(self, X):
        return  self.w @ th.transpose(X, 0, 1) + self.c
        #tensors stored as collection of rows
        # 1 X n (M X n)T + c)
        #1xn (n x M) + c
        # 1 x M

    # instead of saying model.predict(X)
    # this allows us to say model(X)
    def forward(self, X):
        return self.predict(X)

    def loss(self, X, y):
        predictions = self.predict(X)
        return loss_function(predictions, y)

# load data that we plan to use
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)
nrows, ncols = X.shape # ncols is the number of features

X_train, X_test, y_train, y_test = selection.train_test_split(X, y, test_size=0.2, shuffle=True)


# Convert Trainind data to tensors
X_train = th.tensor(X_train, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)

# Set up the usual gradient descent loop
model = LinearRegression(ncols)
lr = 0.3
optimizer = optim.RMSprop(model.parameters(), lr=lr) # Modified SGD
num_iters = 2000
loss_curve = []

for i in range(num_iters):
    optimizer.zero_grad()
    loss_value = model.loss(X_train, y_train)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimizer.step()

# Plot the loss curve
plot_loss(loss_curve)
w = model.w.detach()
c = model.c.detach()
print(w, c)

X_test = th.tensor(X_test, dtype=th.float32)
predictions = model.predict(X_test)
predictions = predictions.detach().numpy()
print(metrics.mean_squared_error(y_test, predictions))

from sklearn import metrics
X_test = th.tensor(X_test, dtype=th.float32)
predictions = model.predict(X_test)
predictions = predictions.detach().numpy()
print(metrics.mean_squared_error(y_test, predictions))

class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.l1 = nn.Linear(n_features, 1, bias=True) # n_features inputs, return output with 1 dimensiion, include an intercept

    def predict(self, X):
        return self.l1(X).reshape(-1)

    # instead of saying model.predict(X)
    # this allows us to say model(X)
    def forward(self, X):
        return self.predict(X)

    def loss(self, X, y):
        predictions = self.predict(X)
#         print(predictions)
#         print(y)
        return loss_function(predictions, y)

model = LinearRegression(ncols)
lr = 0.3
optimizer = optim.RMSprop(model.parameters(), lr=lr) # Modified SGD
num_iters = 2000
loss_curve = []

for i in range(num_iters):
    optimizer.zero_grad()
    loss_value = model.loss(X_train, y_train)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimizer.step()

plot_loss(loss_curve)


X_test = th.tensor(X_test, dtype=th.float32)
predictions = model.predict(X_test)
predictions = predictions.clone().detach()
print("MSE", metrics.mean_squared_error(y_test, predictions))
print("MAE", metrics.mean_absolute_error(y_test, predictions))

