For each of the following 
(a) Find the minimum point without using gradient descent 
(b) Give at least one iteration of the gradient descent by hand 
(c) Write code to solve using gradient descent  

Assuming we have a function with two variables, f(x,y), and we want to find the minimum value 
of this function using gradient descent. The general algorithm for gradient descent is: 
1. Initialize the values of the variables x and y. 
2. Calculate the gradient of the function at the current values of x and y. 
3. Update the values of x and y by subtracting the gradient multiplied by a small learning 
rate α. 
4. Repeat steps 2 and 3 until the values of x and y converge to the minimum value of the 
function. 

---

# Question 1

f(x) = x² - 4x + 3  
Start at x = 5, learning rate α = 0.1

## (a) Find minimum analytically

f'(x) = 2x - 4  
Set f'(x) = 0:  
2x - 4 = 0 → x = 2  
f(2) = (2)² - 4(2) + 3 = 4 - 8 + 3 = -1  

Second derivative: f''(x) = 2 > 0 → confirms minimum  
**Minimum point = (2, -1)**

## (b) Gradient descent by hand

x₀ = 5, α = 0.1

**Iteration 1:**  
f'(x₀) = 2(5) - 4 = 6  
x₁ = x₀ - α · f'(x₀) = 5 - 0.1 × 6 = 4.4  
f(4.4) = (4.4)² - 4(4.4) + 3 = 19.36 - 17.6 + 3 = 4.76  

**Iteration 2:**  
f'(x₁) = 2(4.4) - 4 = 4.8  
x₂ = 4.4 - 0.1 × 4.8 = 3.92  
f(3.92) = (3.92)² - 4(3.92) + 3 = 15.3664 - 15.68 + 3 = 2.6864  

**Iteration 3:**  
f'(x₂) = 2(3.92) - 4 = 3.84  
x₃ = 3.92 - 0.1 × 3.84 = 3.536  
f(3.536) = (3.536)² - 4(3.536) + 3 = 12.503 - 14.144 + 3 = 1.359  

Each step moves x closer to the minimum at x* = 2.

## (c) Code solution

```python
import torch as th
import torch.optim as optim
import torch.nn as nn
import numpy as np

def f(x):
    return x**2 - 4*x + 3

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        x = th.tensor(5.0)  # Start at x = 5
        self.x = nn.Parameter(x)
    
    def loss(self):
        return f(self.x)

model = MyModel()
lr = 0.1
loss_curve = []

optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    loss_curve.append(loss.item())
    optimizer.step()

print('x* =', model.x.data.item())
print('f(x*) =', f(model.x.data).item())
# Expected: x* ≈ 2, f(x*) ≈ -1
```

---

# Question 2

f(x,y) = x² + y²  
Start at x=1, y=1, learning rate α = 0.1

## (a) Find minimum analytically

∂f/∂x = 2x → set to 0 → x = 0  
∂f/∂y = 2y → set to 0 → y = 0  
f(0, 0) = 0  

The Hessian matrix is [[2, 0], [0, 2]] which is positive definite → confirms minimum  
**Minimum point = (0, 0), f(0,0) = 0**

## (b) Gradient descent by hand

(x₀, y₀) = (1, 1), α = 0.1

**Iteration 1:**  
∂f/∂x = 2(1) = 2  
∂f/∂y = 2(1) = 2  
x₁ = 1 - 0.1 × 2 = 0.8  
y₁ = 1 - 0.1 × 2 = 0.8  
f(0.8, 0.8) = 0.64 + 0.64 = 1.28  

**Iteration 2:**  
∂f/∂x = 2(0.8) = 1.6  
∂f/∂y = 2(0.8) = 1.6  
x₂ = 0.8 - 0.1 × 1.6 = 0.64  
y₂ = 0.8 - 0.1 × 1.6 = 0.64  
f(0.64, 0.64) = 0.4096 + 0.4096 = 0.8192  

**Iteration 3:**  
∂f/∂x = 2(0.64) = 1.28  
∂f/∂y = 2(0.64) = 1.28  
x₃ = 0.64 - 0.1 × 1.28 = 0.512  
y₃ = 0.64 - 0.1 × 1.28 = 0.512  
f(0.512, 0.512) = 0.2621 + 0.2621 = 0.5243  

Each step moves (x, y) closer to the minimum at (0, 0).

## (c) Code solution

```python
import torch as th
import torch.optim as optim
import torch.nn as nn

def f(x, y):
    return x**2 + y**2

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(th.tensor(1.0))  # Start at x = 1
        self.y = nn.Parameter(th.tensor(1.0))  # Start at y = 1
    
    def loss(self):
        return f(self.x, self.y)

model = MyModel()
lr = 0.1
loss_curve = []

optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    loss_curve.append(loss.item())
    optimizer.step()

print('x* =', model.x.data.item())
print('y* =', model.y.data.item())
print('f(x*, y*) =', f(model.x.data, model.y.data).item())
# Expected: x* ≈ 0, y* ≈ 0, f ≈ 0
```

---

# Question 3

f(x,y) = x² + 2xy + 3y² + 4x + 5y + 6  
Start at x=1, y=1, learning rate α = 0.01

## (a) Find minimum analytically

∂f/∂x = 2x + 2y + 4 → set to 0 → 2x + 2y + 4 = 0 → x + y = -2  ... (i)  
∂f/∂y = 2x + 6y + 5 → set to 0 → 2x + 6y + 5 = 0  ... (ii)  

From (i): x = -2 - y  
Substitute into (ii): 2(-2 - y) + 6y + 5 = 0  
-4 - 2y + 6y + 5 = 0  
4y + 1 = 0  
y = -1/4 = -0.25  
x = -2 - (-1/4) = -7/4 = -1.75  

f(-7/4, -1/4) = (-7/4)² + 2(-7/4)(-1/4) + 3(-1/4)² + 4(-7/4) + 5(-1/4) + 6  
= 49/16 + 14/16 + 3/16 - 112/16 - 20/16 + 96/16  
= (49 + 14 + 3 - 112 - 20 + 96) / 16  
= 30/16 = 15/8 = 1.875  

**Minimum point = (-1.75, -0.25), f = 1.875**

## (b) Gradient descent by hand

(x₀, y₀) = (1, 1), α = 0.01

**Iteration 1:**  
∂f/∂x = 2(1) + 2(1) + 4 = 8  
∂f/∂y = 2(1) + 6(1) + 5 = 13  
x₁ = 1 - 0.01 × 8 = 0.92  
y₁ = 1 - 0.01 × 13 = 0.87  
f(0.92, 0.87) = (0.92)² + 2(0.92)(0.87) + 3(0.87)² + 4(0.92) + 5(0.87) + 6  
= 0.8464 + 1.6008 + 2.2707 + 3.68 + 4.35 + 6 = 18.7479  

**Iteration 2:**  
∂f/∂x = 2(0.92) + 2(0.87) + 4 = 1.84 + 1.74 + 4 = 7.58  
∂f/∂y = 2(0.92) + 6(0.87) + 5 = 1.84 + 5.22 + 5 = 12.06  
x₂ = 0.92 - 0.01 × 7.58 = 0.8442  
y₂ = 0.87 - 0.01 × 12.06 = 0.7494  

**Iteration 3:**  
∂f/∂x = 2(0.8442) + 2(0.7494) + 4 = 1.6884 + 1.4988 + 4 = 7.1872  
∂f/∂y = 2(0.8442) + 6(0.7494) + 5 = 1.6884 + 4.4964 + 5 = 11.1848  
x₃ = 0.8442 - 0.01 × 7.1872 = 0.7723  
y₃ = 0.7494 - 0.01 × 11.1848 = 0.6376  

Each step moves (x, y) closer to the minimum at (-1.75, -0.25).  
Note: α = 0.01 is small, so convergence is slow — more epochs are needed.

## (c) Code solution

```python
import torch as th
import torch.optim as optim
import torch.nn as nn

def f(x, y):
    return x**2 + 2*x*y + 3*y**2 + 4*x + 5*y + 6

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(th.tensor(1.0))  # Start at x = 1
        self.y = nn.Parameter(th.tensor(1.0))  # Start at y = 1
    
    def loss(self):
        return f(self.x, self.y)

model = MyModel()
lr = 0.01
loss_curve = []

optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 5000  # More epochs needed due to small learning rate
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    loss_curve.append(loss.item())
    optimizer.step()

print('x* =', model.x.data.item())
print('y* =', model.y.data.item())
print('f(x*, y*) =', f(model.x.data, model.y.data).item())
# Expected: x* ≈ -1.75, y* ≈ -0.25, f ≈ 1.875
```

---

# Question 4

f(x, y) = x² + 2y² - 2xy - 2x  
Start at (1, 1), learning rate α = 0.1

## (a) Find minimum analytically

∂f/∂x = 2x - 2y - 2 → set to 0 → 2x - 2y - 2 = 0 → x - y = 1  ... (i)  
∂f/∂y = 4y - 2x → set to 0 → 4y - 2x = 0 → x = 2y  ... (ii)  

Substitute (ii) into (i): 2y - y = 1 → y = 1  
x = 2(1) = 2  

f(2, 1) = (2)² + 2(1)² - 2(2)(1) - 2(2) = 4 + 2 - 4 - 4 = -2  

**Minimum point = (2, 1), f = -2**

## (b) Gradient descent by hand

(x₀, y₀) = (1, 1), α = 0.1

**Iteration 1:**  
∂f/∂x = 2(1) - 2(1) - 2 = -2  
∂f/∂y = 4(1) - 2(1) = 2  
x₁ = 1 - 0.1 × (-2) = 1.2  
y₁ = 1 - 0.1 × 2 = 0.8  
f(1.2, 0.8) = 1.44 + 1.28 - 1.92 - 2.4 = -1.6  

**Iteration 2:**  
∂f/∂x = 2(1.2) - 2(0.8) - 2 = 2.4 - 1.6 - 2 = -1.2  
∂f/∂y = 4(0.8) - 2(1.2) = 3.2 - 2.4 = 0.8  
x₂ = 1.2 - 0.1 × (-1.2) = 1.32  
y₂ = 0.8 - 0.1 × 0.8 = 0.72  
f(1.32, 0.72) = 1.7424 + 1.0368 - 1.9008 - 2.64 = -1.7616  

**Iteration 3:**  
∂f/∂x = 2(1.32) - 2(0.72) - 2 = 2.64 - 1.44 - 2 = -0.8  
∂f/∂y = 4(0.72) - 2(1.32) = 2.88 - 2.64 = 0.24  
x₃ = 1.32 - 0.1 × (-0.8) = 1.4  
y₃ = 0.72 - 0.1 × 0.24 = 0.696  
f(1.4, 0.696) = 1.96 + 0.9686 - 1.9488 - 2.8 = -1.8202  

Each step moves (x, y) closer to the minimum at (2, 1).

## (c) Code solution

```python
import torch as th
import torch.optim as optim
import torch.nn as nn

def f(x, y):
    return x**2 + 2*y**2 - 2*x*y - 2*x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(th.tensor(1.0))  # Start at x = 1
        self.y = nn.Parameter(th.tensor(1.0))  # Start at y = 1
    
    def loss(self):
        return f(self.x, self.y)

model = MyModel()
lr = 0.1
loss_curve = []

optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    loss_curve.append(loss.item())
    optimizer.step()

print('x* =', model.x.data.item())
print('y* =', model.y.data.item())
print('f(x*, y*) =', f(model.x.data, model.y.data).item())
# Expected: x* ≈ 2, y* ≈ 1, f ≈ -2
```