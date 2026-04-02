import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# arr = np.random.geometric(0.2, size=10000)
# with open('mle1.npy', 'wb') as fp:
#     np.save(fp, arr - 1)

# arr = np.random.poisson(5, size=10000)
# with open('mle2.npy', 'wb') as fp:
#     np.save(fp, arr)

def plot_loss(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)
    plt.show()

def loss_function(dataset, p, eps=0.001):
    p1 = dataset * th.log(1 - p)
    p2 = th.log(p + eps) # eps added to prevent underflow
    acc = th.mean(p1 + p2)
    return -acc


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        p_prime = np.abs(np.random.uniform(0, 1))
        self.p_prime = nn.Parameter(th.tensor(p_prime))
    
    def p(self):
        return th.sigmoid(self.p_prime)
    
    def loss(self, dataset):
        return loss_function(dataset, self.p())

fp = open('mle1.npy', 'rb')
dataset = np.load(fp)
fp.close()
dataset = th.tensor(dataset, dtype=th.float32)

model = Model()
lr = 0.1
optimiser = optim.SGD(model.parameters(), lr=lr)
num_iters = 200
loss_curve = []

for i in range(num_iters):
    optimiser.zero_grad()
    loss_value = model.loss(dataset)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimiser.step()

p = model.p().detach().numpy()

plot_loss(loss_curve)

fp = open('mle1.npy', 'rb')
dataset = np.load(fp)
fp.close()
dataset = th.tensor(dataset, dtype=th.float32)



histogram = plt.hist(dataset.numpy(), bins=int(np.max(dataset.numpy())))
arr = []
p = model.p().detach().numpy()
for i in range(41):
    prob = (1 - p) ** i * p
    arr.append(prob * len(dataset))
plt.plot(list(range(41)), arr)

def loss_function(dataset, p, eps=0.001):
    p1 = dataset * th.log(1 - p)
    p2 = th.log(p + eps) # eps added to prevent underflow
    acc = th.mean(p1 + p2)
    return -acc

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        p_prime = np.abs(np.random.uniform(0, 1))
        self.p_prime = nn.Parameter(th.tensor(p_prime))

    def p(self):
        return th.sigmoid(self.p_prime)

    def loss(self, dataset):
        return loss_function(dataset, self.p())

model = Model()
lr = 0.1
optimiser = optim.SGD(model.parameters(), lr=lr)
num_iters = 200
loss_curve = []

for i in range(num_iters):
    optimiser.zero_grad()
    loss_value = model.loss(dataset)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimiser.step()
# Recall that p must be a probability, so instead of learning p directly, we instead learn a paramater that when
# passed through sigmoid gives us p
p = model.p().detach().numpy()
print(p)
plot_loss(loss_curve)

# @title
def loss_function(dataset, lam, eps=0.001):
    p1 = dataset * th.log(lam + eps)
    p2 = -lam
    acc = th.mean(p1 + p2)
    return -acc


# Similarly to p, lam must be positive, so we force it through an absolute function

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        lam_prime = np.random.normal(25)
        self.lam_prime = nn.Parameter(th.tensor(lam_prime))

    def lam(self):
        return th.abs(self.lam_prime)

    def loss(self, dataset):
        return loss_function(dataset, self.lam())

fp = open('mle2.npy', 'rb')
dataset = np.load(fp)
fp.close()
dataset = th.tensor(dataset, dtype=th.float32)

model = Model()
lr = 0.1
optimiser = optim.SGD(model.parameters(), lr=lr)
num_iters = 1000
loss_curve = []

for i in range(num_iters):
    optimiser.zero_grad()
    loss_value = model.loss(dataset)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimiser.step()

lam = model.lam().detach().numpy()
print(lam)

plot_loss(loss_curve)

histogram = plt.hist(dataset.numpy(), bins=int(np.max(dataset.numpy())))
arr = []
for i in range(41):
    prob = (lam ** i) * np.exp(-lam)
    prob = prob / np.math.factorial(i)
    arr.append(prob * len(dataset))
plt.plot(list(range(41)), arr)

