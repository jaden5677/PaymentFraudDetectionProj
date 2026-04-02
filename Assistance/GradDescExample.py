
import torch as th
import torch.optim as optim
import torch.nn as nn
import numpy as np

#Helper function to plot the loss curve
def plot_loss_curve(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)


# set up model with parameter and loss function
def f(x):
    return 2 * x * x + 3 * x - 4
   
class MyModel(nn.Module): #nn.Module is a PyTorch's building block for Neural Nxtowrks

    def __init__(self):
        super().__init__()
        x = th.tensor(np.random.random()) # generate random point
        self.x = nn.Parameter(x) #Here x is set up as an trainable parameter
    
    def loss(self):
        return f(self.x)

model = MyModel() # model instance
lr = 0.1 # our learning rate
loss_curve = []

#Link the parameters to be trained to the optimizer. The optimizer will decide #on how to use the graiden information to update the trainiable parameters

optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad() # zero out gradients to prevent gradients from carrying over into other iterations

    loss = model.loss() # compute loss
    loss.backward() # compute gradient with respect to loss function
    loss_curve.append(loss.item())

    optimizer.step() # use gradient descent to adjust value of paramters in model

print('Our final x^* is ', model.x.data)
print(f(model.x.data))
print(f(-0.75))

plot_loss_curve(loss_curve)
plt.xlabel('$i$')
plt.ylabel('loss')
plt.title('Loss curve')
