# Chaotic Spiking Back-propagation (CSBP)
CSBP introduces chaotic loss to generaten brain-like chaotic dynamics, which can be used as a plug-in unit for SNN direct training. An exmaples is shown below.
## Example1 Single-neuron model
The experiment of a single spiking neuron in the paper is shown in the following.
### 1. import packages for deep learning and ploting
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
```
### 2. define the surrogate function and single-neuron network
```python
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None
# define the single-neuron network with the surrogate gradient
act_fun = ZIF.apply
def mem_update(ops, x, mem, spike):
    mem = mem * 0.2 * (1. - spike) + ops(x)
    spike = act_fun(mem-thresh, 1) # act_fun : approximation firing function
    return mem, spike
class SNN(nn.Module):
    def __init__(self, w=0.5, b=0.5, time_window=20):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        nn.init.constant_(self.layer.weight, w)
        nn.init.constant_(self.layer.bias, b)
        self.time_window = time_window
    def forward(self, x):
        mem = spike = sum_spike = torch.zeros([1, 1], device=device)
        inp_spikes = torch.zeros((x.shape[0], self.time_window), device=device)
        for step in range(self.time_window):
            mem, spike = mem_update(self.layer, x, mem, spike)
            sum_spike += spike
        out = sum_spike / time_window
        h = self.layer(x)
        return out, h, mem
```
### 3. Training without CSBP 
```python
# setting hyper-parameters
thresh = 1 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.5 # decay constants
time_window = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
set_seed(10)
loss_fun = nn.MSELoss()

# training
inp = torch.FloatTensor([[1]]).to(device) # input sample
tgt = torch.FloatTensor([[0]]).to(device) # target of input

net = SNN()
net.to(device)
loss_list = []
w_list = []
b_list = []

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
for epoch in range(4000):
    net.zero_grad()
    optimizer.zero_grad()
    out = net(inp)
    loss = loss_fun(out[0], tgt)
    loss_list.append(loss.item())
    w_list.append(net.layer.weight.item())
    b_list.append(net.layer.bias.item())
    loss.backward()
    optimizer.step()

# ploting
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(range(4000), w_list, s=3)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('epoch', fontproperties="Calibri", fontsize=20)
plt.ylabel('$w$', fontproperties="Calibri", fontsize=20)
xmajorLocator = MultipleLocator(1000)
ax.xaxis.set_major_locator(xmajorLocator)
plt.grid()
plt.show()
```
![image](images/example1.png)
### 4. Training with CSBP 
```python
def chaos_loss_fun(out, z, I0=0.65):
    return -z * (I0 * torch.log(out) + (1 - I0) * torch.log(1 - out))
# training
net = SNN()
net.to(device)
w_list = []
b_list = []
bp_loss_list = []
csbp_loss_list = []

z = 10
beta = 0.9995
optimizer = torch.optim.SGD(net.parameters(), lr=1)
for epoch in range(4000):
    net.zero_grad()
    optimizer.zero_grad()
    out = net(inp)
    bp_loss = loss_fun(out[0], tgt)
    h = torch.sigmoid(out[1])
    chaos_loss = chaos_loss_fun(h, z)
    loss = bp_loss + chaos_loss
    bp_loss_list.append(bp_loss.item())
    csbp_loss_list.append(loss.item())
    w_list.append(net.layer.weight.item())
    b_list.append(net.layer.bias.item())
    loss.backward()
    optimizer.step()
    z *= beta
# ploting
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(range(4000), w_list, s=3)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('epoch', fontproperties="Calibri", fontsize=20)
plt.ylabel('$w$', fontproperties="Calibri", fontsize=20)
xmajorLocator = MultipleLocator(1000)
ax.xaxis.set_major_locator(xmajorLocator)
plt.savefig('example2.png')
plt.grid()
plt.show() 
```
![image](images/example2.png)
## Example2 CSBP for multi-layer spiking neuron networks
According to the defination of the chaotic loss in the paper, the ouputs or the membrane potentials of certain layers are necessary. Here we give an example for a common use. You will find CSBP can be introduced into an arbitrary layer in an arbitrary spiking neural network for direct training, including convolutional layers and fully-connected layers, so this work can be easily followed.  
Here we use a spiking neural network with one convolutional layer and one fully-connected layer to simpilify. A random $16\times16$ image of 3 channels is input to the network for example. The results of the image classification will be output. Total category count is 10.
```python
# This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)
    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)
def preprocess_train_sample(T, x: torch.Tensor):
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
    
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None
    
class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
    def forward(self, x):
        mem = 0
        sum_spike = []
        for t in range(x.shape[0]):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            sum_spike.append(spike)
        return torch.stack(sum_spike, dim=1)
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv = SeqToANNContainer(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1))
        self.fc = SeqToANNContainer(nn.Linear(2048, 10))
        self.act = LIFSpike()
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(0)
image = torch.rand(1, 3, 16, 16)
x = preprocess_train_sample(8, image)
target = torch.tensor([1,0,0,0,0,0,0,0,0,0]).reshape(1, 10).float()
net = SNN()
out = net(x)
```
For a image classification task, a crossentropy loss is usually used. BP loss can be obtained.
```python
bp_loss = loss_fun(out, target)
```
The above is a general procedure of BP. To introduce CSBP into the convoluntional layer, you just need to make the following modification.
```python
class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        sum_spike = []
        for t in range(x.shape[0]):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            sum_spike.append(spike)
        return torch.stack(sum_spike, dim=0), mem
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv = SeqToANNContainer(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1))
        self.fc = SeqToANNContainer(nn.Linear(2048, 10))
        self.act = LIFSpike()
    def forward(self, x):
        x = self.conv(x)
        mid_output = x.clone()
        x, mid_mem = self.act(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(0), mid_output, mid_mem        
image = torch.rand(1, 3, 16, 16)
x = preprocess_train_sample(8, image)
target = torch.tensor([1,0,0,0,0,0,0,0,0,0]).reshape(1, 10).float()
net = SNN()
out = net(x)
bp_loss = loss_fun(out[0], target)
```
Define the chaotic loss function:
```python
def chaos_loss_fun(h, z, I0=0.65):
    out = torch.sigmoid(h/10) 
    log1, log2 = torch.log(out), torch.log(1 - out)
    return -z * (I0 * log1 + (1 - I0) * log2)
```
Chaotic loss can be caculated by:
```python
chaotic_loss = chaos_loss_fun(out[1], 10).sum()
```
or another implement with membrane potential：
```python
chaotic_loss = chaos_loss_fun(out[2], 10).sum()
```
