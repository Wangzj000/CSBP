# Chaotic Spiking Back-propagation (CSBP)
CSBP introduces chaotic loss to generaten brain-like chaotic dynamics, which can be used as a plug-in unit for SNN direct training. An exmaples is shown below.
## Example
The experiment of a single spiking neuron in the paper is shown in the following.
### 1. import packages for deep learning and ploting
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    def __init__(self, w=0.7, b=0.7, time_window=20):
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
        hid = self.layer(x)
        return out, hid, mem
```
### 3. Training without CSBP 
### 4. Training with CSBP 
