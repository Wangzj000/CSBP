import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SG(torch.autograd.Function):
    # Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIF(nn.Module):
    def __init__(self, v_th=1.0, tau=0.25, gamma=1.0):
        super(LIF, self).__init__()
        self.heaviside = SG.apply
        self.v_th = v_th
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        mem_v = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        return torch.stack(mem_v, dim=1)

class LIF_mem(nn.Module):
    def __init__(self, v_th=1.0, tau=0.25, gamma=1.0):
        super(LIF_mem, self).__init__()
        self.heaviside = SG.apply
        self.v_th = v_th
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        mem_v = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        return torch.stack(mem_v, dim=1), mem

class SeqToANNContainer(nn.Module):
    # Altered form SpikingJelly
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


class TEBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device)) # 4应为T

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class TEBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)

    def forward(self, input):
        y = self.fwd(input)
        y = self.bn(y)
        return y


class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10):
        super().__init__()
        self.voting_size = voting_size

    def forward(self, x: torch.Tensor):
        x.unsqueeze_(1)  # [N, C] -> [N, 1, C]
        y = F.avg_pool1d(x, self.voting_size, self.voting_size)
        y.squeeze_(1)
        return y


def input_expand(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, tau=0.25):
        super(BasicBlock, self).__init__()
        self.tau = tau
        self.conv1 = TEBNLayer(in_ch, out_ch, 3, stride, 1)
        self.conv2 = TEBNLayer(out_ch, out_ch, 3, 1, 1)

        self.sn2 = LIF(tau=self.tau)
        self.stride = stride
        self.downsample = downsample

        self.bn = TEBN(out_ch)

    def forward(self, x):
        right = x
        y = self.conv1(x)
        y = self.sn1(y)
        y = self.conv2(y)
        if self.downsample is not None:
            right = self.downsample(x)
        else:
            right = self.bn(x)
        y += right
        y = self.sn2(y)

        return y


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 3, 2], tau=0.25,):
        super(ResNet, self).__init__()
        self.tau = tau
        self.T = 6
        self.in_ch = 128
        self.voting = VotingLayer(10)
        self.conv1 = TEBNLayer(3, self.in_ch, 3, 1, 1)

        self.sn1 = LIF(tau=self.tau)
        self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 512, layers[2], stride=2)
        #         self.layer4 = self.make_layer(block, 512, layers[3], stride=2,)    #

        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * 4 * 4, 256))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25),nn.Linear(256, 100))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, in_ch, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != in_ch * block.expansion:
            downsample = TEBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
        layers = []
        layers.append(block(self.in_ch, in_ch, stride, downsample, method=self.method, tau=self.tau))
        self.in_ch = in_ch * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_ch, in_ch, method=self.method, tau=self.tau))
        return nn.Sequential(*layers)

    def forward_imp(self, input):
        x = input_expand(input, self.T)

        x = self.conv1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #         x = self.layer4(x)
        x = self.pool(x)

        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x

    def forward(self, input):
        return self.forward_imp(input)


class VGG9(nn.Module):
    def __init__(self, tau=0.25):
        super(VGG9, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features = nn.Sequential(
            TEBNLayer(3, 64, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(64, 64, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(64, 128, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIF(tau=self.tau),
            pool,

        )
        self.T = 4
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(256 * 4 * 4, 1024))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(1024, 100), self.voting)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = input_expand(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x
    
class CNN7(nn.Module):
    # hyper: T=10 32batch lr0.1 200epoch cosanneal200
    def __init__(self, tau=0.25):
        super(CNN7, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(64, 128, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
        )
        self.T = 10
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(256 * 3 * 3, 1024))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(1024, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x

class CNN7_chaos(nn.Module):
    # hyper: T=10 32batch lr0.1 200epoch cosanneal200
    def __init__(self, tau=0.25):
        super(CNN7_chaos, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features1 = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1),
            LIF_mem(tau=self.tau),
        )
        self.features2 = nn.Sequential(
            pool,
            TEBNLayer(64, 128, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
        )
        self.T = 10
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(256 * 3 * 3, 1024))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(1024, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x, h1 = self.features1(input)
        x = self.features2(x)
        x = torch.flatten(x, 2)
        h2 = self.fc1(x)
        x = self.fc2(h2)
        return x, h1, h2

class VGGSNN(nn.Module):
    def __init__(self, tau = 0.25):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1),
            LIF(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIF(tau=self.tau),
            pool,

        )
        self.T = 10
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * 3 * 3, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x
    
class CNN6(nn.Module):
    # hyper: T=10 32batch lr0.1 200epoch cosanneal200
    def __init__(self, tau=0.25):
        super(CNN6, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features = nn.Sequential(
            TEBNLayer(2, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
        )
        self.T = 10
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(128 * 3 * 3, 512))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x
    
class CNN6_chaos(nn.Module):
    # hyper: T=10 32batch lr0.1 200epoch cosanneal200
    def __init__(self, tau=0.25):
        super(CNN6_chaos, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.voting = VotingLayer(10)
        self.features1 = nn.Sequential(
            TEBNLayer(2, 128, 3, 1, 1),
            LIF_mem(tau=self.tau),
        )
        self.features2 = nn.Sequential(
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
            TEBNLayer(128, 128, 3, 1, 1),
            LIF(tau=self.tau),
            pool,
        )
        self.T = 10
        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(128 * 3 * 3, 512))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x, h1 = self.features1(input)
        x = self.features2(x)
        x = torch.flatten(x, 2)
        h2 = self.fc1(x)
        x = self.fc2(h2)
        return x, h1, h2