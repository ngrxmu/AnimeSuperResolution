'''
network
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


def make_layer(block, n_layers):
    '''
    Make Layer
    '''
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class PA(nn.Module):
    '''
    Pixel Attention Block
    '''
    def __init__(self, channel):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = torch.mul(x, attention)
        return out

class AttentionBranch(nn.Module):
    '''
    Attention Branch
    '''
    def __init__(self, channel, kernel_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.k2 = nn.Conv2d(channel, channel, 1)
        self.k3 = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.k1(x)
        attention = self.lrelu(attention)
        attention = self.k2(attention)
        attention = self.sigmoid(attention)
        y = self.k3(x)
        out = torch.mul(y, attention)
        out = self.k4(out)
        return out

class AAB(nn.Module):
    '''
    Attention in Attention Block
    '''
    def __init__(self, channel, radio=4, t=30):
        super(AAB, self).__init__()
        self.t = t
        self.conv_first = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        '''
        Attention Dropout Module
        '''
        self.ADM = nn.Sequential(
            nn.Linear(channel, channel // radio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // radio, 2, bias=False),
        )
        self.attention = AttentionBranch(channel)
        self.non_attention = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        bs, c, _, _ = x.shape
        residual = x
        y_in = self.conv_first(x)
        y_in = self.lrelu(y_in)
        # Attention Dropout
        y_attention_dropout = self.avg_pool(y_in).view(bs, c)
        y_attention_dropout = self.ADM(y_attention_dropout)
        y_attention_dropout = F.softmax(y_attention_dropout / self.t, dim=1)
        # AB and NAB
        y_AB = self.attention(y_in)
        y_NAB = self.non_attention(y_in)
        y = y_AB * y_attention_dropout[:, 0].view(bs, 1, 1, 1) + y_NAB * y_attention_dropout[:, 1].view(bs, 1, 1, 1)
        y = self.lrelu(y)
        # Add Residual
        out = self.conv_last(y)
        out += residual
        return out

class AAN(nn.Module):
    '''
    Attention in Attention Network
    '''
    def __init__(self, scale):
        super(AAN, self).__init__()
        channel = 40
        reconstruction_channel = 24
        number_of_AAB = 16
        self.scale = scale
        self.conv_first = nn.Conv2d(3, channel, 3, 1, 1, bias=True)
        # AAB Main Trunk
        AAB_function = functools.partial(AAB, channel=channel)
        self.AAB_trunk = make_layer(AAB_function, number_of_AAB)
        self.trunk_conv = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        # Reconstruction Module
        self.upconv1 = nn.Conv2d(channel, reconstruction_channel, 3, 1, 1, bias=True)
        self.att1 = PA(reconstruction_channel)
        self.HRconv1 = nn.Conv2d(reconstruction_channel, reconstruction_channel, 3, 1, 1, bias=True)
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(reconstruction_channel, reconstruction_channel, 3, 1, 1, bias=True)
            self.att2 = PA(reconstruction_channel)
            self.HRconv2 = nn.Conv2d(reconstruction_channel, reconstruction_channel, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(reconstruction_channel, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feature = self.conv_first(x)
        main_trunk_out = self.trunk_conv(self.AAB_trunk(feature))
        y = feature + main_trunk_out
        if self.scale == 2 or self.scale == 3:
            y = self.upconv1(F.interpolate(y, scale_factor=self.scale, mode='nearest'))
            y = self.lrelu(self.att1(y))
            y = self.lrelu(self.HRconv1(y))
        elif self.scale == 4:
            y = self.upconv1(F.interpolate(y, scale_factor=2, mode='nearest'))
            y = self.lrelu(self.att1(y))
            y = self.lrelu(self.HRconv1(y))
            y = self.upconv2(F.interpolate(y, scale_factor=2, mode='nearest'))
            y = self.lrelu(self.att2(y))
            y = self.lrelu(self.HRconv2(y))
        out = self.conv_last(y)
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out

class L1_Charbonnier_loss(nn.Module):
    '''
    Charbonnier Loss
    '''
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

if __name__=='__main__':
    model = AAN(4)
    print(model)
    input = torch.randn(1, 3, 1024, 1024)
    out = model(input)
    print(out.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
