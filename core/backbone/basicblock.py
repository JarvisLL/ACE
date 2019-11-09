import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self,in_channels,out_channels,atrous_rate):
        super(_ASPPConv,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=atrous_rate,dilation=atrous_rate,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
            )
    def forward(self,x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(_AsppPooling,self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
            )
    def forward(self,x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool,size,mode="bilinear",align_corners=True)
        return out

class _ASPP(nn.Module):
    """docstring for _ASPP"""
    def __init__(self,in_channels,atrous_rate,**kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
            )
        rate1,rate2,rate3 = tuple(atrous_rate)
        self.b1 = _ASPPConv(in_channels,out_channels,rate1)
        self.b2 = _ASPPConv(in_channels,out_channels,rate2)
        self.b3 = _ASPPConv(in_channels,out_channels,rate3)
        self.b4 = _AsppPooling(in_channels,out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
            )
    def forward(self,x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1,feat2,feat3,feat4,feat5),dim=1)
        x = self.project(x)
        return x
        