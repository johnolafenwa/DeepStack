import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd,_ConvTransposeMixin,_single,_pair,_triple


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):

        return inputs * torch.sigmoid(inputs)

class ConvNd(_ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,out_padding):
       
        super(ConvNd,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,False,out_padding,groups,bias)


class DepthwiseConv2d(ConvNd):
    def __init__(self,in_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,multiplier=1):
        
        super(DepthwiseConv2d,self).__init__(in_channels,in_channels*multiplier,_pair(kernel_size),_pair(stride),_pair(padding),_pair(dilation),in_channels,bias,_pair(0))


    def forward(self,input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class _GlobalPoolNd(nn.Module):
    def __init__(self,flatten=True):
        """

        :param flatten:
        """
        super(_GlobalPoolNd,self).__init__()
        self.flatten = flatten

    def pool(self,input):
        """

        :param input:
        :return:
        """
        raise NotImplementedError()

    def forward(self,input):
        """

        :param input:
        :return:
        """
        input = self.pool(input)
        size_0 = input.size(1)
        return input.view(-1,size_0) if self.flatten else input

class GlobalAvgPool2d(_GlobalPoolNd):
    def __init__(self, flatten=True):
        """

        :param flatten:
        """
        super(GlobalAvgPool2d,self).__init__(flatten)

    def pool(self, input):
        return F.adaptive_avg_pool2d(input,1)

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UpSampleInterpolate(nn.Module):
    def __init__(self,scale_factor):
        super(UpSampleInterpolate,self).__init__()

        self.scale_factor = scale_factor

    def forward(self,x):

        return F.interpolate(x,scale_factor=self.scale_factor,mode="nearest")

