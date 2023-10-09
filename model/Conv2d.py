import torch
import torch.nn as nn
from  torchvision.ops import deform_conv2d 


class Conv2d(nn.Module):
    '''ResNet18Deform Residual Block
    Args:
        in_channels(int): input channels
        out_channels(int): output channels
        stride(int): stride    
    '''
    def __init__(self, in_channels: int, conv_ch: list, num_classes: int, stride: int=1, deform: bool=True, modulity: bool=True):
        super().__init__()
        self.stride = stride
        self.deform = deform
        self.modulity = modulity
        self.conv1 = nn.Conv2d(in_channels, conv_ch[0], kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.offset1 = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.modulity1 = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_ch[0])
        self.conv2 = nn.Conv2d(conv_ch[0], conv_ch[1], kernel_size=3, padding=1, bias=False)
        self.offset2 = nn.Conv2d(conv_ch[0], 2 * 3 * 3, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.modulity2 = nn.Conv2d(conv_ch[0], 3 * 3, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_ch[1])
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_ch[1], num_classes)

    def forward(self, x: torch.Tensor):
        if self.deform:
            offset = self.offset1(x)
            if self.modulity:
                modulity = torch.sigmoid(self.modulity1(x))
                out = deform_conv2d(x, offset, weight=self.conv1.weight, mask=modulity, bias=self.conv1.bias, padding=1, stride=self.stride)
            else:
                out = deform_conv2d(x, offset, weight=self.conv1.weight, bias=self.conv1.bias, padding=1, stride=self.stride)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.deform:
            offset = self.offset2(out)
            if self.modulity:
                modulity = torch.sigmoid(self.modulity2(out))
                out = deform_conv2d(out, offset, weight=self.conv2.weight, mask=modulity, bias=self.conv2.bias, padding=1, stride=self.stride)
            else:
                out = deform_conv2d(out, offset, weight=self.conv2.weight, bias=self.conv2.bias, padding=1, stride=self.stride)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.gap(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        
        return out
