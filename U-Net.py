import torch.nn as nn
import torch

class Double_conv(nn.Module):#将两次卷积进行一个封装
    def __init__(self, input, output):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.COnv2d(input, output, 3, 1, 0),#卷积核大小为3，步长为1，padding为0不加边
            nn.BatchNorm2d(output),#归一化处理
            nn.ReLU(inplace=True),#覆盖原来的结果

            nn.COnv2d(input, output, 3, 1, 0),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
    def forward(self, ret):
        ret = self.conv(ret)
        return ret

class Down(nn.Module):#下采样
    def __init__(self, input, output):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Double_conv(input, output), #两次卷积
            nn.MaxPool2d(2)#最大池化，池化核大小为2
            )

