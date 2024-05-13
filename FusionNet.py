# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

class Channel_Attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_Attention,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Linear(channel, channel//ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.line2 = nn.Linear(channel//ratio, channel, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        b, c, _, _ = x.size()
        x1 = self.avgpool(x).view(b,c)
        x2 = self.line1(x1)
        x3 = self.relu(x2)
        x4 = self.line2(x3)
        x5 = self.sig(x4)
        out = x5.view(b,c,1,1)
        return x * out.expand_as(x)

##这里引入了自注意力，增强在交叉时网络注意力在全局上的把控，增强网络对潜在交通信息内在联系的感知能力
# intro the self-attention to enhance the information exchange in cross-path, while improving the ability of the learning potential traffic system connection.
class Self_Attention(nn.Module):
    def __init__(self, channel):
        super(Self_Attention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.sig(x1)
        x1 = torch.transpose(x1, dim0=3, dim1=2)
        x2 = self.conv2(x)
        x2 = self.sig(x2)
        # print('\n', x1.size(), x2.size())
        x3 = torch.matmul(x1,x2)
        x4 = torch.matmul(x,x3)
        out = self.conv3(x4)
        return self.sig(out)


class Rubik_Cube_Attention(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(Rubik_Cube_Attention, self).__init__()
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.line2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.space = nn.Conv2d(1, 1, kernel_size, padding = 3, bias = False)
        self.satt = Self_Attention(channel)
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self,x):
        ##训练通道权重
        ##channel attention
        x_ca = self.channel_avg_pool(x)
        x_ca = self.relu(self.line1(x_ca))
        x_ca = self.relu(self.line2(x_ca))
        x_ca = self.sigmoid(x_ca)
        ##同时训练空间权重
        ##spatial attention
        x_sa = torch.mean(x, dim=1, keepdim=True)
        x_sa = self.space(x_sa)
        x_sa = self.sigmoid(x_sa)
        ##直接相乘
        ##multiple
        x_attention = x*x_sa*x_ca
        ##训练自注意力机制
        ##self-attention
        x_self = self.satt(x)
        ##相加
        ##addition
        out = 0.1*x_self+x_attention
        out = self.sigmoid(self.conv(out))
        return out

## Cross-path during feature extraction
class Cross_Attention(nn.Module):
    def __init__(self, channels):
        super(Cross_Attention, self).__init__()
        self.rubik = Rubik_Cube_Attention(channels)
        self.se = Channel_Attention(channels)

    def forward(self,x1,x2):
        x1_2 = x1 - x2
        x2_1 = x2 - x1
        x1_attention = self.rubik(x2_1)
        x2_attention = self.rubik(x1_2)
        out1 = x1 + x1_attention
        out2 = x2 + x2_attention
        return out1, out2

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class DenSoAM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DenSoAM, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
        self.cattention = Channel_Attention(out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x1=self.cattention(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        x2=self.cattention(x2)

        return F.leaky_relu(x1+x2,negative_slope=0.1)

class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        ch = [16,32,48,64]
        output=1
        self.conv_vis=ConvLeakyRelu2d(1,ch[0])
        self.conv_inf=ConvLeakyRelu2d(1,ch[0])
        self.dsam1_vis = DenSoAM(ch[0], ch[1])
        self.dsam1_inf = DenSoAM(ch[0], ch[1])
        self.dsam2_vis = DenSoAM(ch[1], ch[2])
        self.dsam2_inf = DenSoAM(ch[1], ch[2])
        self.cro_att1 = Cross_Attention(ch[1])
        self.cro_att2 = Cross_Attention(ch[2])
        self.decode4 = ConvBnLeakyRelu2d(ch[2]+ch[2], ch[1]+ch[1])
        self.decode3 = ConvBnLeakyRelu2d(ch[1]+ch[1], ch[0]+ch[0])
        self.decode2 = ConvBnLeakyRelu2d(ch[0]+ch[0], ch[0])
        self.decode1 = ConvBnTanh2d(ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir

        # encode
        x_vis_p=self.conv_vis(x_vis_origin)
        x_vis_p1=self.dsam1_vis(x_vis_p)
        x_inf_p=self.conv_inf(x_inf_origin)
        x_inf_p1=self.dsam1_inf(x_inf_p)
        x_vis_a1, x_inf_a1 = self.cro_att1(x_vis_p1,x_inf_p1)
        x_vis_p2=self.dsam2_vis(x_vis_a1)
        x_inf_p2=self.dsam2_inf(x_inf_a1)
        x_vis_a2, x_inf_a2 = self.cro_att2(x_vis_p2,x_inf_p2)

        # decoder
        x=self.decode4(torch.cat((x_vis_a2,x_inf_a2),dim=1))
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
