import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#如果是单通道输入，那么送入到核选择模块的时候，channel应该是 1 + 3*5 =16 ..这里的话，可以说明我们只是想利用的unet的特征提取，
# 而不是为了最后输出，这里可以展开详细的说明和实验，因为目前大家都以vgg resnet系列作为淡单纯的特征提取网络，我们首次用到unet，，，

#如果是三通道png输入，channel应该是 3+3*5 = 18.

channel_for_input = 18
#out_c = 6
#out_c  需要注意的是，原始其他研究者使用的时候是定义的7，为什么是7呢？因为他送入核选择模块的时候其实是7张图，我们少一张，具体的看论文里。。
#middle = 4
class selective_kernel(nn.Module):
    def __init__(self, middle, out_c):
        super(selective_kernel, self).__init__()
        self.out_ch = out_c
        self.middle = middle

        self.affine1 = nn.Linear(out_c, middle)
        self.affine2 = nn.Linear(middle, out_c)


    def forward(self, sk_conv1, sk_conv2, sk_conv3):
        sum_u = sk_conv1 + sk_conv2 + sk_conv3
        squeeze = nn.functional.adaptive_avg_pool2d(sum_u, (1, 1))
        squeeze = squeeze.view(squeeze.size(0), -1)
        z = self.affine1(squeeze)
        z = F.relu(z)
        a1 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a2 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a3 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)

        before_softmax = torch.cat([a1, a2, a3], dim=1)
        after_softmax = F.softmax(before_softmax, dim=1)
        a1 = after_softmax[:, 0:self.out_ch, :, :]
        a1.reshape(-1, self.out_ch, 1, 1)

        a2 = after_softmax[:, self.out_ch:2*self.out_ch, :, :]
        a2.reshape(-1, self.out_ch, 1, 1)
        a3 = after_softmax[:, 2*self.out_ch:3*self.out_ch, :, :]
        a3.reshape(-1, self.out_ch, 1, 1)

        select_1 = sk_conv1 * a1
        select_2 = sk_conv2 * a2
        select_3 = sk_conv3 * a3

        return select_1 + select_2 + select_3


class Attention(nn.Module):
    def __init__(self, in_channels, n_class ,out_channels ):
        super().__init__()
        self.middle = 4
        self.out_c = 6
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, n_class//2)
        self.fc2 = nn.Linear(n_class//2, out_channels)
        self.act_fc = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.ksm = selective_kernel(self.middle,self.out_c)
        #self.layer_out = nn.Tanh()
        self.layer_out = nn.Conv2d(self.out_c, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.act1(x3)

        u = self.ksm(x1,x2,x3)

        #u = x1 + x2 + x3
       # weight = self.pool(u)
        #weight = self.fc1(weight.squeeze(-1).transpose(-2, -1))
        #weight = self.act_fc(weight)
        #weight = self.fc2(weight)
        #weight = self.sig(weight)
        #weight = weight.permute(0, 2, 1).unsqueeze(-1)
        #print(weight)
        #print(u.shape)
        #u = u * weight
        return self.layer_out(u)

class NSAF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.middle = 2
        self.out_c = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, bias=False, dilation=4)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels//2)
        self.fc2 = nn.Linear(out_channels//2, out_channels)
        self.act_fc = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.ksm = selective_kernel(self.middle,self.out_c)
    def forward(self, input1, input2):
        if input2.shape != input1.shape:
            input2 = F.interpolate(input2, input1.shape[-2:], mode='bilinear')
        x = torch.cat([input1, input2], dim=1)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.act1(x3)

        u = self.ksm(x1,x2,x3)
        #u = x1 + x2 + x3
        #weight = self.pool(u)
        #weight = self.fc1(weight.squeeze(-1).transpose(-2, -1))
        #weight = self.act_fc(weight)
        #weight = self.fc2(weight)
        #weight = self.sig(weight)
        #weight = weight.permute(0, 2, 1).unsqueeze(-1)
        #u = u * weight
        return u


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2.shape != x1.shape:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class PSUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, out_channels, 1)
        
        self.nsaf1 = NSAF(64+64, 64)
        self.nsaf2 = NSAF(64+128, 64)
        self.nsaf3 = NSAF(128+256, 128)

    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        f = self.layer5(e4)
        nsaf1 = self.nsaf1(e1, e2)
        nsaf2 = self.nsaf2(e2, e3)
        nsaf3 = self.nsaf3(e3, e4)
        e1 = e1 + nsaf1
        e2 = e2 + nsaf2
        e3 = e3 + nsaf3
        d4 = self.decode4(f, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2, e1)
        d0 = self.decode0(d1)
        out = self.conv_last(d0)
        out = F.interpolate(out, size=input.shape[2:], mode='bilinear')
        #print(out.shape)
        return out


class PSFUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_class=2):
        super().__init__()
        self.out_c = 6
        self.avg1 = nn.AvgPool2d(kernel_size=16)
        self.avg2 = nn.AvgPool2d(kernel_size=8)
        self.avg3 = nn.AvgPool2d(kernel_size=4)
        self.avg4 = nn.AvgPool2d(kernel_size=2)
        self.avg5 = nn.AvgPool2d(kernel_size=1)
        self.unet1 = PSUnet(in_channels, 3)
        self.unet2 = PSUnet(in_channels, 3)
        self.unet3 = PSUnet(in_channels, 3)
        self.unet4 = PSUnet(in_channels, 3)
        self.unet5 = PSUnet(in_channels, 3)
        self.up1 = nn.Upsample(scale_factor=16)
        self.up2 = nn.Upsample(scale_factor=8)
        self.up3 = nn.Upsample(scale_factor=4)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up5 = nn.Upsample(scale_factor=1)
        self.conv = nn.Conv2d(out_channels, out_channels, 1)
        self.act = nn.ReLU()
        self.attention = Attention(channel_for_input, n_class,self.out_c)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x1 = self.avg1(x)
        x1 = self.unet1(x1)
        x1 = self.up1(x1)
        x2 = self.avg2(x)
        x2 = self.unet2(x2)
        x2 = self.up2(x2)
        x3 = self.avg3(x)
        x3 = self.unet3(x3)
        x3 = self.up3(x3)
        x4 = self.avg4(x)
        x4 = self.unet4(x4)
        x4 = self.up4(x4)
        x5 = self.avg5(x)
        x5 = self.unet5(x5)
        x5 = self.up5(x5)
        #x = x + x1 + x2 + x3 + x4 + x5
        x = torch.cat([x,x1,x2,x3,x4,x5],1)
        #x = self.conv(x) # x.shape = 16channel...
        #print(x.shape)
        x = self.act(x)
        x = self.attention(x)

        return self.sig(x)
        
# if __name__ == '__main__':
#     x = torch.randn(1, 1, 624, 624).to('cuda')
#     model = PSFUNet(in_channels=1, out_channels=1, n_class=2).to('cuda')
#     print(model(x).shape)
