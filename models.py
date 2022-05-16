import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(GridAttentionBlock, self).__init__()
        
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                               kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=gating_channels, out_channels=inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=inter_channels, out_channels=1,
                             kernel_size=1, stride=1, padding=0, bias=False)
        self.W = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                         kernel_size=1, stride=1, padding=0, bias=True),
                               nn.BatchNorm2d(in_channels))
    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], 
                              mode="bilinear", align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:],
                                   mode="bilinear", align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

class SpatialAttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(SpatialAttentionGate, self).__init__()
        
        self.conv_e = nn.Conv2d(in_channels=in_channels, out_channels=1,
                                kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv2d(in_channels=gating_channels, out_channels=1,
                                kernel_size=1, stride=1, padding=0)
        self.conv_long_range = nn.Conv2d(in_channels=3, out_channels=1,
                                         kernel_size=7, stride=1, padding=3)
    def forward(self, Fe, Fd):
        Fe_size = Fe.size()
        Fd = F.interpolate(Fd, size=Fe_size[2:], mode="bilinear", align_corners=False)
        
        Fe_avg = torch.mean(Fe, dim=1, keepdims=True)
        Fe_max = torch.max(Fe, dim=1, keepdims=True)[0]
        Fe_conv = self.conv_e(Fe)
        Me = self.conv_long_range(torch.cat([Fe_avg, Fe_max, Fe_conv], dim=1))
        
        Fd_avg = torch.mean(Fd, dim=1, keepdims=True)
        Fd_max = torch.max(Fd, dim=1, keepdims=True)[0]
        Fd_conv = self.conv_d(Fd)
        Md = self.conv_long_range(torch.cat([Fd_avg, Fd_max, Fd_conv], dim=1))
        
        return torch.sigmoid(Me + Md)


class ChannelAttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(ChannelAttentionGate, self).__init__()
        
        N = int(in_channels / 16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv_N_e = nn.Conv2d(in_channels=in_channels, out_channels=N, 
                                  kernel_size=1, stride=1, padding=0)
        self.conv_N_d = nn.Conv2d(in_channels=gating_channels, out_channels=N,
                                  kernel_size=1, stride=1, padding=0)
        self.conv_C1 = nn.Conv2d(in_channels=N, out_channels=in_channels,
                                 kernel_size=1, stride=1, padding=0)
    def forward(self, Fe, Fd):
        Fe_size = Fe.size()
        Fd = F.interpolate(Fd, size=Fe_size[2:], mode="bilinear", align_corners=False)
        
        Fe_avg = self.avgpool(Fe)
        Fe_max = self.maxpool(Fe)
        Fe_avg = self.conv_N_e(Fe_avg)
        Fe_max = self.conv_N_e(Fe_max)
        Me = Fe_avg + Fe_max
        
        Fd_avg = self.avgpool(Fd)
        Fd_max = self.maxpool(Fd)
        Fd_avg = self.conv_N_d(Fd_avg)
        Fd_max = self.conv_N_d(Fd_max)
        Md = Fd_avg + Fd_max
        
        return torch.sigmoid(self.conv_C1(Me + Md))

class SpatialChannelAttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(SpatialChannelAttentionGate, self).__init__()
        
        self.sAG = SpatialAttentionGate(in_channels, gating_channels)
        self.cAG = ChannelAttentionGate(in_channels, gating_channels)
        self.W = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                         kernel_size=1, stride=1, padding=0, bias=True),
                               nn.BatchNorm2d(in_channels))
    def forward(self, x, g):
        Ms = self.sAG(x, g)
        Mc = self.cAG(x, g)
        x = Ms.expand_as(x) * x
        x = Mc.expand_as(x) * x
        return self.W(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                             kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.conv = DownBlock(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=4, stride=2, padding=1)
    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, inputs2], dim=1))

class DeepSupervision(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DeepSupervision, self).__init__()
        
        self.dsv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode="bilinear", 
                                             align_corners=False))
    def forward(self, inputs):
        return self.dsv(inputs)
    
class Attention_UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, base_features=32):
        super(Attention_UNet, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        
        # downsampling
        self.conv1 = DownBlock(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DownBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DownBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DownBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = DownBlock(filters[3], filters[4])
        self.gating = nn.Sequential(nn.Conv2d(filters[4], filters[3], 
                                              kernel_size=1),
                                    nn.BatchNorm2d(filters[3]),
                                    nn.ReLU(inplace=True))
        
        # attention blocks
        self.attentionblock2 = GridAttentionBlock(filters[1], filters[3], filters[1])
        self.attentionblock3 = GridAttentionBlock(filters[2], filters[3], filters[2])
        self.attentionblock4 = GridAttentionBlock(filters[3], filters[3], filters[3])
        
        # upsampling
        self.upblock4 = UpBlock(filters[4], filters[3])
        self.upblock3 = UpBlock(filters[3], filters[2])
        self.upblock2 = UpBlock(filters[2], filters[1])
        self.upblock1 = UpBlock(filters[1], filters[0])
        
        # deep supervision
        self.dsv4 = DeepSupervision(filters[3], num_classes, scale_factor=8)
        self.dsv3 = DeepSupervision(filters[2], num_classes, scale_factor=4)
        self.dsv2 = DeepSupervision(filters[1], num_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=num_classes, 
                              kernel_size=1, stride=1, padding=0)
        
        # final
        self.final = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        
    def forward(self, inputs):
        # Feature extracting
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        
        # Gating Signal Generation
        center = self.center(self.maxpool4(conv4))
        gating = self.gating(center)
        
        # Attention Mechanism
        g_conv4 = self.attentionblock4(conv4, gating)
        g_conv3 = self.attentionblock3(conv3, gating)
        g_conv2 = self.attentionblock2(conv2, gating)
        
        # Decoder
        up4 = self.upblock4(g_conv4, center)
        up3 = self.upblock3(g_conv3, up4)
        up2 = self.upblock2(g_conv2, up3)
        up1 = self.upblock1(conv1, up2)
        
        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        # Final
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))  
        return torch.sigmoid(final)

class MultiAttention_UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, base_features=32):
        super(MultiAttention_UNet, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        
        # downsampling
        self.conv1 = DownBlock(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DownBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DownBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DownBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = DownBlock(filters[3], filters[4])
        self.gating = nn.Sequential(nn.Conv2d(filters[4], filters[4], 
                                              kernel_size=1),
                                    nn.BatchNorm2d(filters[4]),
                                    nn.ReLU(inplace=True))
        
        # attention blocks
        self.attentionblock2 = GridAttentionBlock(filters[1], filters[2], filters[1])
        self.attentionblock3 = GridAttentionBlock(filters[2], filters[3], filters[2])
        self.attentionblock4 = GridAttentionBlock(filters[3], filters[4], filters[3])
        
        # upsampling
        self.upblock4 = UpBlock(filters[4], filters[3])
        self.upblock3 = UpBlock(filters[3], filters[2])
        self.upblock2 = UpBlock(filters[2], filters[1])
        self.upblock1 = UpBlock(filters[1], filters[0])
        
        # deep supervision
        self.dsv4 = DeepSupervision(filters[3], num_classes, scale_factor=8)
        self.dsv3 = DeepSupervision(filters[2], num_classes, scale_factor=4)
        self.dsv2 = DeepSupervision(filters[1], num_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=num_classes, 
                              kernel_size=1, stride=1, padding=0)
        
        # final
        self.final = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        
    def forward(self, inputs):
        # Feature extracting
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        
        # Gating Signal Generation
        center = self.center(self.maxpool4(conv4))
        gating = self.gating(center)
        
        # Attention Mechanism
        # Decoder
        g_conv4 = self.attentionblock4(conv4, gating)
        up4 = self.upblock4(g_conv4, center)
        g_conv3 = self.attentionblock3(conv3, up4)
        up3 = self.upblock3(g_conv3, up4)
        g_conv2 = self.attentionblock2(conv2, up3)
        up2 = self.upblock2(g_conv2, up3)
        up1 = self.upblock1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        # Final
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))  
        return torch.sigmoid(final)

class SpatialChannelAttention_UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, base_features=32):
        super(SpatialChannelAttention_UNet, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        
        # downsampling
        self.conv1 = DownBlock(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DownBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DownBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DownBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = DownBlock(filters[3], filters[4])
        self.gating = nn.Sequential(nn.Conv2d(filters[4], filters[4], 
                                              kernel_size=1),
                                    nn.BatchNorm2d(filters[4]),
                                    nn.ReLU(inplace=True))
        
        # attention blocks
        self.attentionblock2 = SpatialChannelAttentionGate(filters[1], filters[2])
        self.attentionblock3 = SpatialChannelAttentionGate(filters[2], filters[3])
        self.attentionblock4 = SpatialChannelAttentionGate(filters[3], filters[4])
        
        # upsampling
        self.upblock4 = UpBlock(filters[4], filters[3])
        self.upblock3 = UpBlock(filters[3], filters[2])
        self.upblock2 = UpBlock(filters[2], filters[1])
        self.upblock1 = UpBlock(filters[1], filters[0])
        
        # deep supervision
        self.dsv4 = DeepSupervision(filters[3], num_classes, scale_factor=8)
        self.dsv3 = DeepSupervision(filters[2], num_classes, scale_factor=4)
        self.dsv2 = DeepSupervision(filters[1], num_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=num_classes, 
                              kernel_size=1, stride=1, padding=0)
        
        # final
        self.final = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        
    def forward(self, inputs):
        # Feature extracting
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        
        # Gating Signal Generation
        center = self.center(self.maxpool4(conv4))
        gating = self.gating(center)
        
        # Attention Mechanism
        # Decoder
        g_conv4 = self.attentionblock4(conv4, gating)
        up4 = self.upblock4(g_conv4, center)
        g_conv3 = self.attentionblock3(conv3, up4)
        up3 = self.upblock3(g_conv3, up4)
        g_conv2 = self.attentionblock2(conv2, up3)
        up2 = self.upblock2(g_conv2, up3)
        up1 = self.upblock1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        # Final
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))  
        return torch.sigmoid(final)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, base_features=32):
        super(UNetPlusPlus, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        self.conv0_0 = self._block(in_channels, filters[0])
        self.conv1_0 = self._block(filters[0], filters[1])
        self.conv2_0 = self._block(filters[1], filters[2])
        self.conv3_0 = self._block(filters[2], filters[3])
        self.conv4_0 = self._block(filters[3], filters[4])
        
        self.conv0_1 = self._block(filters[0] + filters[1], filters[0])
        self.conv1_1 = self._block(filters[1] + filters[2], filters[1])
        self.conv2_1 = self._block(filters[2] + filters[3], filters[2])
        self.conv3_1 = self._block(filters[3] + filters[4], filters[3])
        
        self.conv0_2 = self._block(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = self._block(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = self._block(filters[2] * 2 + filters[3], filters[2])
        
        self.conv0_3 = self._block(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = self._block(filters[1] * 3 + filters[2], filters[1])
        
        self.conv0_4 = self._block(filters[0] * 4 + filters[1], filters[0])
        
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        
        
    def forward(self, inputs):
        X0_0 = self.conv0_0(inputs)
        X1_0 = self.conv1_0(self.pool(X0_0))
        X0_1 = self.conv0_1(torch.cat([X0_0, self.up(X1_0)], dim=1))
        
        X2_0 = self.conv2_0(self.pool(X1_0))
        X1_1 = self.conv1_1(torch.cat([X1_0, self.up(X2_0)], dim=1))
        X0_2 = self.conv0_2(torch.cat([X0_0, X0_1, self.up(X1_1)], dim=1))
        
        X3_0 = self.conv3_0(self.pool(X2_0))
        X2_1 = self.conv2_1(torch.cat([X2_0, self.up(X3_0)], dim=1))
        X1_2 = self.conv1_2(torch.cat([X1_0, X1_1, self.up(X2_1)], dim=1))
        X0_3 = self.conv0_3(torch.cat([X0_0, X0_1, X0_2, self.up(X1_2)], dim=1))
        
        X4_0 = self.conv4_0(self.pool(X3_0))
        X3_1 = self.conv3_1(torch.cat([X3_0, self.up(X4_0)], dim=1))
        X2_2 = self.conv2_2(torch.cat([X2_0, X2_1, self.up(X3_1)], dim=1))
        X1_3 = self.conv1_3(torch.cat([X1_0, X1_1, X1_2, self.up(X2_2)], dim=1))
        X0_4 = self.conv0_4(torch.cat([X0_0, X0_1, X0_2, X0_3, self.up(X1_3)], dim=1))
        
        return torch.sigmoid(self.final(X0_4))
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.SiLU(inplace=True),
                             
                             nn.Conv2d(out_channels, out_channels, 
                                       kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.SiLU(inplace=True))
    
class DepthWise_Seperable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(DepthWise_Seperable, self).__init__()
        # Depth Wise Convolution
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, 7, 
                                         stride, 3, groups=in_channels, bias=bias)
        # Pixel Wise Convolution
        self.pixel_wise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.pixel_wise_conv(x)
        return x
    

class LightDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightDownBlock, self).__init__()
        
        self.conv1 = nn.Sequential(DepthWise_Seperable(in_channels=in_channels, out_channels=out_channels, 
                                                      stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(DepthWise_Seperable(in_channels=out_channels, out_channels=out_channels, 
                                                      stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class LightUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightUpBlock, self).__init__()
        
        self.conv = LightDownBlock(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=4, stride=2, padding=1)
    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, inputs2], dim=1))


    
class LightMultiAttention_UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, base_features=32):
        super(LightMultiAttention_UNet, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        
        # downsampling
        self.conv1 = LightDownBlock(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = LightDownBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = LightDownBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = LightDownBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = LightDownBlock(filters[3], filters[4])
        self.gating = nn.Sequential(nn.Conv2d(filters[4], filters[4], 
                                              kernel_size=1),
                                    nn.BatchNorm2d(filters[4]),
                                    nn.ReLU(inplace=True))
        
        # attention blocks
        self.attentionblock2 = GridAttentionBlock(filters[1], filters[2], filters[1])
        self.attentionblock3 = GridAttentionBlock(filters[2], filters[3], filters[2])
        self.attentionblock4 = GridAttentionBlock(filters[3], filters[4], filters[3])
        
        # upsampling
        self.upblock4 = LightUpBlock(filters[4], filters[3])
        self.upblock3 = LightUpBlock(filters[3], filters[2])
        self.upblock2 = LightUpBlock(filters[2], filters[1])
        self.upblock1 = LightUpBlock(filters[1], filters[0])
        
        # deep supervision
        self.dsv4 = DeepSupervision(filters[3], num_classes, scale_factor=8)
        self.dsv3 = DeepSupervision(filters[2], num_classes, scale_factor=4)
        self.dsv2 = DeepSupervision(filters[1], num_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=num_classes, 
                              kernel_size=1, stride=1, padding=0)
        
        # final
        self.final = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        self._initialize_weights()
        
    def forward(self, inputs):
        # Feature extracting
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        
        # Gating Signal Generation
        center = self.center(self.maxpool4(conv4))
        gating = self.gating(center)
        
        # Attention Mechanism
        # Decoder
        g_conv4 = self.attentionblock4(conv4, gating)
        up4 = self.upblock4(g_conv4, center)
        g_conv3 = self.attentionblock3(conv3, up4)
        up3 = self.upblock3(g_conv3, up4)
        g_conv2 = self.attentionblock2(conv2, up3)
        up2 = self.upblock2(g_conv2, up3)
        up1 = self.upblock1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        
        # Final
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))  
        return torch.sigmoid(final)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class LightUNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, base_features=32):
        super(LightUNetPlusPlus, self).__init__()
        
        filters = [base_features * i for i in [1, 2, 4, 8, 16]]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        self.conv0_0 = self._block(in_channels, filters[0])
        self.conv1_0 = self._block(filters[0], filters[1])
        self.conv2_0 = self._block(filters[1], filters[2])
        self.conv3_0 = self._block(filters[2], filters[3])
        self.conv4_0 = self._block(filters[3], filters[4])
        
        self.conv0_1 = self._block(filters[0] + filters[1], filters[0])
        self.conv1_1 = self._block(filters[1] + filters[2], filters[1])
        self.conv2_1 = self._block(filters[2] + filters[3], filters[2])
        self.conv3_1 = self._block(filters[3] + filters[4], filters[3])
        
        self.conv0_2 = self._block(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = self._block(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = self._block(filters[2] * 2 + filters[3], filters[2])
        
        self.conv0_3 = self._block(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = self._block(filters[1] * 3 + filters[2], filters[1])
        
        self.conv0_4 = self._block(filters[0] * 4 + filters[1], filters[0])
        
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self._initialize_weights()
        
    def forward(self, inputs):
        X0_0 = self.conv0_0(inputs)
        X1_0 = self.conv1_0(self.pool(X0_0))
        X0_1 = self.conv0_1(torch.cat([X0_0, self.up(X1_0)], dim=1))
        
        X2_0 = self.conv2_0(self.pool(X1_0))
        X1_1 = self.conv1_1(torch.cat([X1_0, self.up(X2_0)], dim=1))
        X0_2 = self.conv0_2(torch.cat([X0_0, X0_1, self.up(X1_1)], dim=1))
        
        X3_0 = self.conv3_0(self.pool(X2_0))
        X2_1 = self.conv2_1(torch.cat([X2_0, self.up(X3_0)], dim=1))
        X1_2 = self.conv1_2(torch.cat([X1_0, X1_1, self.up(X2_1)], dim=1))
        X0_3 = self.conv0_3(torch.cat([X0_0, X0_1, X0_2, self.up(X1_2)], dim=1))
        
        X4_0 = self.conv4_0(self.pool(X3_0))
        X3_1 = self.conv3_1(torch.cat([X3_0, self.up(X4_0)], dim=1))
        X2_2 = self.conv2_2(torch.cat([X2_0, X2_1, self.up(X3_1)], dim=1))
        X1_3 = self.conv1_3(torch.cat([X1_0, X1_1, X1_2, self.up(X2_2)], dim=1))
        X0_4 = self.conv0_4(torch.cat([X0_0, X0_1, X0_2, X0_3, self.up(X1_3)], dim=1))
        
        return torch.sigmoid(self.final(X0_4))
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(DepthWise_Seperable(in_channels=in_channels, out_channels=out_channels, 
                                                 stride=1, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU6(inplace=True),
                             
                             DepthWise_Seperable(in_channels=out_channels, out_channels=out_channels, 
                                                 stride=1, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU6(inplace=True))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



