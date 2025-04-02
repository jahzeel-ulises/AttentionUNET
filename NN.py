import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    This block performs a two convolutional layers.
    """
    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------------
        in_channels: int
            Number of channels of the entry.
        out_channels: int
            Out channels after the convolutions.
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1,bias=True)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1,bias=True)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.b1(self.conv1(x))
        x = F.relu(x)
        x = self.b2(self.conv2(x))
        x = F.relu(x)
        return x

class AttentionBlock(nn.Module):
    """
    Additive attention module.
    """
    def __init__(self, in_channels, gating_channels, inter_channels):
        """
        Parameters
        ---------------
        in_channels: int
            Number of channels of the x.
        gating_channels: int
            Number of channels of g.
        inter_channels: int
            Number of channels of the invernal latent space.
        """
        super(AttentionBlock, self).__init__()
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=2, padding=0, bias=True)
        self.b1 = nn.BatchNorm2d(inter_channels)
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.b2 = nn.BatchNorm2d(inter_channels)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.b3 = nn.BatchNorm2d(1)
        self.upconv = nn.Upsample(scale_factor=2)

    def forward(self, x, g):
        theta_x = self.b1(self.W_x(x))
        phi_g = self.b2(self.W_g(g))
        f = F.relu(theta_x + phi_g)
        psi_f = torch.sigmoid(self.b3(self.psi(f)))
        psi_f =self.upconv(psi_f)
        return x * psi_f

class AttentionUNet(nn.Module):
    """
    Attention UNET convolutional network.
    """

    def __init__(self, in_channels, out_channels):
        """
        Parameters
        --------------
        in_channels: int
            Number of channels of the entry.
        out_channels: int
            Out channels after the foward propagation.
        """

        #The filters increase and decrease by 2 factor.
        #The shape of the image increase and decrease by 2 factor
        #The filters number starts from 16, it can be changed.

        super(AttentionUNet, self).__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock( 16,  32)
        self.enc3 = ConvBlock( 32,  64)
        self.enc4 = ConvBlock( 64,128)
        
        self.center = ConvBlock(128,256)
        
        #128
        self.att1 = AttentionBlock( 128, 256, 64)
        self.up1_conv = ConvBlock(256,128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.concat_conv_1 = ConvBlock(256,128)

        #64
        self.att2 = AttentionBlock(64,128,32)
        self.up2_conv = ConvBlock(128,64)
        self.up2 = nn.Upsample(scale_factor=2)
        self.concat_conv_2 = ConvBlock(128,64)

        #32
        self.att3 = AttentionBlock(32,64,16)
        self.up3_conv = ConvBlock(64,32)
        self.up3 = nn.Upsample(scale_factor=2)
        self.concat_conv_3 = ConvBlock(64,32)

        #16
        self.att4 = AttentionBlock(16,32,8)
        self.up4_conv = ConvBlock(32,16)
        self.up4 = nn.Upsample(scale_factor=2)
        self.concat_conv_4 = ConvBlock(32,16)

        self.conv_5 = nn.Conv2d(16,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self, x):
        # Codificación
        
        x1 = self.enc1(x)

        x2 = self.pooling(x1)
        x2 = self.enc2(x2)

        x3 = self.pooling(x2)
        x3 = self.enc3(x3)

        x4 = self.pooling(x3)
        x4 = self.enc4(x4)

        x5 = self.pooling(x4)
        x5 = self.center(x5)

        att1 = self.att1(x4,x5)
        d1 = self.up1_conv(x5)
        d1 = self.up1(d1)
        d1 = torch.cat((att1,d1),dim=1)
        d1 = self.concat_conv_1(d1)

        att2 = self.att2(x3,d1)
        d2 = self.up2_conv(d1)
        d2 = self.up2(d2)
        d2 = torch.cat((att2,d2),dim=1)
        d2 = self.concat_conv_2(d2)

        att3 = self.att3(x2,d2)
        d3 = self.up3_conv(d2)
        d3 = self.up3(d3)
        d3 = torch.cat((att3,d3),dim=1)
        d3 = self.concat_conv_3(d3)

        att4 = self.att4(x1,d3)
        d4 = self.up4_conv(d3)
        d4 = self.up4(d4)
        d4 = torch.cat((att4,d4),dim=1)
        d4 = self.concat_conv_4(d4)

        out = self.conv_5(d4)
        
        return out