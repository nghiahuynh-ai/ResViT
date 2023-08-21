import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


def build_enc_dec(enc_dec_cfg, out_layer=False):
    n_stages = int(enc_dec_cfg.n_stages)
    n_resblocks = int(enc_dec_cfg.n_resblocks)
    in_channels = int(enc_dec_cfg.init_channels)
    out_channels = int(enc_dec_cfg.max_channels)
    deform_stages = enc_dec_cfg.deform_stages
    rdeform_stages = []
    for i in deform_stages:
        rdeform_stages.append(n_stages - i - 1)
    
    encoder = Encoder(n_stages, n_resblocks, in_channels, out_channels, deform_stages)
    decoder = Decoder(n_stages, out_channels, in_channels)
    
    if out_layer:
        out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return encoder, decoder, out
    else:
        return encoder, decoder


class Encoder(nn.Module):
    def __init__(self, n_stages, n_resblocks, in_channels=3, out_channels=512, deform_stages=[]):
        super(Encoder, self).__init__()
        layers = []
        for ith in range(n_stages):
            if ith == 0:
                out_channels = out_channels // 2**(n_stages - 1)
            deform = True if ith in deform_stages else False
            layer = EncoderLayer(n_resblocks=n_resblocks, in_channels=in_channels, out_channels=out_channels, deform=deform)
            in_channels = out_channels
            out_channels *= 2
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.layers_outs = []
    
    def forward(self, x):
        # x: (b, in_channels, h, w) -> (b, out_channels, h/2^n_stages, w/2^n_stages)
        self.layers_outs.clear()
        for layer in self.layers:
            x = layer(x)
            self.layers_outs.append(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_stages, in_channels=512, out_channels=3):
        super(Decoder, self).__init__()
        OUT_CHANNELS = out_channels
        layers = []
        for ith in range(n_stages):
            out_channels = in_channels // 2 if ith < n_stages - 1 else OUT_CHANNELS
            layer = DecoderLayer(in_channels=in_channels, out_channels=out_channels) 
            layers.append(layer)
            in_channels = in_channels // 2
        self.layers = nn.ModuleList(layers)
        self.layers_outs = []
    
    def forward(self, x, enc_outs):
        # x: (b, out_channels, h/2^n_stages, w/2^n_stages) -> (b, in_channels, h, w)
        for layer in self.layers:
            x = x + enc_outs.pop(-1)
            x = layer(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_resblocks, in_channels, out_channels, kernel_size=3, stride=2, padding=1, deform=False):
        super(EncoderLayer, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
        )
        resblocks = []
        for _ in range(n_resblocks):
            resblocks.append(ResBlock(out_channels, out_channels, deform=deform))
        self.resblocks = nn.Sequential(*resblocks)
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, 2c, h/2, w/2)
        x = self.downsampling(x)
        x = self.resblocks(x)
        return x


class DecoderLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DecoderLayer, self).__init__()
        self.upsampling = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, 2c, h/2, w/2)
        return nn.functional.relu(self.upsampling(x))


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deform=False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        if deform:
            self.conv2 = DeformConv2d(
                in_channels=out_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
        else:
            self.conv2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, c, h, w)
        residual = x
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)) + residual)
        return x
    
    
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DeformConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )

        # nn.init.constant_(self.offset_conv.weight, 0.)
        # nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )

        # nn.init.constant_(self.modulator_conv.weight, 0.)
        # nn.init.constant_(self.modulator_conv.bias, 0.)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias
        )

    def forward(self, x):

        offset = self.offset(x)
        modulator = 2. * torch.sigmoid(self.modulator(x))
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.conv.weight,
            bias=self.conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
            dilation=self.dilation
        )
        return x