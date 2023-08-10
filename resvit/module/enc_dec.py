import torch.nn as nn


def build_enc_dec(enc_dec_cfg, out_layer=False):
    n_stages = int(enc_dec_cfg.n_stages)
    n_resblocks = int(enc_dec_cfg.n_resblocks)
    in_channels = int(enc_dec_cfg.init_channels)
    out_channels = int(enc_dec_cfg.max_channels)
    
    encoder = Encoder(n_stages, n_resblocks, in_channels, out_channels)
    decoder = Decoder(n_stages, n_resblocks, out_channels, in_channels)
    
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
    def __init__(self, n_stages, n_resblocks, in_channels=3, out_channels=512):
        super(Encoder, self).__init__()
        layers = []
        for ith in range(n_stages):
            if ith == 0:
                out_channels = out_channels // 2**(n_stages - 1)
            layer = EncoderLayer(n_resblocks=n_resblocks, in_channels=in_channels, out_channels=out_channels)
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
    def __init__(self, n_stages, n_resblocks, in_channels=512, out_channels=3):
        super(Decoder, self).__init__()
        OUT_CHANNELS = out_channels
        layers = []
        for ith in range(n_stages):
            out_channels = in_channels // 2 if ith < n_stages - 1 else OUT_CHANNELS
            layer = DecoderLayer(n_resblocks=n_resblocks, in_channels=in_channels, out_channels=out_channels) 
            layers.append(layer)
            in_channels = in_channels // 2
        self.layers = nn.ModuleList(layers)
        self.layers_outs = []
    
    def forward(self, x, enc_outs):
        # x: (b, out_channels, h/2^n_stages, w/2^n_stages) -> (b, in_channels, h, w)
        self.layers_outs.clear()
        for layer in self.layers:
            x = x + enc_outs.pop(-1)
            x = layer(x)
            self.layers_outs.append(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_resblocks, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
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
            resblocks.append(ResBlock(out_channels, out_channels))
        self.resblocks = nn.Sequential(*resblocks)
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, 2c, h/2, w/2)
        x = self.downsampling(x)
        x = self.resblocks(x)
        return x


class DecoderLayer(nn.Module):
    
    def __init__(self, n_resblocks, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DecoderLayer, self).__init__()
        resblocks = []
        for _ in range(n_resblocks):
            resblocks.append(ResBlock(in_channels, in_channels))
        self.resblocks = nn.Sequential(*resblocks)
        self.upsampling = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, 2c, h/2, w/2)
        x = self.resblocks(x)
        return self.upsampling(x)


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        # x: (b, c, h, w) -> (b, c, h, w)
        return nn.functional.relu(x + self.block(x))