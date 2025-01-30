import torch
import torch.nn as nn
from mmcv.cnn import (
        ConvModule,
        build_upsample_layer,
        build_norm_layer,
        build_activation_layer,
        UPSAMPLE_LAYERS
    )
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from utils.functions import ConvBnRelu, initialize_weights


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2

        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        # norm = nn.BatchNorm2d(out_channels)
        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        # activate = nn.ReLU(inplace=True)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super(BasicConvBlock, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class UpConvBlock(nn.Module):
    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 skip_connection=True
                 ):
        super(UpConvBlock, self).__init__()
        self.skip_connection = skip_connection
        # Upsample layer
        self.upsample = build_upsample_layer(
            cfg=upsample_cfg,
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Convolution layer
        self.conv_block = conv_block(
            in_channels=2 * skip_channels if skip_connection else skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, skip, x):
        x = self.upsample(x)
        if self.skip_connection:
            x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)

        return x


class UNetDec(nn.Module):
    def __init__(self,
                 embed_dim=[64, 128, 320, 512],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule'),
                 ):
        super(UNetDec, self).__init__()
        self.extra_stages = 2
        num_stages = len(embed_dim)         # 4
        num_stages += self.extra_stages     # 4 + 2 = 6

        extra_embed_dim = [int(embed_dim[0] * 2 ** ((-i) - 1))
                           for i in reversed(range(self.extra_stages))]

        self.embed_dim = [y for x in [extra_embed_dim, embed_dim] for y in x]

        # Upsample
        for i in range(num_stages - 1):     # Upsampling first, then convolution
            decoder = UpConvBlock(
                conv_block=BasicConvBlock,
                in_channels=self.embed_dim[i + 1] * 4,
                skip_channels=self.embed_dim[i] * 4,
                out_channels=self.embed_dim[i] * 4,
                num_convs=2,
                stride=1,
                dilation=1,
                conv_cfg=conv_cfg,  # None
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                skip_connection=True if i >= self.extra_stages else False
            )
            deconv_name = f'deconv_{i}'
            self.add_module(deconv_name, decoder)

        # Last layer
        in_chs = self.embed_dim[self.extra_stages]
        self.conv = nn.Sequential(
            # ConvModule(in_channels=in_chs, out_channels=in_chs, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvBnRelu(in_chs, in_chs, kernel_size=3, padding=1),
            nn.Conv2d(in_chs, 3, kernel_size=3, padding=1, bias=False)
        )

        for i in range(num_stages - 1):
            deconv = getattr(self, f'deconv_{i}')
            initialize_weights(deconv)
        initialize_weights(self.conv)

    def forward(self, x):
        out = x[-1]
        stages = len(self.embed_dim) - 1
        for i in reversed(range(stages)):
            decoder = getattr(self, f'deconv_{i}')
            if i >= self.extra_stages:
                skip = x[i - self.extra_stages]
                out = decoder(skip, out)
            else:
                out = decoder(out, out)
        out = self.conv(out)

        return out
    '''
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNetDec, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    '''
    def init_weights(self, logger, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')