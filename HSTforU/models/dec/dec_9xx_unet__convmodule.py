import torch
import torch.nn as nn

from mmcv.cnn import (ConvModule,
                      build_upsample_layer,
                      build_activation_layer,
                      build_norm_layer,
                      UPSAMPLE_LAYERS
                      )


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
    """

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


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
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

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

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
        self.upsample = build_upsample_layer(
            cfg=upsample_cfg,
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

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


class UNet(nn.Module):
    def __init__(self,
                 num_input_frames=4,
                 embed_dim=[64, 128, 320, 512],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='DeconvModule')
                 ):
        super(UNet, self).__init__()
        self.extra_stages = 2
        num_stages = len(embed_dim)     # 4
        num_stages += self.extra_stages     # 4 + 2 = 6

        extra_embed_dim = [int(embed_dim[0] * 2 ** ((-i) - 1))
                           for i in reversed(range(self.extra_stages))]

        self.embed_dim = [y for x in [extra_embed_dim, embed_dim] for y in x]

        for i in range(num_stages-1):
            decoder = UpConvBlock(
                conv_block=BasicConvBlock,
                in_channels=self.embed_dim[i + 1] * num_input_frames,
                skip_channels=self.embed_dim[i] * num_input_frames,
                out_channels=self.embed_dim[i] * num_input_frames,
                num_convs=2,
                stride=1,
                dilation=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                skip_connection=True if i >= self.extra_stages else False
            )
            decoder_name = f'decoder_layer{i}'
            self.add_module(decoder_name, decoder)

        # Last layer
        in_chs = self.embed_dim[0] * num_input_frames
        self.conv = nn.Sequential(
            ConvModule(in_channels=in_chs, out_channels=in_chs, kernel_size=3, stride=1,
                       padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            # ConvBnRelu(in_chs, in_chs, kernel_size=3, padding=1),
            nn.Conv2d(in_chs, 3, kernel_size=3, padding=1, bias=False)
        )

        for i in range(num_stages-1):
            decoder_layer = getattr(self, f'decoder_layer{i}')
            initialize_weights(decoder_layer)
        initialize_weights(self.conv)

    def forward(self, x):
        out = x[-1]
        num_stages = len(self.embed_dim) - 1
        for i in reversed(range(num_stages)):
            decoder_layer = getattr(self, f'decoder_layer{i}')
            if i >= self.extra_stages:
                skip = x[i - self.extra_stages]
                out = decoder_layer(skip, out)
            else:
                out = decoder_layer(out, out)
        out = self.conv(out)

        return out

    '''
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    '''
