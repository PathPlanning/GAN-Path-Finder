import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from spectral import SpectralNorm


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy):
    epoch_count = 1
    niter = 100
    niter_decay = 100
    lr_decay_iters = 50

    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_id='cuda:0', context_encoder=False, use_ce=False,  unet=False, use_attn=False, n_blocks=9):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if use_attn:
        net = SAGANGenerator(input_nc, output_nc, norm_layer=norm_layer, n_blocks=n_blocks, ngf=ngf)
    if context_encoder:
        net = ContexEncoder(input_nc, output_nc, nBottleneck=4000, ngf=ngf)
    else:
        if not unet:
            net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, n_blocks=n_blocks, use_ce=use_ce)
        else:
            net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_ce=use_ce)
    return init_net(net, init_type, init_gain, gpu_id)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, use_ce=False, padding_type='reflect'):

        assert(n_blocks >= 0)

        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf, output_nc, use_ce)

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])

        return self.outc(out['u2'])


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias, use_spectral=False):
        super(Inconv, self).__init__()
        if use_spectral:
            self.inconv = nn.Sequential(
                                        nn.ReflectionPad2d(3),
                                        SpectralNorm(nn.Conv2d(in_ch, out_ch,
                                                               kernel_size=7,
                                                               padding=0,
                                                               bias=use_bias)),
                                        norm_layer(out_ch),
                                        nn.ReLU(True)
                                        )

        else:
            self.inconv = nn.Sequential(
                                        nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                                                  bias=use_bias),
                                        norm_layer(out_ch),
                                        nn.ReLU(True)
                                        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias, use_spectral=False):
        super(Down, self).__init__()
        if use_spectral:
            self.down = nn.Sequential(
                                      SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                                             stride=2, padding=1, bias=use_bias)),
                                      norm_layer(out_ch),
                                      nn.ReLU(True)
                                      )
        else:
            self.down = nn.Sequential(
                                      nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                                stride=2, padding=1, bias=use_bias),
                                      norm_layer(out_ch),
                                      nn.ReLU(True)
                                      )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias, use_spectral=False):
        super(Up, self).__init__()
        if use_spectral:
            self.up = nn.Sequential(
                                    # nn.Upsample(scale_factor=2, mode='nearest'),
                                    # nn.Conv2d(in_ch, out_ch,
                                    #           kernel_size=3, stride=1,
                                    #           padding=1, bias=use_bias),

                                    SpectralNorm(nn.ConvTranspose2d(in_ch, out_ch,
                                                                    kernel_size=3, stride=2,
                                                                    padding=1, output_padding=1,
                                                                    bias=use_bias)),
                                    norm_layer(out_ch),
                                    nn.ReLU(True)
                                    )
        else:
            self.up = nn.Sequential(
                                    # nn.Upsample(scale_factor=2, mode='nearest'),
                                    # nn.Conv2d(in_ch, out_ch,
                                    #           kernel_size=3, stride=1,
                                    #           padding=1, bias=use_bias),

                                    nn.ConvTranspose2d(in_ch, out_ch,
                                                       kernel_size=3, stride=2,
                                                       padding=1, output_padding=1,
                                                       bias=use_bias),
                                    norm_layer(out_ch),
                                    nn.ReLU(True)
                                    )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch, use_ce=False, use_spectral=False):
        super(Outconv, self).__init__()
        if use_ce:
            self.outconv = nn.Sequential(
                                         nn.ReflectionPad2d(3),
                                         nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                                        )
        else:
            if use_spectral:
                self.outconv = nn.Sequential(
                                             nn.ReflectionPad2d(3),
                                             SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0)),
                                             nn.Tanh()
                                            )
            else:
                self.outconv = nn.Sequential(
                                             nn.ReflectionPad2d(3),
                                             nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                                             nn.Tanh()
                                            )

    def forward(self, x):
        x = self.outconv(x)
        return x


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False,
             init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'attn':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                  use_attention=True, use_spectral=True)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_ce=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_ce=use_ce)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, use_ce=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if use_ce:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ContexEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, nBottleneck=4000, ngf=64):
        super(ContexEncoder, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 16 x 16
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 8 x 8
            nn.Conv2d(ngf, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 4 x 4
            nn.Conv2d(ngf*4, nBottleneck, 4, bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(nBottleneck, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf*2) x 16 x 16
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out  # attention


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 use_attention=False, use_spectral=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if use_spectral:
            sequence = [
                        SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                        nn.LeakyReLU(0.2, True)
                        ]
        else:
            sequence = [
                        nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                        nn.LeakyReLU(0.2, True)
                        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if use_spectral:
                sequence += [
                              SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                           kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                             ]
            else:
                sequence += [
                              nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                             ]
            sequence += [
                          norm_layer(ndf * nf_mult),
                          nn.LeakyReLU(0.2, True)
                         ]
            # if n == 1:
            #     sequence += [ Self_Attn(128, 'relu')]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if use_spectral:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                      kernel_size=kw, stride=1, padding=padw, bias=use_bias))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=1, padding=padw, bias=use_bias)]

        sequence += [norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        # sequence += [ Self_Attn(512, 'relu') ]

        if use_spectral:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
                    nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                    norm_layer(ndf * 2),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
                   ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class SAGANGenerator(nn.Module):
    """Generator."""

    def __init__(self, input_nc, output_nc, norm_layer,use_ce=False, n_blocks=5, padding_type='reflect', ngf=64):
        super(SAGANGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias, True)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias, True)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias, True)
        self.down3 = Down(ngf * 4, ngf * 8, norm_layer, use_bias, True)
        # self.down4 = Down(ngf * 8, ngf * 16, norm_layer, use_bias, True)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64,  'relu')

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 8, padding_type=padding_type, norm_layer=norm_layer,
                               use_dropout=True, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        # self.up1 = Up(ngf * 16, ngf * 8, norm_layer, use_bias, True)
        self.up1 = Up(ngf * 8, ngf * 4, norm_layer, use_bias, True)
        self.up2 = Up(ngf * 4, ngf * 2, norm_layer, use_bias, True)
        self.up3 = Up(ngf * 2, ngf, norm_layer, use_bias, True)

        self.outc = Outconv(ngf, output_nc, use_ce, True)

    def forward(self):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['d3'] = self.down3(out['d2'])

        out['bottle'] = self.resblocks(out['d3'])

        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])
        out['a1'] = self.attn1(out['u2'])
        out['u3'] = self.up4(out['a1'])
        out['a2'] = self.attn2(out['u3'])

        return self.outc(out['a2'])


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
