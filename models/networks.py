import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nnutils
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from timm.models.layers import trunc_normal_

from .swin_transformer_unet_skip_expand_decoder_sys import (
    SwinTransformerSys,
    PatchEmbed,
    BasicLayer,
    PatchMerging,
)
from .standalone_discriminator import TransGANDiscriminator


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net


def load_swin_pretrained_weights(net, pretrained_path):
    """Load pretrained weights for SwinTransformerSys with key/shape adaptation."""
    if not pretrained_path:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    except TypeError:
        # For older PyTorch versions (e.g. 1.x) that don't support `weights_only`
        pretrained_dict = torch.load(pretrained_path, map_location=device)

    if "model" not in pretrained_dict:
        print("---start load pretrained model by splitting---")
        pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
        for k in list(pretrained_dict.keys()):
            if "output" in k:
                print(f"delete key:{k}")
                del pretrained_dict[k]
        msg = net.load_state_dict(pretrained_dict, strict=False)
        print(msg)
        return

    pretrained_dict = pretrained_dict["model"]
    print("---start load pretrained model of swin encoder---")

    model_dict = net.state_dict()
    full_dict = copy.deepcopy(pretrained_dict)
    for k, v in pretrained_dict.items():
        if "layers." in k:
            current_layer_num = 3 - int(k[7:8])
            current_k = "layers_up." + str(current_layer_num) + k[8:]
            full_dict.update({current_k: v})

    for k in list(full_dict.keys()):
        if k in model_dict and full_dict[k].shape != model_dict[k].shape:
            print(f"delete:{k};shape pretrain:{full_dict[k].shape};shape model:{model_dict[k].shape}")
            del full_dict[k]

    msg = net.load_state_dict(full_dict, strict=False)
    # print(msg)


class SwinUnetGenerator(nn.Module):
    """Swin-Unet generator wrapper with stage-A default hyper-parameters."""

    def __init__(
        self,
        input_nc,
        output_nc,
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_path=None,
        route_dim=None,
        num_experts_per_group=4,
        intra_topk=2,
        topk_renorm=False,
        use_scpe_sft_skip=True,
        use_cfmoe=True,
    ):
        super(SwinUnetGenerator, self).__init__()
        self.model = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            num_classes=output_nc,
            out_chans=output_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            dino_route_dim=route_dim,
            num_experts_per_group=num_experts_per_group,
            intra_topk=intra_topk,
            topk_renorm=topk_renorm,
            use_scpe_sft_skip=use_scpe_sft_skip,
            use_cfmoe=use_cfmoe,
        )
        if pretrained_path:
            print(f"pretrained_path:{pretrained_path}")
            load_swin_pretrained_weights(self.model, pretrained_path)

    def forward(self, x, return_features=False):
        return self.model(x, return_features=return_features)


def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    swin_img_size=256,
    swin_pretrained_path="",
    swin_use_checkpoint=False,
    route_dim=None,
    num_experts_per_group=4,
    intra_topk=2,
    topk_renorm=False,
    use_scpe_sft_skip=True,
    use_cfmoe=True,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256 | swin_unet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        swin_img_size (int) -- input image size for Swin-Unet.
        swin_pretrained_path (str) -- optional pretrained checkpoint path for Swin-Unet.
        swin_use_checkpoint (bool) -- whether to use checkpointing in Swin-Unet.

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "swin_unet":
        net = SwinUnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            img_size=swin_img_size,
            use_checkpoint=swin_use_checkpoint,
            pretrained_path=swin_pretrained_path,
            route_dim=route_dim,
            num_experts_per_group=num_experts_per_group,
            intra_topk=intra_topk,
            topk_renorm=topk_renorm,
            use_scpe_sft_skip=use_scpe_sft_skip,
            use_cfmoe=use_cfmoe,
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return net


def define_D(
    input_nc,
    ndf,
    netD,
    n_layers_D=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    use_spectral_norm=False,
    diff_aug=None,
    crop_size=256,
    oasis_num_res_blocks=6,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel | multiscale | tri_rf | swin_mspatch | transgan
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        use_spectral_norm  -- whether to apply spectral normalization (not used by transgan)
        diff_aug (str)     -- DiffAugment policy for transgan, e.g. 'color,translation,cutout' or 'None'
        crop_size (int)    -- input image size for transgan discriminator (default 256)
        oasis_num_res_blocks (int) -- number of down/up residual blocks for oasis

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

        [multiscale]: three PatchGAN discriminators at 1x, 1/2x, 1/4x scales.

        [tri_rf]: single CNN backbone with local, medium, and global PatchGAN
                  heads whose receptive fields are 10, 70, and 286 pixels.

        [swin_mspatch]: 4-stage cascade Swin-PatchGAN discriminator producing four patch maps.
                         Symmetric to the generator's Swin encoder: depths=(2,2,2,2),
                         dims=128->256->512->1024, window_size=8.

        [transgan]: pure-Transformer discriminator based on TransGAN (multi-scale window-attention).
        Supports optional DiffAugment via the `diff_aug` parameter.

        [oasis]: OASIS-style encoder-decoder dense discriminator. In pix2pix this
        branch predicts two per-pixel logits: real vs fake.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if use_spectral_norm:
        from torch.nn.utils import spectral_norm as sn
    else:
        sn = lambda layer: layer

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, spectral_norm_fn=sn)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, spectral_norm_fn=sn)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, spectral_norm_fn=sn)
    elif netD == "multiscale":
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "tri_rf":
        net = TriRFDiscriminator(input_nc, ndf, norm_layer=norm_layer, spectral_norm_fn=sn)
    elif netD == "swin_mspatch":
        net = MultiScaleSwinPatchDiscriminator(input_nc, use_spectral_norm=use_spectral_norm)
    elif netD == "transgan":
        net = TransGANDiscriminator(
            img_size=crop_size,
            in_chans=input_nc,
            embed_dim=384,
            depth=7,
            num_heads=4,
            patch_size=4,
            window_size=8,
            norm_layer="ln",
            act_layer="leakyrelu",
            diff_aug=diff_aug,
        )
    elif netD == "oasis":
        net = OASISDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            output_nc=2,
            num_res_blocks=oasis_num_res_blocks,
            spectral_norm_fn=sn,
        )
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
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
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, spectral_norm_fn=lambda x: x):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm_fn(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [spectral_norm_fn(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [spectral_norm_fn(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [spectral_norm_fn(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class TriRFDiscriminator(nn.Module):
    """Single-backbone triple-receptive-field PatchGAN discriminator.

    Three branches share the shallow feature-extraction trunk and then
    fork off to produce single-channel prediction maps at three distinct
    receptive-field scales — each optimal for a different image-detail
    level in SAR-to-optical translation.

    Forward path (all with kernel=4, padding=1, LeakyReLU 0.2):
      conv1  : 256x256 -> 128x128   (stride=2)  RF=6
      conv2  : 128x128 ->  64x64   (stride=2)  RF=10   <-- local branch fork
      conv3  :  64x64  ->  32x32   (stride=2)  RF=34   <-- medium branch fork
      conv4  :  32x32  ->  16x16   (stride=2)  RF=70   <-- shared before dilation

    Three prediction heads:
      1. Local  (from conv2,  64x64, k=4 s=1): RF=10   — 2x2 patches over 10x10 px image regions
                 Detects: fine-grain textures (sand ripples, water noise, sensor speckle).
      2. Medium (conv3+medium_body, 32x32, k=4 s=1): RF=70  — classic PatchGAN depth.
                 Detects: edges, contours, object boundaries.
      3. Global (conv4 + dilated stack, 16x16, k=3 s=1 d=4/8): RF=286
                 — theoretical RF exceeds image size (256+padding), so every
                   spatial position "sees" the entire image.
                 Detects: large-area colour discontinuities, unrealistic landscape assemblies.

    Input:  [B, input_nc, 256, 256]  (for cGAN, input_nc = 6: [SAR, optical/target])
    Output: [pred_local (B,1,64,64), pred_medium (B,1,32,32), pred_global (B,1,16,16)]
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, spectral_norm_fn=lambda x: x):
        super(TriRFDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        def conv_norm_relu(in_ch, out_ch, stride, kernel=kw, padding=padw):
            return nn.Sequential(
                spectral_norm_fn(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=use_bias)
                ),
                norm_layer(out_ch),
                nn.LeakyReLU(0.2, True),
            )

        # ---- Shared trunk ----
        self.conv1 = nn.Sequential(
            spectral_norm_fn(nn.Conv2d(input_nc, ndf,     kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = conv_norm_relu(ndf,     ndf * 2, stride=2)          # 128->64, RF=10
        self.conv3 = conv_norm_relu(ndf * 2, ndf * 4, stride=2)          #  64->32, RF=34
        self.conv4 = conv_norm_relu(ndf * 4, ndf * 8, stride=2)          #  32->16, RF=70

        # ---- Branch 1: Local (10x10 RF, 64x64 output) ----
        self.local_head = spectral_norm_fn(
            nn.Conv2d(ndf * 2, 1, kernel_size=kw, stride=1, padding=padw)
        )

        # ---- Branch 2: Medium (70x70 RF, 32x32 output) ----
        # Lightweight body to refine features before final classification.
        self.medium_body = nn.Sequential(
            conv_norm_relu(ndf * 4, ndf * 4, stride=1),   # preserve 32x32
        )
        self.medium_head = spectral_norm_fn(
            nn.Conv2d(ndf * 4, 1, kernel_size=kw, stride=1, padding=padw)
        )

        # ---- Branch 3: Global (286x286 RF, 16x16 output) ----
        # Progressive channel narrowing (512->256->128) keeps the global branch
        # lightweight while still reaching 286-pixel theoretical RF via two
        # dilated convolutions (d=4, d=8) instead of stacking heavy stride-2 layers.
        #   conv4 base          -> 16x16,  RF=70
        #   Dilated d=4 (k=3)  -> 16x16,  RF=118
        #   Dilated d=8 (k=3)   -> 16x16,  RF=214
        # With 3px padding accumulated from stride layers: 214 + 72 = 286
        self.global_dilated = nn.Sequential(
            nn.Sequential(
                spectral_norm_fn(
                    nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=4, dilation=4, bias=use_bias)
                ),
                norm_layer(ndf * 4),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                spectral_norm_fn(
                    nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=8, dilation=8, bias=use_bias)
                ),
                norm_layer(ndf * 2),
                nn.LeakyReLU(0.2, True),
            ),
        )
        self.global_head = spectral_norm_fn(
            nn.Conv2d(ndf * 2, 1, kernel_size=kw, stride=1, padding=padw)
        )

    def forward(self, x):
        """Return [local, medium, global] single-channel prediction maps."""
        f1 = self.conv1(x)     # 128x128
        f2 = self.conv2(f1)    #  64x64, RF=10  <- local
        f3 = self.conv3(f2)    #  32x32, RF=34  <- medium
        f4 = self.conv4(f3)    #  16x16, RF=70  <- global (base)

        pred_local  = self.local_head(f2)
        pred_medium = self.medium_head(self.medium_body(f3))
        pred_global = self.global_head(self.global_dilated(f4))

        return [pred_local, pred_medium, pred_global]


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, spectral_norm_fn=lambda x: x):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            spectral_norm_fn(nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.2, True),
            spectral_norm_fn(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_norm_fn(nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class OASISResidualBlockD(nn.Module):
    """Residual block used by the OASIS-style dense discriminator."""

    def __init__(self, fin, fout, up_or_down, spectral_norm_fn, first=False):
        super().__init__()
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = fin != fout

        fmiddle = fout
        if first:
            self.conv1 = spectral_norm_fn(nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1))
        elif up_or_down > 0:
            self.conv1 = nn.Sequential(
                nn.LeakyReLU(0.2, False),
                nn.Upsample(scale_factor=2),
                spectral_norm_fn(nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1)),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.LeakyReLU(0.2, False),
                spectral_norm_fn(nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1)),
            )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            spectral_norm_fn(nn.Conv2d(fmiddle, fout, kernel_size=3, stride=1, padding=1)),
        )
        if self.learned_shortcut:
            self.conv_s = spectral_norm_fn(nn.Conv2d(fin, fout, kernel_size=1, stride=1, padding=0))

        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            return x

        if self.up_or_down > 0:
            x = self.sampling(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.up_or_down < 0:
            x = self.sampling(x)
        return x

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        return x_s + dx


class OASISDiscriminator(nn.Module):
    """OASIS-style U-Net discriminator for dense real/fake supervision.

    The original OASIS discriminator predicts semantic classes plus one fake
    class. This pix2pix branch keeps the same encoder-decoder residual shape,
    but uses two dense classes because this project has no semantic labels.
    """

    def __init__(self, input_nc, ndf=64, output_nc=2, num_res_blocks=6, spectral_norm_fn=lambda layer: layer):
        super().__init__()
        if num_res_blocks < 2:
            raise ValueError("OASISDiscriminator requires at least 2 residual blocks.")
        if num_res_blocks > 6:
            raise ValueError("OASISDiscriminator supports at most 6 residual blocks with the default channel schedule.")

        multipliers = [2, 2, 4, 4, 8, 8]
        self.channels = [input_nc] + [min(ndf * multipliers[i], ndf * 8) for i in range(num_res_blocks)]

        self.body_down = nn.ModuleList()
        for i in range(num_res_blocks):
            self.body_down.append(
                OASISResidualBlockD(
                    self.channels[i],
                    self.channels[i + 1],
                    up_or_down=-1,
                    spectral_norm_fn=spectral_norm_fn,
                    first=(i == 0),
                )
            )

        self.body_up = nn.ModuleList()
        self.body_up.append(
            OASISResidualBlockD(
                self.channels[-1],
                self.channels[-2],
                up_or_down=1,
                spectral_norm_fn=spectral_norm_fn,
            )
        )
        for i in range(1, num_res_blocks - 1):
            self.body_up.append(
                OASISResidualBlockD(
                    2 * self.channels[-1 - i],
                    self.channels[-2 - i],
                    up_or_down=1,
                    spectral_norm_fn=spectral_norm_fn,
                )
            )
        self.body_up.append(
            OASISResidualBlockD(
                2 * self.channels[1],
                ndf,
                up_or_down=1,
                spectral_norm_fn=spectral_norm_fn,
            )
        )
        self.layer_up_last = nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        encoder_res = []
        for block in self.body_down:
            x = block(x)
            encoder_res.append(x)

        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            skip = encoder_res[-i - 1]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = self.body_up[i](torch.cat((skip, x), dim=1))
        return self.layer_up_last(x)



class MultiScaleSwinPatchDiscriminator(nn.Module):
    """Multi-scale Swin-PatchGAN discriminator with 4-stage cascade structure.

    The discriminator uses a 4-stage Swin Transformer encoder with symmetric
    architecture to the generator's encoder. It outputs four single-channel
    patch maps from Stage 0, Stage 1, Stage 2, and Stage 3 features.

    Architecture is symmetric with the generator's Swin encoder:
      - 4 stages with depths=(2,2,2,2)
      - Feature dims: 128 -> 256 -> 512 -> 1024
      - Window size: 8 for stable W-MSA shift attention
      - num_heads: (4, 8, 16, 32) maintaining 32 channels per head

    Key fixes vs. original:
      - Spectral Normalization on PatchEmbed projection and final heads (Lipschitz约束，稳定GAN训练)
      - Per-stage loss weights: coarse stages weighted more (避免细粒度stage主导训练)
      - Stage 0 (4x4 patch) heavily downweighted to suppress pixel-level overfitting
    """

    def __init__(
        self,
        input_nc,
        img_size=256,
        patch_size=4,
        embed_dim=128,
        depths=(2, 2, 2, 2),
        num_heads=(4, 8, 16, 32),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths

        sn_fn = nnutils.spectral_norm if use_spectral_norm else lambda x: x

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.patch_embed.proj = sn_fn(self.patch_embed.proj)
        patches_resolution = self.patch_embed.patches_resolution

        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                route_dim=None,
            )
            self.layers.append(layer)
            if i_layer < len(depths) - 1:
                self.downsamples.append(
                    PatchMerging(
                        input_resolution=(
                            patches_resolution[0] // (2 ** i_layer),
                            patches_resolution[1] // (2 ** i_layer),
                        ),
                        dim=int(embed_dim * 2 ** i_layer),
                        norm_layer=nn.LayerNorm,
                    )
                )

        self.norms = nn.ModuleList([
            nn.LayerNorm(int(embed_dim * 2 ** i)) for i in range(len(depths))
        ])
        self.heads = nn.ModuleList([
            sn_fn(nn.Conv2d(int(embed_dim * 2 ** i), 1, kernel_size=3, padding=1))
            for i in range(len(depths))
        ])

        # Per-stage loss weights: heavily downweight the finest stage (4x4 patch)
        # Stage 0: 4x4  -> weight 0.05  (too fine, causes grid artifact)
        # Stage 1: 8x8  -> weight 0.15
        # Stage 2: 16x16-> weight 0.30
        # Stage 3: 32x32-> weight 0.50
        weights = torch.tensor([0.05, 0.15, 0.30, 0.50], dtype=torch.float32)
        self.register_buffer("stage_weights", weights / weights.sum())

    def _reshape_to_2d(self, x, stage_idx):
        B, L, C = x.shape
        H = self.patch_embed.patches_resolution[0] // (2 ** stage_idx)
        W = self.patch_embed.patches_resolution[1] // (2 ** stage_idx)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x_norm = self.norms[idx](x)
            feat = self._reshape_to_2d(x_norm, idx)
            outputs.append(self.heads[idx](feat))
            if idx < len(self.layers) - 1:
                x = self.downsamples[idx](x)

        return outputs

    def multiscale_gan_loss(self, preds, target_is_real, criterionGAN):
        """Compute weighted multi-scale GAN loss. Coarser stages have higher weight."""
        weighted_losses = []
        for i, pred in enumerate(preds):
            weight = self.stage_weights[i].item()
            loss = criterionGAN(pred, target_is_real)
            weighted_losses.append(weight * loss)
        return sum(weighted_losses)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
