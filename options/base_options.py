import argparse
from pathlib import Path
from util import util
import torch
import models
import data


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--dataroot", required=True, help="path to images (should have subfolders trainA, trainB, valA, valB, etc)")
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        # model parameters
        parser.add_argument("--model", type=str, default="cycle_gan", help="chooses which model to use. [cycle_gan | pix2pix | test | colorization]")
        parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels: 3 for RGB and 1 for grayscale")
        parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels: 3 for RGB and 1 for grayscale")
        parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in the last conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in the first conv layer")
        parser.add_argument("--netD", type=str, default="basic", help="specify discriminator architecture [basic | n_layers | pixel | multiscale | tri_rf | swin_mspatch | transgan | oasis | projected]. The basic model is a 70x70 PatchGAN. tri_rf uses one CNN backbone with 10, 70, and 286 receptive-field heads. n_layers allows you to specify the layers in the discriminator")
        parser.add_argument("--netG", type=str, default="resnet_9blocks", help="specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128 | swin_unet]")
        parser.add_argument("--n_layers_D", type=int, default=3, help="only used if netD==n_layers")
        parser.add_argument("--oasis_num_res_blocks", type=int, default=6, help="number of down/up residual blocks used if netD==oasis")
        parser.add_argument("--projected_cond_netD", type=str, default="basic", help="conditional pix2pix discriminator kept alongside projected D when netD==projected")
        parser.add_argument("--projected_backbone", type=str, default="tf_efficientnet_lite0", help="timm EfficientNet-like backbone for projected discriminator")
        parser.add_argument("--projected_no_pretrained", action="store_true", help="disable ImageNet pretrained weights for projected discriminator backbone")
        parser.add_argument("--projected_cout", type=int, default=64, help="random projection channel width for projected discriminator")
        parser.add_argument("--projected_proj_type", type=int, default=2, choices=[0, 1, 2], help="projected discriminator projection type: 0 none, 1 cross-channel mixing, 2 cross-scale mixing")
        parser.add_argument("--projected_num_discs", type=int, default=4, help="number of projected multi-scale discriminator heads")
        parser.add_argument("--projected_no_expand", action="store_true", help="disable expanded channel schedule in projected random projections")
        parser.add_argument("--projected_no_diffaug", action="store_true", help="disable DiffAugment inside projected discriminator")
        parser.add_argument("--projected_diffaug_policy", type=str, default="color,translation,cutout", help="DiffAugment policy for projected discriminator")
        parser.add_argument("--projected_no_interp224", action="store_true", help="disable 224x224 resize before projected discriminator backbone")
        parser.add_argument("--projected_separable", action="store_true", help="use separable down blocks in projected discriminator heads")
        parser.add_argument("--projected_patch", action="store_true", help="use patch down blocks in projected discriminator heads")
        parser.add_argument("--diff_aug", type=str, default="None", help="DiffAugment policy for transgan discriminator, e.g. 'color,translation,cutout'. Set to 'None' to disable")
        parser.add_argument("--norm", type=str, default="instance", help="instance normalization or batch normalization [instance | batch | none | syncbatch]")
        parser.add_argument("--init_type", type=str, default="normal", help="network initialization [normal | xavier | kaiming | orthogonal]")
        parser.add_argument("--init_gain", type=float, default=0.02, help="scaling factor for normal, xavier and orthogonal.")
        parser.add_argument("--no_dropout", action="store_true", help="no dropout for the generator")
        parser.add_argument("--swin_pretrained_path", type=str, default="", help="optional pretrained checkpoint path for Swin-Unet")
        parser.add_argument("--swin_use_checkpoint", action="store_true", help="use gradient checkpointing in Swin-Unet")
        parser.add_argument("--no_scpe_sft_skip", action="store_true", help="disable SCPE-SFT physical-prior skip replacement in Swin-Unet")
        parser.add_argument("--no_cfmoe", action="store_true", help="disable CF-MoE in Swin-Unet decoder")
        parser.add_argument("--loss_type", type=str, default="baseline", choices=["baseline", "baseline_improved", "gan_l1", "gan_l1_ev", "dino_semantic", "dino_structure", "rsac", "dino_topo_micro", "scf_cmpc", "ms_mmd", "gan_l1_ev_dino_tile_topo", "sam_rfl", "sam_align", "wavelet", "wavelet_topo", "strict_freq", "char_ssim_dino_tile", "physical_dino_bridge"], help="loss type for pix2pix model")
        parser.add_argument("--dino_repo", type=str, default="/home/zjj/.cache/torch/hub/facebookresearch_dinov3_main", help="torch.hub repo for DINOv3")
        parser.add_argument("--dino_model_name", type=str, default="dinov3_vitb16", help="DINOv3 model name")
        parser.add_argument("--dino_use_hub_pretrained", action="store_true", help="download DINOv3 weights via torch.hub")
        parser.add_argument("--dino_pretrained_path", type=str, default="", help="local path to DINOv3 pretrained weights")
        parser.add_argument("--dino_input_size", type=int, default=224, help="input size for DINOv3 encoder")
        parser.add_argument("--dino_patch_size", type=int, default=16, help="patch size for DINOv3 encoder")
        parser.add_argument("--sam_checkpoint", type=str, default="/home/zjj/pycharmProject/CS-MoE/datasets/sam_vit_h_4b8939.pth", help="path to SAM ViT-H checkpoint")
        parser.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
        parser.add_argument("--sam_cache_dir", type=str, default="", help="directory containing offline SAM priors")
        parser.add_argument("--sam_cache_in_memory", action="store_true", help="preload all SAM priors into RAM for faster training")
        parser.add_argument("--sam_use_offline_cache", action="store_true", help="load precomputed SAM priors from cache")
        parser.add_argument("--sam_mask_min_area", type=int, default=64, help="minimum mask area to keep")
        parser.add_argument("--sam_mask_max_area", type=int, default=1000000000, help="maximum mask area to keep")
        parser.add_argument("--sam_boundary_width", type=int, default=3, help="boundary width for SAM boundary masks")
        parser.add_argument("--sam_max_masks", type=int, default=32, help="maximum number of masks to keep per image")
        parser.add_argument("--sam_points_per_side", type=int, default=32, help="points per side for SAM automatic mask generator")
        parser.add_argument("--sam_pred_iou_thresh", type=float, default=0.88, help="predicted IoU threshold for SAM mask generator")
        parser.add_argument("--sam_stability_score_thresh", type=float, default=0.95, help="stability threshold for SAM mask generator")
        parser.add_argument("--sam_use_online_fallback", action="store_true", help="generate SAM priors online if cache is missing")
        parser.add_argument("--lambda_huber", type=float, default=100.0, help="weight for Huber reconstruction loss in SAM-RFL")
        parser.add_argument("--lambda_sam_recon", type=float, default=1.0, help="weight for SAM region reconstruction loss")
        parser.add_argument("--lambda_sam_rfc", type=float, default=1.0, help="weight for SAM region frequency consistency loss")
        parser.add_argument("--lambda_sam_boundary", type=float, default=1.0, help="weight for SAM boundary loss")
        # dataset parameters
        parser.add_argument("--dataset_mode", type=str, default="unaligned", help="chooses how datasets are loaded. [unaligned | aligned | single | colorization]")
        parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
        parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")
        parser.add_argument("--num_threads", default=4, type=int, help="# threads for loading data")
        parser.add_argument("--batch_size", type=int, default=4, help="input batch size")
        parser.add_argument("--load_size", type=int, default=286, help="scale images to this size")
        parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
        parser.add_argument("--max_dataset_size", type=int, default=float("inf"), help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.")
        parser.add_argument("--preprocess", type=str, default="resize_and_crop", help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]")
        parser.add_argument("--no_flip", action="store_true", help="if specified, do not flip the images for data augmentation")
        parser.add_argument("--display_winsize", type=int, default=256, help="display window size for both visdom and HTML")
        # additional parameters
        parser.add_argument("--epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument("--load_iter", type=int, default="0", help="which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]")
        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        parser.add_argument("--suffix", default="", type=str, help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}")
        # MoE hierarchical expert variance loss
        parser.add_argument("--lambda_ev", type=float, default=0.01, help="overall weight for hierarchical expert variance loss")
        parser.add_argument("--lambda_ev_macro", type=float, default=1.0, help="relative weight for macro physical-direction expert variance loss")
        parser.add_argument("--lambda_ev_micro", type=float, default=1.0, help="relative weight for intra-group sub-expert variance loss")
        parser.add_argument("--ev_eps", type=float, default=1e-10, help="epsilon for hierarchical expert variance loss")
        parser.add_argument("--ev_tau", type=float, default=0.01, help="reserved margin tau for survival-style expert variance loss")
        parser.add_argument("--num_experts_per_group", type=int, default=4, help="number of experts per semantic group in CF-MoE")
        parser.add_argument("--intra_topk", type=int, default=2, help="Top-k experts selected per pixel inside each group")
        parser.add_argument("--topk_renorm", action="store_true", help="renormalize surviving Top-k routing weights to sum to 1")
        # wandb parameters
        parser.add_argument("--use_wandb", action="store_true", help="if specified, then init wandb logging")
        parser.add_argument("--wandb_project_name", type=str, default="CycleGAN-and-pix2pix", help="specify wandb project name")
        # Localized Wavelet Energy Style Loss
        parser.add_argument("--lambda_wavelet", type=float, default=1.0, help="weight for Localized Wavelet Energy Style loss")
        parser.add_argument("--wavelet_levels", type=int, default=3, help="number of DWT decomposition levels (K)")
        parser.add_argument("--wavelet_var_weight", type=float, default=1.0, help="variance balancing coefficient lambda_var for wavelet loss")
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        util.mkdirs(expr_dir)
        file_name = expr_dir / f"{opt.phase}_opt.txt"
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt
