import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as pytorch_msssim_ssim
except ImportError:
    pytorch_msssim_ssim = None

from util.loss import FocalFrequencyLoss, StrictFocalFreqLoss
from .base_model import BaseModel
from . import networks
from .dino_encoder import FrozenDinoV3Encoder
from .projected_discriminator import ProjectedDiscriminator
from util.dino_losses import (
    relaxed_inter_image_nce,
    relaxed_inter_image_nce_improved,
    intra_image_affinity_alignment,
    local_gram_alignment,
    non_local_semantic_decoupling,
    semantic_aware_masked_l1,
    dino_soft_correspondence_feature_loss,
    dino_cross_modal_prototype_contrastive_loss,
    dino_topological_consistency,
    dino_semantic_edge_alignment,
    dino_ipd_cos_alignment,
    multi_scale_dino_perceptual_loss,
    dino_tile_cosine_loss,
    tara_topological_kl_loss,
)
from util.sam_rfl import SamPriorCache, SamRFLosses, SamPrior, build_boundary_from_mask
from .sam_align_modules import SOAStack
from util.wavelet_loss import LocalizedWaveletEnergyLoss


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument("--lambda_char", type=float, default=100.0, help="weight for Charbonnier reconstruction loss")
            parser.add_argument("--char_eps", type=float, default=1e-3, help="epsilon for Charbonnier reconstruction loss")
            parser.add_argument("--lambda_ssim", type=float, default=10.0, help="weight for SSIM loss")
            parser.add_argument("--lambda_dino_tile", type=float, default=1.0, help="weight for random-tile DINO perceptual loss")
            parser.add_argument("--dino_tile_size", type=int, default=224, help="random tile size for DINO perceptual loss")
            parser.add_argument("--dino_tile_layers", type=str, default="3,6,9,11", help="comma-separated DINO block indices for random-tile perceptual loss")
            parser.add_argument("--dino_tile_layer_weights", type=str, default="1.0,1.0,1.0,1.0", help="comma-separated per-layer weights for random-tile DINO perceptual loss")
            parser.add_argument("--tara_tile_size", type=int, default=224, help="aligned random tile size for TARA DINO tile/topology losses")
            parser.add_argument("--tara_tile_layers", type=str, default="3,6,9,11", help="comma-separated DINO block indices for TARA tile cosine loss")
            parser.add_argument("--tara_tile_layer_weights", type=str, default="1.0,1.0,0.5,0.5", help="comma-separated per-layer weights for TARA tile cosine loss")
            parser.add_argument("--lambda_tara_tile", type=float, default=0.02, help="weight for TARA local DINO tile cosine loss")
            parser.add_argument("--lambda_tara_topo", type=float, default=0.01, help="weight for TARA intra-image topological KL loss")
            parser.add_argument("--tara_topo_layer", type=int, default=11, help="DINO block index used for TARA topological KL loss")
            parser.add_argument("--tara_topo_tau", type=float, default=0.07, help="temperature for TARA topological KL distributions")
            parser.add_argument("--lambda_projected_D", type=float, default=1.0, help="weight for the extra projected discriminator loss when netD==projected")
            parser.add_argument("--lambda_dino_inter", type=float, default=0.2, help="weight for DINO relaxed inter-image loss")
            parser.add_argument("--lambda_dino_intra", type=float, default=1.0, help="weight for DINO intra-image affinity loss")
            parser.add_argument("--dino_window", type=int, default=3, help="local window size for relaxed NCE")
            parser.add_argument("--dino_tau", type=float, default=0.07, help="temperature for DINO losses")
            parser.add_argument("--dino_neg_samples", type=int, default=256, help="negative samples for relaxed NCE")
            parser.add_argument("--dino_semantic_neg_threshold", type=float, default=0.4, help="GT-internal similarity threshold for negative filtering (0~1)")
            parser.add_argument("--dino_min_negatives", type=int, default=64, help="minimum negatives per patch after filtering")
            parser.add_argument("--dino_affinity_samples", type=int, default=256, help="patch samples for affinity loss")
            parser.add_argument("--lambda_dino_masked_l1", type=float, default=100.0, help="weight for DINO semantic-aware masked L1 loss")
            # parser.add_argument("--lambda_dino_perceptual", type=float, default=1.0, help="weight for DINO perceptual L1 loss")
            parser.add_argument("--lambda_dino_lga", type=float, default=1.0, help="weight for DINO local gram alignment loss")
            parser.add_argument("--lambda_dino_nsd", type=float, default=1.0, help="weight for DINO non-local semantic decoupling loss")
            parser.add_argument("--lambda_dino_sam", type=float, default=1.0, help="weight for DINO semantic-aware masked L1 loss (SAM)")
            # parser.add_argument("--dino_repo", type=str, default="facebookresearch/dinov3", help="torch.hub repo for DINOv3")
            # parser.add_argument("--dino_model_name", type=str, default="dinov3_vitb16", help="DINOv3 model name")
            # parser.add_argument("--dino_use_hub_pretrained", action="store_true", help="download DINOv3 weights via torch.hub")
            # parser.add_argument("--dino_pretrained_path", type=str, default="", help="local path to DINOv3 pretrained weights")
            # parser.add_argument("--dino_input_size", type=int, default=224, help="input size for DINOv3 encoder")
            # parser.add_argument("--dino_patch_size", type=int, default=16, help="patch size for DINOv3 encoder")
            
            parser.add_argument("--lambda_rsac", type=float, default=1.0, help="weight for RSAC routing semantic anchor loss")
            parser.add_argument("--dino_mask_gamma", type=float, default=5.0, help="gamma for DINO confidence mask")
            # parser.add_argument("--dino_perceptual_layers", type=str, default="3,6,9,11", help="comma-separated DINO block indices for perceptual loss")
            parser.add_argument("--dino_lga_window", type=int, default=3, help="local window size for LGA loss")
            parser.add_argument("--dino_nsd_pairs", type=int, default=1024, help="number of non-local pairs for NSD loss")
            parser.add_argument("--dino_nsd_margin", type=float, default=0.2, help="margin for NSD loss")
            parser.add_argument("--dino_nsd_min_distance", type=int, default=2, help="min patch distance for NSD loss")
            parser.add_argument("--dino_sam_gamma", type=float, default=5.0, help="gamma for SAM mask")
            parser.add_argument("--lambda_dino_topo", type=float, default=1.0, help="weight for DINO topological consistency loss")
            parser.add_argument("--lambda_dino_micro", type=float, default=1.0, help="weight for DINO micro quality loss")
            parser.add_argument("--lambda_dino_edge", type=float, default=1.0, help="weight for DINO semantic edge alignment loss")
            parser.add_argument("--lambda_dino_ipd", type=float, default=1.0, help="weight for DINO IPD cosine alignment loss")
            parser.add_argument("--dino_topo_layer", type=int, default=9, help="DINO block index used for topological loss")
            parser.add_argument("--dino_micro_layer", type=int, default=3, choices=[3, 4], help="DINO block index used for micro loss")
            parser.add_argument("--dino_topo_remove_diag", action="store_true", help="remove diagonal entries when computing DINO SSM topology loss")
            parser.add_argument("--lambda_dino_scf", type=float, default=1.0, help="weight for DINO soft-correspondence feature loss")
            parser.add_argument("--lambda_dino_cmpc", type=float, default=1.0, help="weight for DINO cross-modal prototype contrastive loss")
            parser.add_argument("--dino_scf_window", type=int, default=3, help="local token window size for SCF loss")
            parser.add_argument("--dino_scf_tau", type=float, default=0.07, help="temperature for SCF local attention")
            parser.add_argument("--dino_scf_p", type=int, default=1, choices=[1, 2], help="distance type for SCF: 1 for L1, 2 for L2")
            parser.add_argument("--dino_cmpc_k", type=int, default=4, help="number of K-means clusters for CMPC")
            parser.add_argument("--dino_cmpc_tau", type=float, default=0.07, help="temperature for CMPC contrastive logits")
            parser.add_argument("--dino_cmpc_iters", type=int, default=5, help="K-means iterations for CMPC")
            parser.add_argument(
                "--lambda_ms_mmd",
                type=float,
                default=1.0,
                help="weight for multi-scale simple MMD loss",
            )
            parser.add_argument(
                "--ms_mmd_layers",
                type=str,
                default="3,6,9,11",
                help="comma-separated DINO block indices for MS-MMD",
            )
            parser.add_argument(
                "--ms_mmd_layer_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="comma-separated per-layer weights for MS-MMD",
            )
            parser.add_argument(
                "--ms_mmd_proj_dim",
                type=int,
                default=256,
                help="projection dimension for MS-MMD adapters",
            )
            parser.add_argument(
                "--ms_mmd_sigma",
                type=float,
                default=1.0,
                help="single-bandwidth sigma for RBF kernel in MS-MMD",
            )
            parser.add_argument(
                "--ms_mmd_eps",
                type=float,
                default=1e-8,
                help="epsilon for numerical stability in MS-MMD",
            )
            parser.add_argument(
                "--lambda_sam_align_deep",
                type=float,
                default=0.1,
                help="weight for SAM deep alignment loss",
            )
            parser.add_argument(
                "--lambda_sam_align_mid",
                type=float,
                default=0.1,
                help="weight for SAM mid alignment loss",
            )
            parser.add_argument(
                "--lambda_sam_align_shallow",
                type=float,
                default=0.1,
                help="weight for SAM shallow alignment loss",
            )
            parser.add_argument(
                "--lambda_sam_align_l1_deep",
                type=float,
                default=0.1,
                help="weight for SAM deep L1 alignment loss",
            )
            parser.add_argument(
                "--lambda_sam_align_l1_mid",
                type=float,
                default=0.1,
                help="weight for SAM mid L1 alignment loss",
            )
            parser.add_argument(
                "--lambda_sam_align_l1_shallow",
                type=float,
                default=0.1,
                help="weight for SAM shallow L1 alignment loss",
            )
            parser.add_argument(
                "--sam_align_size",
                type=int,
                default=1024,
                help="input size for SAM alignment features",
            )
            parser.add_argument(
                "--sam_align_layers",
                type=str,
                default="4,8,12",
                help="comma-separated SAM block indices used for alignment",
            )
            parser.add_argument(
                "--lambda_dino_perceptual",
                type=float,
                default=1.0,
                help="weight for Multi-Scale DINO Perceptual Alignment loss",
            )
            parser.add_argument(
                "--dino_perceptual_layers",
                type=str,
                default="3,6,9,11",
                help="comma-separated DINO block indices for perceptual loss",
            )
            parser.add_argument(
                "--dino_perceptual_layer_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="comma-separated per-layer weights for DINO perceptual loss",
            )
            parser.add_argument(
                "--lambda_strict_freq",
                type=float,
                default=1.0,
                help="weight for Strict Focal Frequency Loss (complex domain)",
            )
            parser.add_argument(
                "--strict_freq_alpha",
                type=float,
                default=2.0,
                help="focal exponent alpha for Strict Focal Frequency Loss",
            )
            parser.add_argument(
                "--lambda_phy_dino",
                type=float,
                default=1.0,
                help="overall weight for synchronized physical-DINO bridge loss",
            )
            parser.add_argument(
                "--phy_dino_tile_size",
                type=int,
                default=224,
                help="synchronized crop size for physical-DINO bridge loss",
            )
            parser.add_argument(
                "--phy_dino_layers",
                type=str,
                default="3,6,9,11",
                help="comma-separated DINO block indices paired with c2,c3,c4,c5",
            )
            parser.add_argument(
                "--phy_pasa_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="per-layer weights for PASA loss",
            )
            parser.add_argument(
                "--phy_cti_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="per-layer weights for CTI loss",
            )
            parser.add_argument(
                "--phy_rge_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="per-layer weights for RGE loss",
            )
            parser.add_argument(
                "--phy_gac_weights",
                type=str,
                default="1.0,1.0,1.0,1.0",
                help="per-layer weights for GAC loss",
            )
            parser.add_argument(
                "--phy_rge_window",
                type=int,
                default=3,
                help="local token window size for rigid Gram energy loss",
            )
            parser.add_argument(
                "--phy_rge_chunk_size",
                type=int,
                default=64,
                help="number of local Gram windows computed per chunk",
            )
            parser.add_argument(
                "--phy_eps",
                type=float,
                default=1e-8,
                help="epsilon for physical-DINO bridge numerical stability",
            )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.use_projected_D = self.isTrain and opt.netD == "projected"
        self.netD_mode = opt.projected_cond_netD if self.use_projected_D else opt.netD
        if self.netD_mode == "projected":
            raise ValueError("projected_cond_netD cannot be 'projected'. Use a normal conditional discriminator such as 'basic'.")
        if self.isTrain and opt.loss_type == "char_ssim_dino_tile" and pytorch_msssim_ssim is None:
            raise ImportError("loss_type=char_ssim_dino_tile requires pytorch-msssim. Install it with: pip install pytorch-msssim")
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        ev_loss_names = ["G_EV", "G_EV_MACRO", "G_EV_MICRO"]
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"] + ev_loss_names
        if self.use_projected_D:
            self.loss_names += ["G_PROJECTED", "D_projected_real", "D_projected_fake"]
        if opt.loss_type in {"baseline", "baseline_improved", "dino_semantic", "dino_structure", "dino_topo_micro", "scf_cmpc", "ms_mmd", "sam_rfl", "sam_align", "wavelet", "wavelet_topo", "strict_freq"}:
            self.loss_names += ["G_DINO_INTER", "G_DINO_INTRA"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
            if self.use_projected_D:
                self.model_names.append("D_projected")
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        self.ms_mmd_layers = []
        self.ms_mmd_layer_weights = []
        self.ms_mmd_adapters = None
        self.tara_tile_layers = []
        self.tara_tile_layer_weights = []
        self.tara_dino_layers = []
        self.phy_dino_layers = []
        self.phy_pasa_weights = []
        self.phy_cti_weights = []
        self.phy_rge_weights = []
        self.phy_gac_weights = []
        self.phy_sar_keys = ["c2", "c3", "c4", "c5"]
        self.phy_sar_adapters = None
        self.sam_cache = SamPriorCache(opt.sam_cache_dir) if opt.sam_cache_dir else None
        self.sam_boundary_width = opt.sam_boundary_width
        self.sam_priors = []
        self.sam_use_online_fallback = opt.sam_use_online_fallback
        self.sam_cache_in_memory = opt.sam_cache_in_memory
        # define networks (both generator and discriminator)
        self.dino_encoder = None
        if opt.loss_type in {"rsac", "baseline", "baseline_improved", "dino_semantic", "dino_structure", "dino_topo_micro", "scf_cmpc", "ms_mmd", "wavelet", "wavelet_topo", "strict_freq", "char_ssim_dino_tile", "physical_dino_bridge", "gan_l1_ev_dino_tile_topo"}:
            self.dino_encoder = FrozenDinoV3Encoder(
                repo=opt.dino_repo,
                model_name=opt.dino_model_name,
                input_size=opt.dino_input_size,
                patch_size=opt.dino_patch_size,
                use_hub_pretrained=opt.dino_use_hub_pretrained,
                pretrained_path=opt.dino_pretrained_path,
            ).to(self.device)
            self.dino_encoder.eval()
        self.sam_model = None
        self.sam_model_type = getattr(opt, "sam_model_type", None)
        self.sam_align_adapters = None
        if opt.loss_type == "sam_rfl" or opt.loss_type == "sam_align":
            try:
                from segment_anything import sam_model_registry
                self.sam_model_type = self._infer_sam_model_type(opt.sam_checkpoint, self.sam_model_type)
                sam = sam_model_registry[self.sam_model_type](checkpoint=opt.sam_checkpoint)
                self.sam_model = sam.to(self.device)
                self.sam_model.eval()
                for p in self.sam_model.parameters():
                    p.requires_grad = False
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize SAM model from {opt.sam_checkpoint}: {exc}") from exc

        if self.isTrain and opt.loss_type == "sam_align":
            self.sam_align_adapters = nn.ModuleDict(
                {
                    "deep": SOAStack(
                        out_channels=384,
                        num_blocks=2,
                        target_size=(16, 16),
                        downsample_schedule=[True, True],
                    ),
                    "mid": SOAStack(
                        out_channels=192,
                        num_blocks=1,
                        target_size=(32, 32),
                        downsample_schedule=[True],
                    ),
                    "shallow": SOAStack(
                        out_channels=96,
                        num_blocks=1,
                        target_size=(64, 64),
                        downsample_schedule=[False],
                    ),
                }
            ).to(self.device)

        if self.isTrain and opt.loss_type == "ms_mmd":
            self.ms_mmd_layers = [int(x) for x in opt.ms_mmd_layers.split(",") if x.strip()]
            self.ms_mmd_layer_weights = [float(x) for x in opt.ms_mmd_layer_weights.split(",") if x.strip()]
            if len(self.ms_mmd_layers) != len(self.ms_mmd_layer_weights):
                raise ValueError("ms_mmd_layers and ms_mmd_layer_weights must have the same length.")
            if self.dino_encoder is None or self.dino_encoder.embed_dim is None:
                raise ValueError("DINO encoder embed_dim is required for ms_mmd loss.")
            self.ms_mmd_adapters = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.Conv2d(self.dino_encoder.embed_dim, opt.ms_mmd_proj_dim, kernel_size=1, bias=False),
                        nn.GroupNorm(1, opt.ms_mmd_proj_dim),
                        nn.ReLU(inplace=True),
                    )
                    for layer in self.ms_mmd_layers
                }
            ).to(self.device)

        route_dim = None
        if opt.loss_type == "rsac":
            route_dim = self.dino_encoder.embed_dim
            if route_dim is None:
                name = (opt.dino_model_name or "").lower()
                if "vitg" in name:
                    route_dim = 1536
                elif "vitl" in name:
                    route_dim = 1024
                elif "vitb" in name:
                    route_dim = 768
                elif "vits" in name:
                    route_dim = 384
            if route_dim is None:
                raise ValueError("Unable to infer DINOv3 embed dim for RSAC; please set dino_model_name to a vit{g,l,b,s} variant.")

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.crop_size,
            opt.swin_pretrained_path,
            opt.swin_use_checkpoint,
            route_dim=route_dim,
            num_experts_per_group=opt.num_experts_per_group,
            intra_topk=opt.intra_topk,
            topk_renorm=opt.topk_renorm,
            use_scpe_sft_skip=not getattr(opt, "no_scpe_sft_skip", False),
            use_cfmoe=not getattr(opt, "no_cfmoe", False),
        )

        if self.isTrain and opt.loss_type == "physical_dino_bridge":
            if opt.netG != "swin_unet":
                raise ValueError("loss_type=physical_dino_bridge requires --netG swin_unet to expose c2-c5 SCPE priors.")
            if getattr(opt, "no_scpe_sft_skip", False):
                raise ValueError("loss_type=physical_dino_bridge requires SCPE priors; remove --no_scpe_sft_skip.")
            if self.dino_encoder is None or self.dino_encoder.embed_dim is None:
                raise ValueError("DINO encoder embed_dim is required for physical_dino_bridge loss.")

            self.phy_dino_layers = self._parse_int_list(opt.phy_dino_layers, "phy_dino_layers")
            self.phy_pasa_weights = self._parse_float_list(opt.phy_pasa_weights, "phy_pasa_weights")
            self.phy_cti_weights = self._parse_float_list(opt.phy_cti_weights, "phy_cti_weights")
            self.phy_rge_weights = self._parse_float_list(opt.phy_rge_weights, "phy_rge_weights")
            self.phy_gac_weights = self._parse_float_list(opt.phy_gac_weights, "phy_gac_weights")
            self._validate_physical_dino_config()

            swin_embed_dim = int(getattr(self.netG.model, "embed_dim", 96))
            sar_channels = {
                "c2": swin_embed_dim // 2,
                "c3": swin_embed_dim,
                "c4": swin_embed_dim * 2,
                "c5": swin_embed_dim * 4,
            }
            self.phy_sar_adapters = nn.ModuleDict(
                {
                    key: nn.Conv2d(channels, self.dino_encoder.embed_dim, kernel_size=1)
                    for key, channels in sar_channels.items()
                }
            ).to(self.device)
            self.netG.phy_sar_adapters = self.phy_sar_adapters

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                self.netD_mode,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                use_spectral_norm=opt.use_spectral_norm_D,
                diff_aug=opt.diff_aug,
                crop_size=opt.crop_size,
                oasis_num_res_blocks=opt.oasis_num_res_blocks,
            )
            if self.use_projected_D:
                self.netD_projected = ProjectedDiscriminator(
                    input_nc=opt.output_nc,
                    im_res=opt.crop_size,
                    cout=opt.projected_cout,
                    expand=not opt.projected_no_expand,
                    proj_type=opt.projected_proj_type,
                    num_discs=opt.projected_num_discs,
                    backbone=opt.projected_backbone,
                    pretrained=not opt.projected_no_pretrained,
                    diffaug=not opt.projected_no_diffaug,
                    diffaug_policy=opt.projected_diffaug_policy,
                    interp224=not opt.projected_no_interp224,
                    separable=opt.projected_separable,
                    patch=opt.projected_patch,
                )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            self.criterionOASIS = torch.nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss()
            if opt.loss_type == "sam_rfl":
                self.criterionHuber = torch.nn.SmoothL1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            g_params = list(self.netG.parameters())
            if self.ms_mmd_adapters is not None:
                g_params += list(self.ms_mmd_adapters.parameters())
            d_params = list(self.netD.parameters())
            if self.use_projected_D:
                d_params += self.netD_projected.trainable_parameters()
            self.optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr * opt.lr_D_factor, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.loss_type == "baseline": #v1
                self.loss_names += ["G_DINO_INTER", "G_DINO_INTRA"]
            if opt.loss_type == "gan_l1_ev":
                self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"] + ev_loss_names
            if opt.loss_type == "gan_l1_ev_dino_tile_topo":
                self.loss_names += ["G_DINO_TILE", "G_TARA_TOPO"]
                self.tara_tile_layers = self._parse_int_list(opt.tara_tile_layers, "tara_tile_layers")
                self.tara_tile_layer_weights = self._parse_float_list(opt.tara_tile_layer_weights, "tara_tile_layer_weights")
                if len(self.tara_tile_layers) != len(self.tara_tile_layer_weights):
                    raise ValueError("tara_tile_layers and tara_tile_layer_weights must have the same length.")
                if int(opt.tara_tile_size) <= 0:
                    raise ValueError("tara_tile_size must be positive.")
                if float(opt.tara_topo_tau) <= 0:
                    raise ValueError("tara_topo_tau must be positive.")
                self.tara_dino_layers = list(dict.fromkeys(self.tara_tile_layers + [int(opt.tara_topo_layer)]))
                self.wavelet_loss = LocalizedWaveletEnergyLoss(
                    levels=opt.wavelet_levels,
                    lambda_var=opt.wavelet_var_weight,
                ).to(self.device)


            if opt.loss_type == "wavelet_topo":
                self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"] + ev_loss_names + ["G_WAVELET", "G_DINO_TOPO_INTRA"]
                self.wavelet_loss = LocalizedWaveletEnergyLoss(
                    levels=opt.wavelet_levels,
                    lambda_var=opt.wavelet_var_weight,
                ).to(self.device)

            if opt.loss_type == "dino_semantic":
                self.dino_perceptual_layers = [int(x) for x in opt.dino_perceptual_layers.split(",") if x.strip()]
            if opt.loss_type == "strict_freq":
                self.strict_freq_loss = StrictFocalFreqLoss(
                    alpha=getattr(opt, "strict_freq_alpha", 2.0)
                ).to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        self.sam_priors = []
        if self.opt.loss_type == "sam_rfl" and self.sam_cache is not None:
            image_paths = self.image_paths if isinstance(self.image_paths, (list, tuple)) else [self.image_paths]
            if self.sam_cache_in_memory:
                self.sam_cache.preload(image_paths)
            self.sam_priors = [self.sam_cache.load(path) for path in image_paths]
            if self.sam_use_online_fallback and self.sam_model is not None:
                missing_indices = [i for i, prior in enumerate(self.sam_priors) if prior is None]
                if missing_indices:
                    try:
                        from segment_anything import SamAutomaticMaskGenerator
                        from PIL import Image
                        import numpy as np
                        mask_generator = SamAutomaticMaskGenerator(
                            self.sam_model,
                            points_per_side=self.opt.sam_points_per_side,
                            pred_iou_thresh=self.opt.sam_pred_iou_thresh,
                            stability_score_thresh=self.opt.sam_stability_score_thresh,
                            min_mask_region_area=self.opt.sam_mask_min_area,
                        )
                        for i in missing_indices:
                            image_path = image_paths[i]
                            img = np.array(Image.open(image_path).convert("RGB"))
                            masks_info = mask_generator.generate(img)
                            masks = []
                            scores = []
                            areas = []
                            for item in sorted(masks_info, key=lambda x: x.get("area", 0), reverse=True)[: self.opt.sam_max_masks]:
                                mask = item["segmentation"].astype(np.float32)
                                area = int(mask.sum())
                                if self.opt.sam_mask_min_area <= area <= self.opt.sam_mask_max_area:
                                    masks.append(mask)
                                    scores.append(float(item.get("predicted_iou", 1.0)))
                                    areas.append(area)
                            boundaries = [build_boundary_from_mask(mask, width=self.sam_boundary_width) for mask in masks]
                            prior = SamPrior(masks=masks, boundaries=boundaries, areas=areas, scores=scores, image_size=img.shape[:2])
                            self.sam_priors[i] = prior
                            self.sam_cache.memory_cache[image_path] = prior
                    except Exception:
                        pass

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.loss_type == "sam_align":
            self.fake_B, self.g_features = self.netG(self.real_A, return_features=True)
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)
            self.g_features = {}

    def _multiscale_gan_loss(self, preds, target_is_real):
        losses = [self.criterionGAN(pred, target_is_real) for pred in preds]
        return torch.stack(losses).mean()

    def _tri_rf_lsgan_loss(self, preds, target_is_real):
        if not isinstance(preds, (list, tuple)) or len(preds) != 3:
            raise ValueError("tri_rf discriminator must return [local, medium, global] prediction maps.")
        target_value = 1.0 if target_is_real else 0.0
        # Branch weights: local=0.2, medium=0.3, global=0.5
        # Local dominates early training (high-res signal, strong gradients) so
        # it gets the smallest weight. Global gets the largest to encourage the
        # generator to focus on overall structure before fine details.
        weights = torch.tensor([0.2, 0.3, 0.5], device=preds[0].device, dtype=preds[0].dtype)
        losses = torch.stack([
            F.mse_loss(pred, torch.full_like(pred, target_value))
            for pred in preds
        ])
        return (losses * weights).sum()

    def _projected_D_module(self):
        net = self.netD_projected
        if hasattr(net, "module"):
            net = net.module
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        return net

    def _set_projected_D_requires_grad(self, requires_grad):
        self._projected_D_module().set_discriminator_requires_grad(requires_grad)

    @staticmethod
    def _parse_int_list(value, name):
        items = [x.strip() for x in str(value).split(",") if x.strip()]
        if not items:
            raise ValueError(f"{name} must contain at least one integer.")
        try:
            return [int(x) for x in items]
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated integer list, got {value!r}.") from exc

    @staticmethod
    def _parse_float_list(value, name):
        items = [x.strip() for x in str(value).split(",") if x.strip()]
        if not items:
            raise ValueError(f"{name} must contain at least one float.")
        try:
            return [float(x) for x in items]
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated float list, got {value!r}.") from exc

    def _validate_physical_dino_config(self):
        expected = len(self.phy_sar_keys)
        configs = {
            "phy_dino_layers": self.phy_dino_layers,
            "phy_pasa_weights": self.phy_pasa_weights,
            "phy_cti_weights": self.phy_cti_weights,
            "phy_rge_weights": self.phy_rge_weights,
            "phy_gac_weights": self.phy_gac_weights,
        }
        for name, values in configs.items():
            if len(values) != expected:
                raise ValueError(f"{name} must have {expected} entries for c2,c3,c4,c5, got {len(values)}.")
        if int(self.opt.phy_dino_tile_size) <= 0:
            raise ValueError("phy_dino_tile_size must be positive.")
        if int(self.opt.phy_rge_window) <= 0:
            raise ValueError("phy_rge_window must be positive.")
        if int(self.opt.phy_rge_chunk_size) <= 0:
            raise ValueError("phy_rge_chunk_size must be positive.")

    def _generator_module(self):
        net = self.netG
        if hasattr(net, "module"):
            net = net.module
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        return net

    def _synchronized_random_tile_with_boxes(self, pred, target, tile_size):
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"pred and target must have the same spatial size, got {pred.shape[-2:]} and {target.shape[-2:]}.")

        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")

        batch, _, height, width = pred.shape
        if height < tile_size or width < tile_size:
            pred = F.interpolate(pred, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            crop_boxes = [(0, 0, height, width, height, width) for _ in range(batch)]
            return pred, target, crop_boxes

        max_top = height - tile_size
        max_left = width - tile_size
        pred_tiles = []
        target_tiles = []
        crop_boxes = []
        for batch_idx in range(batch):
            top = torch.randint(0, max_top + 1, (1,), device=pred.device).item() if max_top > 0 else 0
            left = torch.randint(0, max_left + 1, (1,), device=pred.device).item() if max_left > 0 else 0
            pred_tiles.append(pred[batch_idx : batch_idx + 1, :, top : top + tile_size, left : left + tile_size])
            target_tiles.append(target[batch_idx : batch_idx + 1, :, top : top + tile_size, left : left + tile_size])
            crop_boxes.append((top, left, tile_size, tile_size, height, width))

        return torch.cat(pred_tiles, dim=0), torch.cat(target_tiles, dim=0), crop_boxes

    def _crop_pool_sar_feature(self, sar_feat, crop_boxes, output_size):
        if sar_feat.shape[0] != len(crop_boxes):
            raise ValueError(f"SAR feature batch {sar_feat.shape[0]} does not match crop box count {len(crop_boxes)}.")

        _, _, feat_h, feat_w = sar_feat.shape
        pooled = []
        for batch_idx, (top, left, crop_h, crop_w, image_h, image_w) in enumerate(crop_boxes):
            image_h = max(int(image_h), 1)
            image_w = max(int(image_w), 1)
            top = max(int(top), 0)
            left = max(int(left), 0)
            bottom = min(top + max(int(crop_h), 1), image_h)
            right = min(left + max(int(crop_w), 1), image_w)

            feat_top = min(max((top * feat_h) // image_h, 0), feat_h - 1)
            feat_left = min(max((left * feat_w) // image_w, 0), feat_w - 1)
            feat_bottom = min(max(((bottom * feat_h) + image_h - 1) // image_h, feat_top + 1), feat_h)
            feat_right = min(max(((right * feat_w) + image_w - 1) // image_w, feat_left + 1), feat_w)

            sar_crop = sar_feat[batch_idx : batch_idx + 1, :, feat_top:feat_bottom, feat_left:feat_right]
            pooled.append(F.adaptive_avg_pool2d(sar_crop, output_size=output_size))

        return torch.cat(pooled, dim=0)

    def _gather_token_windows(self, token_grid, window):
        # token_grid: [B, H, W, D]
        _, height, width, _ = token_grid.shape
        radius = int(window) // 2
        padded = F.pad(token_grid, (0, 0, radius, radius, radius, radius))
        windows = []
        for dy in range(int(window)):
            for dx in range(int(window)):
                windows.append(padded[:, dy : dy + height, dx : dx + width, :])
        return torch.stack(windows, dim=-2)

    def _local_gram_l1(self, pred_tokens, gt_tokens, grid_size, window, chunk_size):
        batch, num_tokens, channels = pred_tokens.shape
        grid_h, grid_w = grid_size
        if num_tokens != grid_h * grid_w:
            raise ValueError(f"Token count {num_tokens} does not match DINO grid {grid_size}.")

        pred_grid = pred_tokens.view(batch, grid_h, grid_w, channels)
        gt_grid = gt_tokens.view(batch, grid_h, grid_w, channels)
        pred_windows = self._gather_token_windows(pred_grid, window).reshape(-1, int(window) * int(window), channels)
        gt_windows = self._gather_token_windows(gt_grid, window).reshape(-1, int(window) * int(window), channels)

        total_windows = pred_windows.shape[0]
        loss_sum = pred_tokens.new_tensor(0.0)
        chunk_size = int(chunk_size)
        for start in range(0, total_windows, chunk_size):
            end = min(start + chunk_size, total_windows)
            pred_chunk = pred_windows[start:end]
            gt_chunk = gt_windows[start:end]
            gram_pred = torch.bmm(pred_chunk.transpose(1, 2), pred_chunk)
            gram_gt = torch.bmm(gt_chunk.transpose(1, 2), gt_chunk)
            loss_sum = loss_sum + (gram_pred - gram_gt).abs().mean(dim=(1, 2)).sum()

        return loss_sum / max(total_windows, 1)

    def _compute_physical_dino_bridge_loss(self):
        if self.dino_encoder is None:
            raise RuntimeError("DINO encoder is required for physical_dino_bridge loss.")
        if self.phy_sar_adapters is None:
            raise RuntimeError("SAR adapters are required for physical_dino_bridge loss.")

        pred_tile, gt_tile, crop_boxes = self._synchronized_random_tile_with_boxes(
            self.fake_B,
            self.real_B,
            self.opt.phy_dino_tile_size,
        )

        pred_layers = self.dino_encoder.forward_intermediate_layers(pred_tile, self.phy_dino_layers)
        with torch.no_grad():
            gt_layers = self.dino_encoder.forward_intermediate_layers(gt_tile, self.phy_dino_layers)

        num_expected = len(self.phy_dino_layers)
        if len(pred_layers) < num_expected or len(gt_layers) < num_expected:
            raise RuntimeError(
                f"DINO returned {len(pred_layers)} pred layers and {len(gt_layers)} GT layers; "
                f"expected {num_expected} layers {self.phy_dino_layers}."
            )

        generator = self._generator_module()
        generator_core = getattr(generator, "model", generator)
        scpe_all_priors = getattr(generator_core, "last_scpe_all_priors", None)
        if not scpe_all_priors:
            raise RuntimeError("Generator did not expose SCPE c2-c5 priors for physical_dino_bridge loss.")

        grid_size = self.dino_encoder.grid_size
        target_n = int(grid_size[0] * grid_size[1])
        eps = float(self.opt.phy_eps)
        pasa_losses = []
        cti_losses = []
        rge_losses = []
        gac_losses = []

        for idx, sar_key in enumerate(self.phy_sar_keys):
            pred_tokens = pred_layers[idx]
            gt_tokens = gt_layers[idx].detach()
            pred_tokens, gt_tokens = self._align_pred_gt_tokens(pred_tokens, gt_tokens, target_n=target_n)

            sar_feat = scpe_all_priors.get(sar_key)
            if sar_feat is None:
                raise RuntimeError(f"Missing SCPE prior {sar_key} for physical_dino_bridge loss.")
            sar_pooled = self._crop_pool_sar_feature(sar_feat.detach(), crop_boxes, output_size=grid_size)
            v_sar = self.phy_sar_adapters[sar_key](sar_pooled)
            v_sar = v_sar.flatten(2).transpose(1, 2).contiguous()
            v_sar = self._align_tokens_to_target(v_sar, target_n)

            pred_norm = F.normalize(pred_tokens, dim=-1, eps=eps)
            gt_norm = F.normalize(gt_tokens, dim=-1, eps=eps)

            sar_strength = v_sar.norm(p=2, dim=-1)
            sar_strength = sar_strength / sar_strength.amax(dim=1, keepdim=True).clamp_min(eps)
            pasa = (sar_strength * (pred_norm - gt_norm).abs().sum(dim=-1)).mean()

            sim_pred = F.cosine_similarity(v_sar, pred_tokens, dim=-1, eps=eps)
            sim_gt = F.cosine_similarity(v_sar, gt_tokens, dim=-1, eps=eps)
            cti = F.mse_loss(sim_pred, sim_gt)

            rge = self._local_gram_l1(
                pred_tokens,
                gt_tokens,
                grid_size=grid_size,
                window=self.opt.phy_rge_window,
                chunk_size=self.opt.phy_rge_chunk_size,
            )

            gt_global = gt_tokens.mean(dim=1, keepdim=True)
            gt_global = gt_global.expand_as(gt_tokens)
            d_pred = F.cosine_similarity(pred_tokens, gt_global, dim=-1, eps=eps)
            d_gt = F.cosine_similarity(gt_tokens, gt_global, dim=-1, eps=eps)
            gac = F.mse_loss(d_pred, d_gt)

            pasa_losses.append(pasa * self.phy_pasa_weights[idx])
            cti_losses.append(cti * self.phy_cti_weights[idx])
            rge_losses.append(rge * self.phy_rge_weights[idx])
            gac_losses.append(gac * self.phy_gac_weights[idx])

        return (
            torch.stack(pasa_losses).sum(),
            torch.stack(cti_losses).sum(),
            torch.stack(rge_losses).sum(),
            torch.stack(gac_losses).sum(),
        )

    def _oasis_target(self, prediction, target_is_real):
        target_value = 0 if target_is_real else 1
        return torch.full(
            (prediction.shape[0], prediction.shape[2], prediction.shape[3]),
            target_value,
            dtype=torch.long,
            device=prediction.device,
        )

    def _oasis_loss(self, prediction, target_is_real):
        target = self._oasis_target(prediction, target_is_real)
        return self.criterionOASIS(prediction, target)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)

        if self.netD_mode == "oasis":
            self.loss_D_fake = self._oasis_loss(pred_fake, False)
            self.loss_D_real = self._oasis_loss(pred_real, True)
        elif self.netD_mode == "swin_mspatch":
            # swin_mspatch uses its own weighted multiscale loss (coarse stages weighted more)
            self.loss_D_fake = self.netD.multiscale_gan_loss(pred_fake, False, self.criterionGAN)
            self.loss_D_real = self.netD.multiscale_gan_loss(pred_real, True, self.criterionGAN)
        elif self.netD_mode == "tri_rf":
            self.loss_D_fake = self._tri_rf_lsgan_loss(pred_fake, False)
            self.loss_D_real = self._tri_rf_lsgan_loss(pred_real, True)
        elif self.netD_mode == "multiscale":
            self.loss_D_fake = self._multiscale_gan_loss(pred_fake, False)
            self.loss_D_real = self._multiscale_gan_loss(pred_real, True)
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.use_projected_D:
            pred_projected_fake = self.netD_projected(self.fake_B.detach())
            pred_projected_real = self.netD_projected(self.real_B)
            self.loss_D_projected_fake = self.criterionGAN(pred_projected_fake, False)
            self.loss_D_projected_real = self.criterionGAN(pred_projected_real, True)
            self.loss_D = self.loss_D + (self.loss_D_projected_fake + self.loss_D_projected_real) * 0.5 * self.opt.lambda_projected_D
        self.loss_D.backward()

    def _routing_map_list(self, routing_maps):
        if routing_maps is None:
            return []
        if isinstance(routing_maps, dict):
            return [routing_maps]
        return routing_maps

    def _cv_squared(self, usage, eps):
        mean = usage.mean()
        var = (usage - mean).pow(2).mean()
        return var / (mean + eps).pow(2)

    def _hierarchical_macro_ev_loss(self, routing_maps, eps):
        losses = []
        for routing_map in self._routing_map_list(routing_maps):
            for key in ("low", "high"):
                prob = routing_map.get(key)
                if prob is None:
                    continue
                usage = prob.mean(dim=(0, 2, 3))
                losses.append(self._cv_squared(usage, eps))

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _hierarchical_micro_ev_loss(self, routing_maps, eps):
        losses = []
        for routing_map in self._routing_map_list(routing_maps):
            for key in ("l2l", "l2h", "h2l", "h2h"):
                prob = routing_map.get(key)
                if prob is None:
                    continue
                usage = prob.mean(dim=(0, 2, 3))
                losses.append(self._cv_squared(usage, eps))

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _align_tokens_to_target(self, tokens, target_n):
        # tokens: [B, N, D]
        B, N, D = tokens.shape
        if N == target_n:
            return tokens
        if N > target_n:
            # Prefer removing potential global/register tokens at the front.
            return tokens[:, -target_n:, :]

        # N < target_n: pad with mean token to keep graph differentiable.
        pad_n = target_n - N
        mean_token = tokens.mean(dim=1, keepdim=True)
        pad_tokens = mean_token.expand(B, pad_n, D)
        return torch.cat([tokens, pad_tokens], dim=1)

    def _align_pred_gt_tokens(self, pred_tokens, gt_tokens, target_n=None):
        if target_n is not None:
            pred_tokens = self._align_tokens_to_target(pred_tokens, target_n)
            gt_tokens = self._align_tokens_to_target(gt_tokens, target_n)
        else:
            common_n = min(pred_tokens.shape[1], gt_tokens.shape[1])
            pred_tokens = pred_tokens[:, :common_n, :]
            gt_tokens = gt_tokens[:, :common_n, :]
        return pred_tokens, gt_tokens

    def _charbonnier_loss(self, pred, target, eps=1e-3):
        diff = pred - target
        return torch.sqrt(diff * diff + eps * eps).mean()

    def _to_unit_range(self, x):
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    def _synchronized_random_tile(self, pred, target, tile_size):
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"pred and target must have the same spatial size, got {pred.shape[-2:]} and {target.shape[-2:]}.")
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ValueError("dino_tile_size must be positive.")

        _, _, height, width = pred.shape
        if height < tile_size or width < tile_size:
            new_height = max(height, tile_size)
            new_width = max(width, tile_size)
            pred = F.interpolate(pred, size=(new_height, new_width), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(new_height, new_width), mode="bilinear", align_corners=False)
            height, width = new_height, new_width

        max_top = height - tile_size
        max_left = width - tile_size
        pred_tiles = []
        target_tiles = []
        for batch_idx in range(pred.shape[0]):
            top = torch.randint(0, max_top + 1, (1,), device=pred.device).item() if max_top > 0 else 0
            left = torch.randint(0, max_left + 1, (1,), device=pred.device).item() if max_left > 0 else 0
            pred_tiles.append(pred[batch_idx : batch_idx + 1, :, top : top + tile_size, left : left + tile_size])
            target_tiles.append(target[batch_idx : batch_idx + 1, :, top : top + tile_size, left : left + tile_size])

        return torch.cat(pred_tiles, dim=0), torch.cat(target_tiles, dim=0)

    def _random_tile_dino_perceptual_loss(self):
        if self.dino_encoder is None:
            raise RuntimeError("DINO encoder is required for char_ssim_dino_tile loss.")
        pred_tile, gt_tile = self._synchronized_random_tile(
            self.fake_B,
            self.real_B,
            self.opt.dino_tile_size,
        )
        pred_layers = self.dino_encoder.forward_intermediate_layers(pred_tile, self.dino_tile_layers)
        with torch.no_grad():
            gt_layers = self.dino_encoder.forward_intermediate_layers(gt_tile, self.dino_tile_layers)
        return multi_scale_dino_perceptual_loss(
            pred_layers,
            gt_layers,
            layer_weights=self.dino_tile_layer_weights,
        )

    def _random_tile_tara_losses(self):
        if self.dino_encoder is None:
            raise RuntimeError("DINO encoder is required for gan_l1_ev_dino_tile_topo loss.")

        pred_tile, gt_tile = self._synchronized_random_tile(
            self.fake_B,
            self.real_B,
            self.opt.tara_tile_size,
        )
        pred_layers = self.dino_encoder.forward_intermediate_layers(pred_tile, self.tara_dino_layers)
        with torch.no_grad():
            gt_layers = self.dino_encoder.forward_intermediate_layers(gt_tile, self.tara_dino_layers)

        expected_layers = len(self.tara_dino_layers)
        if len(pred_layers) < expected_layers or len(gt_layers) < expected_layers:
            raise RuntimeError(
                f"DINO returned {len(pred_layers)} pred layers and {len(gt_layers)} GT layers; "
                f"expected {expected_layers} layers {self.tara_dino_layers}."
            )

        dino_input_size = getattr(self.dino_encoder, "input_size", None) or self.opt.tara_tile_size
        dino_patch_size = getattr(self.dino_encoder, "patch_size", None) or self.opt.dino_patch_size
        target_n = (int(dino_input_size) // int(dino_patch_size)) ** 2

        pred_by_layer = {}
        gt_by_layer = {}
        for idx, layer_id in enumerate(self.tara_dino_layers):
            pred_tokens = pred_layers[idx]
            gt_tokens = gt_layers[idx].detach()
            pred_tokens, gt_tokens = self._align_pred_gt_tokens(pred_tokens, gt_tokens, target_n=target_n)
            pred_by_layer[layer_id] = pred_tokens
            gt_by_layer[layer_id] = gt_tokens

        tile_pred = [pred_by_layer[layer_id] for layer_id in self.tara_tile_layers]
        tile_gt = [gt_by_layer[layer_id] for layer_id in self.tara_tile_layers]
        tile_loss = dino_tile_cosine_loss(
            tile_pred,
            tile_gt,
            layer_weights=self.tara_tile_layer_weights,
        )

        topo_layer = int(self.opt.tara_topo_layer)
        topo_loss = tara_topological_kl_loss(
            pred_by_layer[topo_layer],
            gt_by_layer[topo_layer],
            tau=self.opt.tara_topo_tau,
        )
        return tile_loss, topo_loss

    def _tokens_to_grid(self, tokens):
        # tokens: [B, N, C] -> [B, C, H, W]
        B, N, C = tokens.shape
        side = int(round(N ** 0.5))
        if side * side != N:
            raise ValueError(f"Token count {N} is not a perfect square and cannot be reshaped into a grid.")
        return tokens.view(B, side, side, C).permute(0, 3, 1, 2).contiguous()

    def _rbf_kernel(self, x, y, sigma, eps):
        # x: [B, N, C], y: [B, M, C]
        dist2 = torch.cdist(x, y, p=2) ** 2
        denom = 2.0 * max(float(sigma) ** 2, eps)
        return torch.exp(-dist2 / denom)

    def _simple_mmd(self, x, y, sigma, eps):
        k_xx = self._rbf_kernel(x, x, sigma=sigma, eps=eps)
        k_yy = self._rbf_kernel(y, y, sigma=sigma, eps=eps)
        k_xy = self._rbf_kernel(x, y, sigma=sigma, eps=eps)
        mmd2 = k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2.0 * k_xy.mean(dim=(1, 2))
        return torch.sqrt(torch.clamp(mmd2, min=eps)).mean()

    def _compute_ms_mmd_loss(self, pred_layers, gt_layers):
        if self.ms_mmd_adapters is None:
            return torch.tensor(0.0, device=self.device)

        layer_losses = []
        eps = float(self.opt.ms_mmd_eps)
        sigma = float(self.opt.ms_mmd_sigma)

        for layer_id, layer_weight, pred_tokens, gt_tokens in zip(
            self.ms_mmd_layers,
            self.ms_mmd_layer_weights,
            pred_layers,
            gt_layers,
        ):
            pred_tokens, gt_tokens = self._align_pred_gt_tokens(pred_tokens, gt_tokens)
            pred_grid = self._tokens_to_grid(pred_tokens)
            gt_grid = self._tokens_to_grid(gt_tokens)

            adapter = self.ms_mmd_adapters[str(layer_id)]
            z_pred = adapter(pred_grid)
            z_gt = adapter(gt_grid)

            B, C, H, W = z_pred.shape
            z_pred = z_pred.permute(0, 2, 3, 1).reshape(B, H * W, C)
            z_gt = z_gt.permute(0, 2, 3, 1).reshape(B, H * W, C)

            mmd_l = self._simple_mmd(z_pred, z_gt, sigma=sigma, eps=eps)
            layer_losses.append(layer_weight * mmd_l)

        if not layer_losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(layer_losses).sum()

    def _cosine_align_loss(self, feat_pred, feat_gt, eps=1e-8):
        return 1.0 - F.cosine_similarity(feat_pred, feat_gt, dim=1, eps=eps).mean()

    def _l1_align_loss(self, feat_pred, feat_gt):
        return F.l1_loss(feat_pred, feat_gt)

    def _infer_sam_model_type(self, checkpoint_path, explicit_type=None):
        if explicit_type:
            return explicit_type
        ckpt = (checkpoint_path or "").lower()
        if "vit_h" in ckpt or "vit-h" in ckpt:
            return "vit_h"
        if "vit_l" in ckpt or "vit-l" in ckpt:
            return "vit_l"
        if "vit_b" in ckpt or "vit-b" in ckpt:
            return "vit_b"
        return "vit_b"

    def _extract_sam_align_features(self, image_256):
        if self.sam_model is None:
            raise RuntimeError("SAM model is not initialized for sam_align loss.")
        image_size = int(getattr(self.opt, "sam_align_size", 1024))
        # Convert training tensor from [-1, 1] back to image-like [0, 255].
        image_256 = image_256.detach().float()
        image_256 = (image_256 + 1.0) * 0.5
        image_256 = image_256.clamp(0.0, 1.0) * 255.0
        image_256 = F.interpolate(image_256, size=(image_size, image_size), mode="bilinear", align_corners=False)
        if hasattr(self.sam_model, "preprocess") and callable(getattr(self.sam_model, "preprocess")):
            image_256 = self.sam_model.preprocess(image_256)
        image_encoder = self.sam_model.image_encoder

        captured = {}
        handles = []
        layer_cfg = [int(x) for x in str(getattr(self.opt, "sam_align_layers", "4,8,12")).split(",") if x.strip()]
        target_layers = set(layer_cfg) if layer_cfg else {4, 8, 12}

        def _make_hook(layer_idx):
            def _hook(_module, _inputs, output):
                captured[layer_idx] = output
            return _hook

        num_blocks = len(image_encoder.blocks)
        for layer_idx in target_layers:
            if 1 <= layer_idx <= num_blocks:
                handles.append(image_encoder.blocks[layer_idx - 1].register_forward_hook(_make_hook(layer_idx)))

        try:
            with torch.no_grad():
                _ = image_encoder(image_256)
        finally:
            for handle in handles:
                handle.remove()

        def _to_bchw(feat, layer_idx):
            if feat.dim() == 4:
                # Expected layout from the hooked SAM blocks is BHWC.
                # Convert BHWC -> BCHW directly. If the tensor is already BCHW,
                # keep it as-is.
                if feat.shape[-1] > feat.shape[1] and feat.shape[-1] > feat.shape[2]:
                    return feat.permute(0, 3, 1, 2).contiguous()
                if feat.shape[1] > feat.shape[2] and feat.shape[1] > feat.shape[-1]:
                    return feat.contiguous()
                raise RuntimeError(
                    f"Unexpected 4D SAM feature layout at layer {layer_idx}: {tuple(feat.shape)}. "
                    "Expected BHWC or BCHW."
                )
            if feat.dim() == 3:
                B, N, C = feat.shape
                side = int(round(N ** 0.5))
                if side * side != N:
                    raise RuntimeError(f"SAM token count {N} at layer {layer_idx} is not a square number.")
                return feat.transpose(1, 2).contiguous().view(B, C, side, side)
            raise RuntimeError(f"Unexpected SAM feature rank at layer {layer_idx}: {feat.dim()}")

        features = {}
        for idx in target_layers:
            feat = captured.get(idx)
            if feat is None:
                continue
            features[idx] = _to_bchw(feat, idx)
        return features

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        if self.netD_mode == "oasis":
            self.loss_G_GAN = self._oasis_loss(pred_fake, True)
        elif self.netD_mode == "swin_mspatch":
            self.loss_G_GAN = self.netD.multiscale_gan_loss(pred_fake, True, self.criterionGAN)
        elif self.netD_mode == "tri_rf":
            self.loss_G_GAN = self._tri_rf_lsgan_loss(pred_fake, True)
        elif self.netD_mode == "multiscale":
            self.loss_G_GAN = self._multiscale_gan_loss(pred_fake, True)
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        if self.use_projected_D:
            self.loss_G_PROJECTED = self.criterionGAN(self.netD_projected(self.fake_B), True) * self.opt.lambda_projected_D
        else:
            self.loss_G_PROJECTED = torch.tensor(0.0, device=self.device)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # MoE hierarchical expert variance loss
        routing_maps = getattr(self.netG.model, "last_routing_maps", [])
        self.loss_G_EV_MACRO = (
            self._hierarchical_macro_ev_loss(routing_maps, self.opt.ev_eps)
            * self.opt.lambda_ev
            * self.opt.lambda_ev_macro
        )
        self.loss_G_EV_MICRO = (
            self._hierarchical_micro_ev_loss(routing_maps, self.opt.ev_eps)
            * self.opt.lambda_ev
            * self.opt.lambda_ev_micro
        )
        self.loss_G_EV = self.loss_G_EV_MACRO + self.loss_G_EV_MICRO

        # DINOv3 losses
        dino_pred = None
        dino_gt = None
        grid_size = None
        if self.dino_encoder is not None and self.opt.loss_type not in {"char_ssim_dino_tile", "physical_dino_bridge", "gan_l1_ev_dino_tile_topo"}:
            if self.opt.loss_type in {"dino_topo_micro", "ms_mmd", "wavelet", "wavelet_topo"}:
                dino_pred = self.dino_encoder(self.fake_B)
                with torch.no_grad():
                    dino_gt = self.dino_encoder(self.real_B)
            else:
                with torch.no_grad():
                    dino_pred = self.dino_encoder(self.fake_B)
                    dino_gt = self.dino_encoder(self.real_B)
            grid_size = self.dino_encoder.grid_size

        if self.opt.loss_type == "baseline":
            self.loss_G_DINO_INTER = relaxed_inter_image_nce(
                dino_pred,
                dino_gt,
                grid_size=grid_size,
                window=self.opt.dino_window,
                tau=self.opt.dino_tau,
                neg_samples=self.opt.dino_neg_samples,
            ) * self.opt.lambda_dino_inter
            self.loss_G_DINO_INTRA = intra_image_affinity_alignment(
                dino_pred,
                dino_gt,
                tau=self.opt.dino_tau,
                affinity_samples=self.opt.dino_affinity_samples,
            ) * self.opt.lambda_dino_intra

        if self.opt.loss_type == "dino_structure":
            self.loss_G_DINO_LGA = local_gram_alignment(
                dino_pred,
                dino_gt,
                grid_size=grid_size,
                window=self.opt.dino_lga_window,
            ) * self.opt.lambda_dino_lga
            self.loss_G_DINO_NSD = non_local_semantic_decoupling(
                dino_pred,
                grid_size=grid_size,
                margin=self.opt.dino_nsd_margin,
                num_pairs=self.opt.dino_nsd_pairs,
                min_distance=self.opt.dino_nsd_min_distance,
            ) * self.opt.lambda_dino_nsd
            self.loss_G_DINO_SAM = semantic_aware_masked_l1(
                dino_pred,
                dino_gt,
                self.fake_B,
                self.real_B,
                grid_size=grid_size,
                gamma=self.opt.dino_sam_gamma,
            ) * self.opt.lambda_dino_sam
        else:
            self.loss_G_DINO_LGA = torch.tensor(0.0, device=self.device)
            self.loss_G_DINO_NSD = torch.tensor(0.0, device=self.device)
            self.loss_G_DINO_SAM = torch.tensor(0.0, device=self.device)


        # combine loss and calculate gradients
        if self.opt.loss_type == "gan_l1":
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        elif self.opt.loss_type == "gan_l1_ev":
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_EV
        elif self.opt.loss_type == "gan_l1_ev_dino_tile_topo":
            tile_loss, topo_loss = self._random_tile_tara_losses()
            self.loss_G_DINO_TILE = tile_loss * self.opt.lambda_tara_tile
            self.loss_G_TARA_TOPO = topo_loss * self.opt.lambda_tara_topo
            # self.loss_G_WAVELET = self.wavelet_loss(self.fake_B, self.real_B) * self.opt.lambda_wavelet
            self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_L1
                + self.loss_G_EV
                + self.loss_G_DINO_TILE
                + self.loss_G_TARA_TOPO
                # + self.loss_G_WAVELET
            )
        elif self.opt.loss_type == "char_ssim_dino_tile":
            self.loss_G_L1 = torch.tensor(0.0, device=self.device)
            self.loss_G_CHAR = self._charbonnier_loss(
                self.fake_B,
                self.real_B,
                eps=self.opt.char_eps,
            ) * self.opt.lambda_char
            self.loss_G_SSIM = (
                1.0
                - pytorch_msssim_ssim(
                    self._to_unit_range(self.fake_B),
                    self._to_unit_range(self.real_B),
                    data_range=1.0,
                    size_average=True,
                )
            ) * self.opt.lambda_ssim
            self.loss_G_DINO_TILE = self._random_tile_dino_perceptual_loss() * self.opt.lambda_dino_tile
            self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_EV
                + self.loss_G_CHAR
                + self.loss_G_SSIM
                + self.loss_G_DINO_TILE
            )
       
            else:
                self.loss_G_SAM_RECON = torch.tensor(0.0, device=self.device)
                self.loss_G_SAM_RFC = torch.tensor(0.0, device=self.device)
                self.loss_G_SAM_BOUNDARY = torch.tensor(0.0, device=self.device)
            self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_HUBER
                + self.loss_G_EV
                + self.loss_G_SAM_RECON
                + self.loss_G_SAM_RFC
                + self.loss_G_SAM_BOUNDARY
            )
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G = (
                self.loss_G_GAN
                + self.loss_G_L1
                + self.loss_G_EV
                + self.loss_G_DINO_INTER
                + self.loss_G_DINO_INTRA
                + self.loss_G_DINO_MASKED_L1
                + self.loss_G_DINO_PERCEPTUAL
                + self.loss_G_DINO_LGA
                + self.loss_G_DINO_NSD
                + self.loss_G_DINO_SAM
                + self.loss_G_RSAC
                + self.loss_G_DINO_TOPO
                + self.loss_G_DINO_MICRO
                + self.loss_G_DINO_SCF
                + self.loss_G_DINO_CMPC
                + self.loss_G_MS_MMD
            )
        if self.use_projected_D:
            self.loss_G = self.loss_G + self.loss_G_PROJECTED
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        if self.use_projected_D:
            self._set_projected_D_requires_grad(True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        if self.use_projected_D:
            self._set_projected_D_requires_grad(False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
