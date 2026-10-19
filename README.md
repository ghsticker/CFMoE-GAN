# CFMoE-GAN

This repository contains the PyTorch implementation of `CFMoE-GAN` for paired SAR-to-optical image translation. The codebase includes training, inference, and auxiliary analysis tools for experiments on SAR2Opt, UNICORN, and QXS–SAROPT.

## Overview

CFMoE-GAN learns a mapping from SAR images to corresponding optical images. The framework includes:

- a Swin-Unet generator with SCPE-SFT skip connections
- CF-MoE modules with four frequency transformation expert groups and intra-group Top-k routing
- a multi-scale discriminator
- joint adversarial, L1 reconstruction, and hierarchical expert variance losses
- local feature and topological consistency losses using a frozen DINOv3 encoder

The repository is intended as a research code release. Datasets, pretrained weights, and trained checkpoints are distributed separately from the source code.

## Repository Structure

```text
CFMoE-GAN/
|-- data/              dataset loading and paired image transforms
|-- models/            generators, discriminators, and loss integration
|-- options/           training and inference arguments
|-- util/              losses, logging, visualization, and data utilities
|-- plot/              supplementary analysis and figure scripts
|-- docs/              supporting documentation
|-- train.py           training entrypoint
|-- test.py            inference entrypoint
|-- environment.yml    base environment and project dependencies
|-- LICENSE            upstream license notices
`-- README.md
```

Main components:

- `models/pix2pix_model.py`: model setup, optimization, and training objectives
- `models/networks.py`: generator and discriminator construction
- `models/swin_transformer_unet_skip_expand_decoder_sys.py`: Swin-Unet, SCPE-SFT, and CF-MoE modules
- `models/dino_encoder.py`: frozen DINOv3 feature extraction
- `data/paired_dataset.py`: filename-based SAR and optical image pairing
- `options/base_options.py`: shared model and dataset settings
- `options/train_options.py`: learning rates, training schedule, and checkpoint settings

## Data

### Download

The experiments use three datasets. Download links are provided below:

| Dataset    | Baidu Netdisk                                                | Extraction Code |
| ---------- | ------------------------------------------------------------ | --------------- |
| SAR2Opt    | [Download](https://pan.baidu.com/s/1dWD9fcDeoaYCu3rhJ6BY9g?pwd=gtj2) | `gtj2`          |
| UNICORN    | [Download](https://pan.baidu.com/s/162MRgbDTxk7zwXClZCaSLQ?pwd=faus) | `faus`          |
| QXS–SAROPT | [Download](https://pan.baidu.com/s/1lsckKFpdhrfsNkbAkiBhbQ?pwd=bu4n) | `bu4n`          |

Please follow the original datasets' usage terms and cite their respective publications when using the data.

### Directory Organization

Each dataset should be organized as paired SAR and optical image folders:

```text
DATA_ROOT/
|-- trainA/    training SAR images
|-- trainB/    training optical images
|-- testA/     test SAR images
`-- testB/     test optical images
```

Use `--dataset_mode paired --direction AtoB`, where domain A contains SAR images and domain B contains optical images. Paired images must be spatially aligned and have identical filenames, including their extensions, such as `trainA/0001.png` and `trainB/0001.png`.

The loader converts images to RGB and applies synchronized transforms to each pair. The default training preprocessing resizes images to `286 × 286` and randomly crops `256 × 256` patches. Original complex-valued SAR data and multispectral bands require separate preprocessing.

## Training

### Pretrained Weights

Before training, obtain the Swin-T initialization checkpoint from [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet#1-download-pre-trained-swin-transformer-model-swin-t) and the SAT-493M ViT-L/16 checkpoint from [DINOv3](https://github.com/facebookresearch/dinov3#pretrained-models). Place them in:

```text
pretrained/
|-- swin_tiny_patch4_window7_224.pth
`-- dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

Prepare the DINOv3 source repository for local Torch Hub loading:

```bash
mkdir -p pretrained third_party
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

### Training

The following configuration follows the original `scripts/ablation/train_ablation_06_full.sh`. Update the dataset and weight paths, then run from the repository root:

```bash
python train.py \
    --dataroot ./datasets/sar2opt \
    --name train_CFMoE_GAN \
    --model pix2pix \
    --netG swin_unet \
    --netD multiscale \
    --swin_pretrained_path ./pretrained/swin_tiny_patch4_window7_224.pth \
    --direction AtoB \
    --dataset_mode paired \
    --norm batch \
    --pool_size 0 \
    --dino_repo ./third_party/dinov3 \
    --dino_model_name dinov3_vitl16 \
    --dino_pretrained_path ./pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
    --loss_type gan_l1_ev_dino_tile_topo \
    --n_epochs 200 \
    --n_epochs_decay 100 \
    --batch_size 4 \
    --lambda_tara_tile 1.0 \
    --lambda_tara_topo 1.0
```
## Inference

After training, load the generator checkpoint using the same experiment name:

```bash
python test.py \
    --dataroot ./datasets/sar2opt \
    --name train_CFMoE_GAN \
    --model pix2pix \
    --netG swin_unet \
    --dataset_mode paired \
    --direction AtoB \
    --norm batch \
    --load_size 256 \
    --crop_size 256 \
    --loss_type gan_l1 \
    --epoch latest \
    --eval
```

For the full model, `--loss_type gan_l1` preserves the generator architecture while avoiding initialization of the DINOv3 loss encoder during inference. Only the trained generator checkpoint is required. Match any architecture-related options, including expert counts and module ablations, to those used during training.


The paired loader requires both `testA/` and `testB/`. Inference uses batch size 1 with flipping disabled and processes up to 10,000 pairs by default; adjust `--num_test` as needed.

## Environment

The base environment is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate CFMoE-GAN
```

The environment specifies Python 3.11, PyTorch 2.4.0, torchvision 0.19.0, and CUDA 12.1. Main dependencies include NumPy, Pillow, timm, einops, OpenCV, TensorBoard, dominate, pytorch-msssim, scikit-image, and wandb. W&B logging is enabled only when `--use_wandb` is specified.

DINOv3 is loaded from a separate source checkout. Check compatibility between that revision and your PyTorch environment using the [official DINOv3 installation instructions](https://github.com/facebookresearch/dinov3#installation). The supplied environment file is a base configuration rather than a complete, locked environment for every DINOv3 revision.


## Acknowledgements

This project builds on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and incorporates components based on [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) and [DINOv3](https://github.com/facebookresearch/dinov3).

Upstream copyright and license notices are retained in [LICENSE](LICENSE). Third-party code, pretrained weights, and datasets remain subject to their respective licenses and usage terms.
