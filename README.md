# CBCT Diffusion

Diffusion-based Cone-Beam CT (CBCT) reconstruction with sinogram-guided DDIM sampling. This library provides training and evaluation code for both **UNet** and **diffusion** models applied to sparse-view CBCT reconstruction across multiple anatomical datasets.

## Method overview

1. A 2D UNet denoiser is trained on individual axial slices to map noisy inputs to clean reconstructions.
2. During inference the denoiser is applied slice-by-slice across a 3D volume using a DDIM reverse-diffusion schedule.
3. At every DDIM timestep a **sinogram-consistency guidance** step enforces agreement with the measured projections via gradient-descent (GD) on the ASTRA forward model.
4. Multiple diffusion samples are averaged and optionally fine-tuned with a final GD pass.

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/<your-org>/cbct-diffusion.git
cd cbct-diffusion

# 2. Install the package
pip install -e .

# 3. Install astra-torch (CUDA-accelerated CT reconstruction)
pip install git+https://github.com/<your-org>/astra-torch.git
```

> **Note:** `astra-torch` requires a CUDA-capable GPU and the ASTRA Toolbox backend.

## Project structure

```
cbct-diffusion/
├── cbct_diffusion/
│   ├── models/             # LatentUnet2D, TomographicCBCTDiffusion, GuidanceConfig
│   ├── data/               # CBCTDataset, SliceCBCTDataset, Walnut512
│   ├── schedulers/         # GuidedDDIMScheduler, DDIMPipeline
│   ├── utils/              # PSNR/SSIM metrics, clamp presets, external_slices
│   ├── training/           # PyTorchExperiment, train_unet, train_diffusion
│   └── inference/          # reconstruct_unet, reconstruct_diffusion
├── scripts/                # Shell scripts for batch reconstruction
├── checkpoints/            # Place trained model checkpoints here
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Data preparation

The library supports two dataset formats:

| Format | Resolution | Datasets | Description |
|--------|-----------|----------|-------------|
| **HuggingFace** | 256³ | Walnut, Dental, Spine | Pre-processed volumes with `transforms.json`, `gt_volume.nii.gz`, `proj.nii.gz` |
| **Raw TIFF** | 501³ | Walnut | Raw projection images + per-slice TIFF reconstructions |

Set the environment variable `DATA_DIR` to point to your dataset root:

```bash
# HuggingFace format (256³)
export DATA_DIR=/path/to/huggingface/walnut   # or dental, spine

# Raw Walnut format (501³)
export DATA_DIR=/path/to/walnut               # contains Train/ and Test/ subdirs
```

Set `CKPT_DIR` to the directory containing trained model checkpoints:

```bash
export CKPT_DIR=/path/to/checkpoints
```

## Training

### UNet (L2 reconstruction loss)

```bash
# Walnut 256³
python -m cbct_diffusion.training.train_unet \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Unet_Walnut_CBCT_256 --epochs 60 --batch_size 4 --wandb

# Dental 256³
python -m cbct_diffusion.training.train_unet \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Unet_Dental_CBCT_256 --epochs 60 --batch_size 4

# Spine 256³
python -m cbct_diffusion.training.train_unet \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Unet_Spine_CBCT_256 --epochs 60 --batch_size 4

# Walnut 501³ (full resolution, with slice-index embedding)
python -m cbct_diffusion.training.train_unet \
    --data_path $DATA_DIR --image_size 501 --use_slice_idx \
    --exp_name Unet_Walnut_CBCT_501 --epochs 60 --batch_size 1
```

### Diffusion (DDPM noise prediction)

```bash
# Walnut 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Diffusion_Walnut_CBCT_256_ft20 --epochs 60 --batch_size 8

# Walnut 256³ – conditional (FDK prior as 2nd channel)
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 --conditioning \
    --exp_name Diffusion_Walnut_CBCT_256_ft20_cond --epochs 60 --batch_size 8

# Walnut 501³ – unconditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 501 \
    --exp_name Diffusion_Walnut_CBCT_501 --epochs 60 --batch_size 4

# Walnut 501³ – conditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 501 --conditioning \
    --exp_name Diffusion_Walnut_CBCT_501_cond --epochs 60 --batch_size 4

# Dental 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Diffusion_Dental_CBCT_256_ft20 --epochs 60 --batch_size 8

# Dental 256³ – conditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 --conditioning \
    --exp_name Diffusion_Dental_CBCT_256_ft20_cond --epochs 60 --batch_size 8

# Spine 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 \
    --exp_name Diffusion_Spine_CBCT_256_ft20 --epochs 60 --batch_size 8

# Spine 256³ – conditional
python -m cbct_diffusion.training.train_diffusion \
    --data_path $DATA_DIR --image_size 256 --conditioning \
    --exp_name Diffusion_Spine_CBCT_256_ft20_cond --epochs 60 --batch_size 8
```

## Inference / reconstruction

### Single-volume evaluation

```bash
# UNet reconstruction (Walnut 256³, CBCT ID 1, 20 views)
python -m cbct_diffusion.inference.reconstruct_unet \
    --data_path $DATA_DIR --cbct_id 1 --nviews 20 \
    --unet_checkpoint $CKPT_DIR/Unet_Walnut_CBCT_256.pt

# Diffusion reconstruction – conditional
python -m cbct_diffusion.inference.reconstruct_diffusion \
    --data_path $DATA_DIR --cbct_id 1 --nviews 20 \
    --diffusion_checkpoint $CKPT_DIR/Diffusion_Walnut_CBCT_256_ft20_cond.pt \
    --conditioning --guidance_lr 5e-4 --guidance_max_epochs 5

# Diffusion reconstruction – unconditional
python -m cbct_diffusion.inference.reconstruct_diffusion \
    --data_path $DATA_DIR --cbct_id 1 --nviews 20 \
    --diffusion_checkpoint $CKPT_DIR/Diffusion_Walnut_CBCT_256_ft20.pt \
    --guidance_lr 2e-3 --guidance_max_epochs 20
```

### Batch reconstruction scripts

Set `DATA_DIR` and `CKPT_DIR`, then run:

```bash
# Walnut 256³ (IDs 0–4)
bash scripts/reconstruct_walnut_256.sh

# Walnut 501³ (IDs 0–4, multiple view counts)
bash scripts/reconstruct_walnut_501.sh --type cond-diffusion

# Dental 256³ (IDs 0–19)
bash scripts/reconstruct_dental.sh

# Spine 256³ (IDs 0–19)
bash scripts/reconstruct_spine.sh
```

## Metrics

The library evaluates reconstructions using:

- **PSNR** – computed after clamping to dataset-specific ranges and normalising to [0, 1].
- **3D SSIM** – averaged across depth, height, and width planes. Parallelised via `joblib` for large volumes.

Dataset-specific clamp ranges:

| Dataset | Min | Max |
|---------|-----|-----|
| Walnut  | 0.0 | 0.084 |
| Dental  | 0.0 | 0.09009 |
| Spine   | 0.0 | 0.051744 |

## Key components

| Module | Description |
|--------|-------------|
| `LatentUnet2D` | Wraps HuggingFace `UNet2DModel` with pixel-shuffle compression and auto-padding to power-of-2 sizes |
| `TomographicCBCTDiffusion` | Applies 2D UNet slice-wise on 3D volumes with sinogram GD guidance |
| `GuidedDDIMScheduler` | DDIM scheduler that injects a guidance function between the x₀ prediction and the direction-pointing step |
| `DDIMPipeline` | Full DDIM reverse-diffusion loop with optional FDK conditioning channel |
| `GuidanceConfig` | Dataclass bundling all GD guidance hyper-parameters |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
