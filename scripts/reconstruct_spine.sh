#!/bin/bash
# Spine 256³ reconstructions (UNet + diffusion + cond-diffusion)
# Usage: bash scripts/reconstruct_spine.sh
#
# Set DATA_DIR to the HuggingFace spine data directory.
# Set CKPT_DIR to the checkpoints directory.

set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the HuggingFace spine data directory}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

echo "Starting Spine 256³ reconstructions..."

for id in {0..19}; do
    echo "========================================"
    echo "CBCT ID: $id"
    echo "========================================"

    # Conditional diffusion
    python -m cbct_diffusion.inference.reconstruct_diffusion \
        --cbct_id=$id --nviews 20 \
        --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Spine_CBCT_256_ft20_cond.pt" \
        --conditioning \
        --data_path="${DATA_DIR}" \
        --guidance_lr 5e-4

    # Unconditional diffusion
    python -m cbct_diffusion.inference.reconstruct_diffusion \
        --cbct_id=$id --nviews 20 \
        --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Spine_CBCT_256_ft20.pt" \
        --data_path="${DATA_DIR}" \
        --guidance_lr 2e-3 --guidance_max_epochs 20

    # UNet
    python -m cbct_diffusion.inference.reconstruct_unet \
        --cbct_id=$id --nviews 20 \
        --unet_checkpoint="${CKPT_DIR}/Unet_Spine_CBCT_256.pt" \
        --data_path="${DATA_DIR}" \
        --gd_finetune_epochs 5 --gd_finetune_lr 5e-4
done

echo "All Spine 256³ reconstructions completed!"
