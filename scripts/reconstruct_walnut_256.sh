#!/bin/bash
# Walnut 256³ reconstructions (UNet + diffusion + cond-diffusion)
# Usage: bash scripts/reconstruct_walnut_256.sh
#
# Set DATA_DIR and CKPT_DIR to point to your data and checkpoints.
# DATA_DIR should contain the HuggingFace-format walnut dataset.

set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the HuggingFace walnut data directory}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

echo "Starting Walnut 256³ reconstructions..."

for id in {0..4}; do
    echo "========================================"
    echo "CBCT ID: $id"
    echo "========================================"

    # Conditional diffusion
    python -m cbct_diffusion.inference.reconstruct_diffusion \
        --cbct_id=$id --nviews 20 \
        --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Walnut_CBCT_256_ft20_cond.pt" \
        --conditioning \
        --data_path="${DATA_DIR}" \
        --guidance_lr 5e-4 --guidance_max_epochs 5

    # Unconditional diffusion
    python -m cbct_diffusion.inference.reconstruct_diffusion \
        --cbct_id=$id --nviews 20 \
        --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Walnut_CBCT_256_ft20.pt" \
        --data_path="${DATA_DIR}" \
        --guidance_lr 2e-3 --guidance_max_epochs 20

    # UNet
    python -m cbct_diffusion.inference.reconstruct_unet \
        --cbct_id=$id --nviews 20 \
        --unet_checkpoint="${CKPT_DIR}/Unet_Walnut_CBCT_256.pt" \
        --data_path="${DATA_DIR}"
done

echo "All Walnut 256³ reconstructions completed!"
