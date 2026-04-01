#!/bin/bash
# Walnut 501³ (high-resolution) reconstructions across multiple view counts.
# Usage: bash scripts/reconstruct_walnut_501.sh --type <unet|diffusion|cond-diffusion>
#
# Set DATA_DIR to the raw Walnut data root (contains Train/ and Test/).
# Set CKPT_DIR to the checkpoints directory.

set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the raw Walnut data root}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

# Parse --type argument
TYPE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift 2 ;;
        *) echo "Unknown option $1"; echo "Usage: $0 --type <unet|diffusion|cond-diffusion>"; exit 1 ;;
    esac
done

if [[ "$TYPE" != "unet" && "$TYPE" != "diffusion" && "$TYPE" != "cond-diffusion" ]]; then
    echo "Error: --type must be one of: unet, diffusion, cond-diffusion"
    exit 1
fi

echo "Starting Walnut 501³ $TYPE reconstructions..."

for id in {0..4}; do
    for nviews in 20 40 60 80 100 120 140 160 180; do
        echo "--- CBCT $id | $nviews views | $TYPE ---"

        case $TYPE in
            "unet")
                python -m cbct_diffusion.inference.reconstruct_unet \
                    --cbct_id=$id --nviews=$nviews --high_resolution \
                    --unet_checkpoint="${CKPT_DIR}/Unet_Walnut_CBCT_501.pt" \
                    --data_path="${DATA_DIR}" \
                    --wandb_project=cbct-reconstructions-501 \
                    --enable_gd_zero --enable_gd_fdk
                ;;
            "diffusion")
                python -m cbct_diffusion.inference.reconstruct_diffusion \
                    --cbct_id=$id --nviews=$nviews --high_resolution \
                    --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Walnut_CBCT_501.pt" \
                    --data_path="${DATA_DIR}" \
                    --wandb_project=cbct-reconstructions-501 \
                    --guidance_lr 5e-3 --guidance_max_epochs 10
                ;;
            "cond-diffusion")
                python -m cbct_diffusion.inference.reconstruct_diffusion \
                    --cbct_id=$id --nviews=$nviews --high_resolution \
                    --diffusion_checkpoint="${CKPT_DIR}/Diffusion_Walnut_CBCT_501_cond.pt" \
                    --conditioning \
                    --data_path="${DATA_DIR}" \
                    --wandb_project=cbct-reconstructions-501
                ;;
        esac
    done
done

echo "All Walnut 501³ $TYPE reconstructions completed!"
