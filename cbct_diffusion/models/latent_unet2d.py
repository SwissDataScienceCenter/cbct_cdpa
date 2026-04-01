"""LatentUnet2D: UNet2DModel wrapper with pixel-space compression.

This module wraps HuggingFace's ``UNet2DModel`` to support optional spatial
compression (block-based reshuffling) and automatic padding to the nearest
power-of-two resolution. The model operates on single-channel 2D images
(e.g., CT slices) and predicts single-channel outputs.

Key features:
- Pixel-shuffle compression controlled by the ``compression`` parameter.
- Automatic zero-padding of non-power-of-two inputs, with cropping on output.
- Accepts ``class_labels`` for slice-index conditioning via timestep embeddings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.utils import BaseOutput


@dataclass
class UNet2DOutput(BaseOutput):
    """Output container for :class:`LatentUnet2D`.

    Attributes:
        sample: Tensor of shape ``(B, C, H, W)`` — the model prediction.
    """
    sample: torch.Tensor


def compress_image(image_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Rearrange a ``(C, H, W)`` tensor into ``(C*bs^2, H//bs, W//bs)``."""
    C, H, W = image_tensor.shape
    blocks = image_tensor.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    blocks = blocks.permute(1, 2, 0, 3, 4).reshape(-1, C, block_size, block_size)
    num_blocks = H // block_size
    blocks = blocks.reshape(num_blocks, num_blocks, -1)
    return blocks.permute(2, 0, 1)


def decompress_image(blocks_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Inverse of :func:`compress_image`."""
    blocks_tensor = blocks_tensor.permute(1, 2, 0)
    num_blocks, _, _ = blocks_tensor.shape
    C = 1
    blocks_tensor = blocks_tensor.reshape(num_blocks, num_blocks, C, block_size, block_size)
    blocks_tensor = blocks_tensor.permute(2, 0, 3, 1, 4).reshape(
        C, num_blocks * block_size, num_blocks * block_size
    )
    return blocks_tensor


class LatentUnet2D(UNet2DModel):
    """UNet2DModel with optional pixel-shuffle compression.

    Parameters
    ----------
    compression : int, optional
        Spatial compression factor via pixel shuffling (default: 1 = no compression).
    input_channels : int, optional
        Number of input image channels (default: 1). When conditioning is used,
        set to 2 (noisy image + FDK prior).
    output_channels : int, optional
        Number of output channels (default: 1).
    sample_size : int or tuple, optional
        Target spatial resolution of the input images.
    **kwargs
        Additional arguments forwarded to ``UNet2DModel.__init__``.
    """

    def __init__(
        self,
        compression: int = 1,
        input_channels: int = 1,
        output_channels: int = 1,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__(
            sample_size=sample_size // compression,
            in_channels=compression ** 2 * input_channels,
            out_channels=compression ** 2 * output_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps,
        )
        self.compression = compression
        self.b_compress = torch.vmap(lambda x: compress_image(x, compression))
        self.b_decompress = torch.vmap(lambda x: decompress_image(x, compression))
        self.sample_size = sample_size

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        """Forward pass with automatic padding for non-power-of-two inputs.

        Parameters
        ----------
        sample : torch.Tensor
            Input tensor of shape ``(B, C, H, W)``.
        timestep : torch.Tensor or int
            Diffusion timestep or sparsity level.
        class_labels : torch.Tensor, optional
            Slice-index class labels for conditional generation.
        return_dict : bool
            Whether to return a dataclass (ignored; always returns a list).

        Returns
        -------
        list[torch.Tensor]
            Single-element list containing the output ``(B, C, H, W)`` tensor.
        """
        orig_h, orig_w = sample.shape[-2:]

        def _is_power_of_two(n: int) -> bool:
            return (n & (n - 1) == 0) and n != 0

        # Pad to nearest power of two if needed
        padded = False
        if not (_is_power_of_two(orig_h) and _is_power_of_two(orig_w)):
            target_side = 1 << (max(orig_h, orig_w) - 1).bit_length()
            pad_h = target_side - orig_h
            pad_w = target_side - orig_w
            if pad_h > 0 or pad_w > 0:
                sample = F.pad(sample, (0, pad_w, 0, pad_h))
                padded = True

        compressed_sample = self.b_compress(sample)
        output = super().forward(compressed_sample, timestep, class_labels, return_dict=False)
        decompressed = self.b_decompress(output[0])

        # Crop back to original size
        if padded:
            decompressed = decompressed[..., :orig_h, :orig_w]

        return [decompressed]
