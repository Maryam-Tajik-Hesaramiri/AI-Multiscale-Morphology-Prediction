#!/usr/bin/env python
"""
unet.py
================

U-Net for binary ZnO semantic segmentation from grayscale SEM images.

Design summary
--------------
- Input:
      Grayscale SEM patch, shape (B, 1, H, W)

- Output:
      One-channel raw logits, shape (B, 1, H, W)

- Encoder:
      Four resolution levels with max pooling.

- Decoder:
      Transposed-convolution upsampling and skip connections.

- Normalization:
      Group Normalization instead of Batch Normalization.
      This is more stable when training with small batch sizes.

- Activation:
      ReLU.

- Regularization:
      Dropout is applied only in the bottleneck.

- Output activation:
      No Sigmoid is included inside the model.
      Sigmoid will be applied only during metric calculation and inference.
      This keeps the model compatible with BCEWithLogitsLoss.

- CPU-friendly:
      base_channels=32 provides a practical balance between representation
      capacity and computational cost.

Usage
-----
Run the architecture smoke test:

    python unet.py

The script verifies:
- Input/output shape compatibility
- Forward propagation
- Backward propagation
- Number of trainable parameters
- Binary segmentation output contract

This public portfolio version preserves the architecture used in the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# 1. MODEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class UNetConfig:
    """Configuration for the binary segmentation U-Net."""

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    dropout_probability: float = 0.10
    groups: int = 8

    def validate(self) -> None:
        """Validate configuration before model construction."""
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if self.out_channels != 1:
            raise ValueError(
                "Binary segmentation requires out_channels=1."
            )

        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "dropout_probability must be in the interval [0, 1)."
            )

        if self.groups <= 0:
            raise ValueError("groups must be positive.")


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def valid_group_count(num_channels: int, requested_groups: int) -> int:
    """
    Select the largest valid GroupNorm group count.

    GroupNorm requires num_channels to be divisible by num_groups.
    """
    max_groups = min(num_channels, requested_groups)

    for group_count in range(max_groups, 0, -1):
        if num_channels % group_count == 0:
            return group_count

    return 1


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize trainable layers.

    - Convolution weights: Kaiming initialization for ReLU activations.
    - Convolution biases: zeros.
    - GroupNorm scale: ones.
    - GroupNorm bias: zeros.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def align_to_skip(
    decoder_tensor: Tensor,
    skip_tensor: Tensor,
) -> Tensor:
    """
    Align decoder spatial size to the corresponding skip tensor.

    Patch sizes divisible by 16 normally align exactly. This safeguard makes
    the model robust to other image dimensions by using bilinear interpolation
    only when a mismatch occurs.
    """
    if decoder_tensor.shape[-2:] != skip_tensor.shape[-2:]:
        decoder_tensor = F.interpolate(
            decoder_tensor,
            size=skip_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return decoder_tensor


# =============================================================================
# 3. BASIC U-NET BLOCKS
# =============================================================================

class DoubleConv(nn.Module):
    """
    Two consecutive convolution-normalization-activation operations.

    Spatial dimensions are preserved using kernel_size=3 and padding=1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
    ) -> None:
        super().__init__()

        group_count = valid_group_count(out_channels, groups)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=group_count,
                num_channels=out_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=group_count,
                num_channels=out_channels,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    One encoder stage.

    Returns both:
    - skip feature map before pooling
    - pooled feature map for the next encoder stage
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
    ) -> None:
        super().__init__()

        self.features = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        skip = self.features(x)
        pooled = self.pool(skip)

        return skip, pooled


class DecoderBlock(nn.Module):
    """
    One decoder stage.

    The lower-resolution tensor is upsampled, concatenated with the matching
    encoder skip tensor, and refined by a DoubleConv block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        self.features = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            groups=groups,
        )

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
    ) -> Tensor:
        x = self.upsample(x)
        x = align_to_skip(x, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.features(x)

        return x


# =============================================================================
# 4. COMPLETE U-NET
# =============================================================================

class UNet(nn.Module):
    """
    Four-level U-Net for binary semantic segmentation.

    Channel progression with base_channels=32:
        Encoder:   32 -> 64 -> 128 -> 256
        Bottleneck: 512
        Decoder:   256 -> 128 -> 64 -> 32
        Output:    1 logit channel
    """

    def __init__(
        self,
        config: UNetConfig = UNetConfig(),
    ) -> None:
        super().__init__()

        config.validate()
        self.config = config

        c1 = config.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        # Encoder pathway.
        self.encoder1 = EncoderBlock(
            config.in_channels,
            c1,
            config.groups,
        )
        self.encoder2 = EncoderBlock(
            c1,
            c2,
            config.groups,
        )
        self.encoder3 = EncoderBlock(
            c2,
            c3,
            config.groups,
        )
        self.encoder4 = EncoderBlock(
            c3,
            c4,
            config.groups,
        )

        # Lowest-resolution representation.
        self.bottleneck = nn.Sequential(
            DoubleConv(
                in_channels=c4,
                out_channels=c5,
                groups=config.groups,
            ),
            nn.Dropout2d(
                p=config.dropout_probability,
            ),
        )

        # Decoder pathway.
        self.decoder4 = DecoderBlock(
            in_channels=c5,
            skip_channels=c4,
            out_channels=c4,
            groups=config.groups,
        )
        self.decoder3 = DecoderBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            groups=config.groups,
        )
        self.decoder2 = DecoderBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            groups=config.groups,
        )
        self.decoder1 = DecoderBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            groups=config.groups,
        )

        # Raw binary logits. No sigmoid here.
        self.output_head = nn.Conv2d(
            in_channels=c1,
            out_channels=config.out_channels,
            kernel_size=1,
        )

        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform one forward pass.

        Parameters
        ----------
        x:
            Tensor of shape (B, 1, H, W).

        Returns
        -------
        Tensor:
            Raw logits of shape (B, 1, H, W).
        """
        self._validate_input(x)

        skip1, pooled1 = self.encoder1(x)
        skip2, pooled2 = self.encoder2(pooled1)
        skip3, pooled3 = self.encoder3(pooled2)
        skip4, pooled4 = self.encoder4(pooled3)

        bottleneck = self.bottleneck(pooled4)

        decoded4 = self.decoder4(bottleneck, skip4)
        decoded3 = self.decoder3(decoded4, skip3)
        decoded2 = self.decoder2(decoded3, skip2)
        decoded1 = self.decoder1(decoded2, skip1)

        logits = self.output_head(decoded1)

        # Final safeguard for arbitrary input dimensions.
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def _validate_input(self, x: Tensor) -> None:
        """Check the expected input tensor contract."""
        if x.ndim != 4:
            raise ValueError(
                "U-Net input must be a 4D BCHW tensor. "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channel(s), "
                f"received {x.shape[1]}."
            )

        minimum_size = 16

        if x.shape[-2] < minimum_size or x.shape[-1] < minimum_size:
            raise ValueError(
                "Input height and width must each be at least 16 pixels."
            )


# =============================================================================
# 5. MODEL REPORTING UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def model_size_megabytes(model: nn.Module) -> float:
    """
    Estimate parameter memory in megabytes using current parameter dtypes.

    This excludes optimizer state, gradients, and intermediate activations.
    """
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    total_bytes += sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    return total_bytes / (1024**2)


def build_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    dropout_probability: float = 0.10,
    groups: int = 8,
) -> UNet:
    """
    Factory function used by training and evaluation scripts.
    """
    config = UNetConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
        groups=groups,
    )

    return UNet(config=config)


# =============================================================================
# 6. ARCHITECTURE SMOKE TEST
# =============================================================================

def run_smoke_test() -> None:
    """
    Verify forward and backward propagation on the available device.
    """
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_unet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        dropout_probability=0.10,
        groups=8,
    ).to(device)

    batch_size = 2
    patch_height = 256
    patch_width = 256

    sample_input = torch.randn(
        batch_size,
        1,
        patch_height,
        patch_width,
        device=device,
    )

    # Forward pass.
    logits = model(sample_input)

    expected_shape = (
        batch_size,
        1,
        patch_height,
        patch_width,
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(logits.shape)}. "
            f"Expected: {expected_shape}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError("Model output contains NaN or Inf.")

    # Simple backward-pass integrity test.
    dummy_target = torch.randint(
        low=0,
        high=2,
        size=expected_shape,
        device=device,
    ).float()

    loss = F.binary_cross_entropy_with_logits(
        logits,
        dummy_target,
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if not gradients_exist:
        raise RuntimeError(
            "Backward test failed: no parameter gradients were created."
        )

    total_parameters, trainable_parameters = count_parameters(model)

    print("=" * 72)
    print("U-NET ARCHITECTURE SUMMARY")
    print("=" * 72)
    print(f"Device:                     {device}")
    print(f"Input shape:                {tuple(sample_input.shape)}")
    print(f"Output shape:               {tuple(logits.shape)}")
    print(f"Input channels:             {model.config.in_channels}")
    print(f"Output logit channels:      {model.config.out_channels}")
    print(f"Base channels:              {model.config.base_channels}")
    print(f"Normalization:              GroupNorm")
    print(f"Dropout probability:        {model.config.dropout_probability}")
    print(f"Total parameters:           {total_parameters:,}")
    print(f"Trainable parameters:       {trainable_parameters:,}")
    print(f"Parameter memory:           {model_size_megabytes(model):.2f} MB")
    print(f"Test BCE-with-logits loss:  {loss.item():.6f}")
    print("Output activation:          None (raw logits)")
    print("=" * 72)
    print("\nForward and backward smoke tests completed successfully.")


def main() -> None:
    """Run the standalone model validation."""
    run_smoke_test()


if __name__ == "__main__":
    main()
