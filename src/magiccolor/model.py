from math import sqrt, floor, ceil

from typing import Self

from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    Sequential,
    Linear,
    Conv2d,
    Sigmoid,
    SiLU,
    PixelShuffle,
    AdaptiveAvgPool2d,
    Flatten,
    Parameter,
)

from torch.nn.utils.parametrize import (
    register_parametrization,
    is_parametrized,
    remove_parametrizations,
)

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class MagicColor(Module, PyTorchModelHubMixin):
    """
    A deep learning model for colorizing grayscale images. Magic Color employs a
    residual U-Net architecture with four encoding stages and four decoding stages.

    Each stage consists of multiple encoder/decoder blocks with spatial attention and
    inverted bottleneck layers. Down and upsampling of the feature maps is performed
    using pure convolutions.
    """

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 1, "Number of primary layers must be greater than 1."

        assert (
            secondary_layers > 1
        ), "Number of secondary layers must be greater than 1."

        assert tertiary_layers > 1, "Number of tertiary layers must be greater than 1."

        assert (
            quaternary_layers > 1
        ), "Number of quaternary layers must be greater than 1."

        self.encoder = Encoder(
            primary_channels,
            ceil(primary_layers / 2),
            secondary_channels,
            ceil(secondary_layers / 2),
            tertiary_channels,
            ceil(tertiary_layers / 2),
            quaternary_channels,
            ceil(quaternary_layers / 2),
            hidden_ratio,
        )

        self.decoder = Decoder(
            quaternary_channels,
            floor(quaternary_layers / 2),
            tertiary_channels,
            floor(tertiary_layers / 2),
            secondary_channels,
            floor(secondary_layers / 2),
            primary_channels,
            floor(primary_layers / 2),
            hidden_ratio,
        )

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def freeze_parameters(self) -> None:
        """Freeze all model parameters to prevent them from being updated during training."""

        for param in self.parameters():
            param.requires_grad = False

    def add_weight_norms(self) -> None:
        """Add weight normalization parameterization to the network."""

        self.encoder.add_weight_norms()
        self.decoder.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Add LoRA adapters to all convolutional layers in the network."""

        self.encoder.add_lora_adapters(rank, alpha)
        self.decoder.add_lora_adapters(rank, alpha)

    def remove_parameterizations(self) -> None:
        """Remove all network parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder and decoder block.
        """

        self.encoder.enable_activation_checkpointing()
        self.decoder.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        """
        Predict the color channels of a grayscale image.

        Args:
            x: Input image tensor of shape (B, 1, H, W) in Lab color format.

        Returns:
            z: The A and B color channels of the image.
        """

        z1, z2, z3, z4 = self.encoder.forward(x)
        z = self.decoder.forward(z4, z3, z2, z1)

        return z

    @torch.inference_mode()
    def colorize(self, x: Tensor) -> Tensor:
        """
        Convenience method for inference.

        Args:
            x: Input image tensor of shape (B, 1, H, W).
        """

        x = torch.clamp(x, 0, 100)

        z = self.forward(x)

        z = torch.clamp(z, -128, 127)

        z = torch.cat([x, z], dim=1)

        return z


class ONNXModel(Module):
    """A wrapper class for exporting to ONNX format."""

    def __init__(self, model: MagicColor):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 1, H, W).
        """

        return self.model.colorize(x)


class Encoder(Module):
    """An encoder subnetwork employing a deep stack of encoder blocks."""

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stem = Conv2d(1, primary_channels, kernel_size=1)

        self.stage1 = Sequential(
            *[
                EncoderBlock(primary_channels, hidden_ratio)
                for _ in range(primary_layers)
            ]
        )

        self.stage2 = Sequential(
            *[
                EncoderBlock(secondary_channels, hidden_ratio)
                for _ in range(secondary_layers)
            ]
        )

        self.stage3 = Sequential(
            *[
                EncoderBlock(tertiary_channels, hidden_ratio)
                for _ in range(tertiary_layers)
            ]
        )

        self.stage4 = Sequential(
            *[
                EncoderBlock(quaternary_channels, hidden_ratio)
                for _ in range(quaternary_layers)
            ]
        )

        self.downsample1 = PixelCrush(primary_channels, secondary_channels, 2)
        self.downsample2 = PixelCrush(secondary_channels, tertiary_channels, 2)
        self.downsample3 = PixelCrush(tertiary_channels, quaternary_channels, 2)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def add_weight_norms(self) -> None:
        self.stem = weight_norm(self.stem)

        for layer in self.stage1:
            layer.add_weight_norms()

        for layer in self.stage2:
            layer.add_weight_norms()

        for layer in self.stage3:
            layer.add_weight_norms()

        for layer in self.stage4:
            layer.add_weight_norms()

        self.downsample1.add_weight_norms()
        self.downsample2.add_weight_norms()
        self.downsample3.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for layer in self.stage1:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage2:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage3:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage4:
            layer.add_lora_adapters(rank, alpha)

        self.downsample1.add_lora_adapters(rank, alpha)
        self.downsample2.add_lora_adapters(rank, alpha)
        self.downsample3.add_lora_adapters(rank, alpha)

        register_parametrization(
            self.stem, "weight", ChannelLoRA(self.stem, rank, alpha)
        )

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.stem.forward(x)
        z1 = self.checkpoint(self.stage1, z1)

        z2 = self.downsample1.forward(z1)
        z2 = self.checkpoint(self.stage2, z2)

        z3 = self.downsample2.forward(z2)
        z3 = self.checkpoint(self.stage3, z3)

        z4 = self.downsample3.forward(z3)
        z4 = self.checkpoint(self.stage4, z4)

        return z1, z2, z3, z4


class EncoderBlock(Module):
    """A single encoder block consisting of two stages and a residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = InvertedBottleneck(num_channels, hidden_ratio)

    def add_weight_norms(self) -> None:
        self.stage1.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        self.stage1.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stage1.forward(x)

        z = x + z  # Local residual connection

        return z


class InvertedBottleneck(Module):
    """A wide non-linear activation block with 3x3 convolutions."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.silu = SiLU()

    def add_weight_norms(self) -> None:
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv1,
            "weight",
            ChannelLoRA(self.conv1, rank, alpha),
        )

        register_parametrization(
            self.conv2,
            "weight",
            ChannelLoRA(self.conv2, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z


class PixelCrush(Module):
    """Downsample the feature maps using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, crush_factor: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert crush_factor in {
            2,
            3,
            4,
        }, "Crush factor must be either 2, 3, or 4."

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=crush_factor,
            stride=crush_factor,
            bias=False,
        )

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_spectral_norms(self) -> None:
        self.conv = spectral_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Decoder(Module):
    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stage1 = Sequential(
            *[
                DecoderBlock(primary_channels, hidden_ratio)
                for _ in range(primary_layers)
            ]
        )

        self.stage2 = Sequential(
            *[
                DecoderBlock(secondary_channels, hidden_ratio)
                for _ in range(secondary_layers)
            ]
        )

        self.stage3 = Sequential(
            *[
                DecoderBlock(tertiary_channels, hidden_ratio)
                for _ in range(tertiary_layers)
            ]
        )

        self.stage4 = Sequential(
            *[
                DecoderBlock(quaternary_channels, hidden_ratio)
                for _ in range(quaternary_layers)
            ]
        )

        self.upsample1 = SubpixelConv2d(primary_channels, secondary_channels, 2)
        self.upsample2 = SubpixelConv2d(secondary_channels, tertiary_channels, 2)
        self.upsample3 = SubpixelConv2d(tertiary_channels, quaternary_channels, 2)

        self.head = Conv2d(quaternary_channels, 2, kernel_size=1)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def add_weight_norms(self) -> None:
        for layer in self.stage1:
            layer.add_weight_norms()

        for layer in self.stage2:
            layer.add_weight_norms()

        for layer in self.stage3:
            layer.add_weight_norms()

        for layer in self.stage4:
            layer.add_weight_norms()

        self.upsample1.add_weight_norms()
        self.upsample2.add_weight_norms()
        self.upsample3.add_weight_norms()

        self.head = weight_norm(self.head)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for layer in self.stage1:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage2:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage3:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage4:
            layer.add_lora_adapters(rank, alpha)

        self.upsample1.add_lora_adapters(rank, alpha)
        self.upsample2.add_lora_adapters(rank, alpha)
        self.upsample3.add_lora_adapters(rank, alpha)

        register_parametrization(
            self.head, "weight", ChannelLoRA(self.head, rank, alpha)
        )

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> Tensor:
        z = self.checkpoint(self.stage1, x1)
        z = self.upsample1.forward(z)

        z = x2 + z  # Global residual connection

        z = self.checkpoint(self.stage2, z)
        z = self.upsample2.forward(z)

        z = x3 + z  # Global residual connection

        z = self.checkpoint(self.stage3, z)
        z = self.upsample3.forward(z)

        z = x4 + z  # Global residual connection

        z = self.checkpoint(self.stage4, z)
        z = self.head.forward(z)

        return z


class DecoderBlock(EncoderBlock):
    pass


class SubpixelConv2d(Module):
    """Upsample the feature maps using subpixel convolution."""

    def __init__(self, in_channels: int, out_channels: int, upscale_ratio: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert upscale_ratio in {
            2,
            3,
            4,
        }, "Upscale ratio must be either 2, 3, or 4."

        out_channels = out_channels * upscale_ratio**2

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # Effective stride will be 1 / upscale_ratio
            padding=1,
            bias=False,
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z


class ChannelLoRA(Module):
    """Low rank channel decomposition transformation."""

    def __init__(self, layer: Conv2d, rank: int, alpha: float):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        out_channels, in_channels, h, w = layer.weight.shape

        lora_a = torch.randn(h, w, out_channels, rank) / sqrt(rank)
        lora_b = torch.zeros(h, w, rank, in_channels)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.alpha = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_a @ self.lora_b

        z *= self.alpha

        # Move channels to front to match weight shape
        z = z.permute(2, 3, 0, 1)

        z = weight + z

        return z


class Bouncer(Module):
    """A critic network for detecting real and fake images for adversarial training."""

    AVAILABLE_MODEL_SIZES = {"small", "medium", "large"}

    @classmethod
    def from_preconfigured(cls, model_size: str) -> Self:
        """Return a new pre-configured model."""

        assert model_size in cls.AVAILABLE_MODEL_SIZES, "Invalid model size."

        num_primary_layers = 3
        num_quaternary_layers = 3

        match model_size:
            case "small":
                num_primary_channels = 64
                num_secondary_channels = 128
                num_secondary_layers = 3
                num_tertiary_channels = 256
                num_tertiary_layers = 9
                num_quaternary_channels = 512

            case "medium":
                num_primary_channels = 96
                num_secondary_channels = 192
                num_secondary_layers = 3
                num_tertiary_channels = 384
                num_tertiary_layers = 24
                num_quaternary_channels = 768

            case "large":
                num_primary_channels = 128
                num_secondary_channels = 256
                num_secondary_layers = 6
                num_tertiary_channels = 512
                num_tertiary_layers = 36
                num_quaternary_channels = 1024

        return cls(
            num_primary_channels,
            num_primary_layers,
            num_secondary_channels,
            num_secondary_layers,
            num_tertiary_channels,
            num_tertiary_layers,
            num_quaternary_channels,
            num_quaternary_layers,
        )

    def __init__(
        self,
        num_primary_channels: int,
        num_primary_layers: int,
        num_secondary_channels: int,
        num_secondary_layers: int,
        num_tertiary_channels: int,
        num_tertiary_layers: int,
        num_quaternary_channels: int,
        num_quaternary_layers: int,
    ):
        super().__init__()

        self.detector = Detector(
            num_primary_channels,
            num_primary_layers,
            num_secondary_channels,
            num_secondary_layers,
            num_tertiary_channels,
            num_tertiary_layers,
            num_quaternary_channels,
            num_quaternary_layers,
        )

        self.pool = AdaptiveAvgPool2d(1)

        self.flatten = Flatten(start_dim=1)

        self.classifier = BinaryClassifier(num_quaternary_channels)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_spectral_norms(self) -> None:
        """Add spectral normalization to the network."""

        self.detector.add_spectral_norms()

    def remove_parameterizations(self) -> None:
        """Remove all parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.detector.forward(x)

        z = self.pool.forward(z4)
        z = self.flatten.forward(z)
        z = self.classifier.forward(z)

        return z1, z2, z3, z4, z

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Return the probability that the input image is real or fake."""

        _, _, _, _, z = self.forward(x)

        return z


class Detector(Module):
    """A deep feature extraction network using convolutions."""

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.downsample1 = PixelCrush(3, primary_channels, 2)

        self.stage1 = Sequential(
            *[DetectorBlock(primary_channels, 4) for _ in range(primary_layers)],
        )

        self.downsample2 = PixelCrush(primary_channels, secondary_channels, 2)

        self.stage2 = Sequential(
            *[DetectorBlock(secondary_channels, 4) for _ in range(secondary_layers)],
        )

        self.downsample3 = PixelCrush(secondary_channels, tertiary_channels, 2)

        self.stage3 = Sequential(
            *[DetectorBlock(tertiary_channels, 4) for _ in range(tertiary_layers)],
        )

        self.downsample4 = PixelCrush(tertiary_channels, quaternary_channels, 2)

        self.stage4 = Sequential(
            *[DetectorBlock(quaternary_channels, 4) for _ in range(quaternary_layers)],
        )

        self.checkpoint = lambda layer, x: layer(x)

    def add_spectral_norms(self) -> None:
        self.downsample1.add_spectral_norms()
        self.downsample2.add_spectral_norms()
        self.downsample3.add_spectral_norms()
        self.downsample4.add_spectral_norms()

        for layer in self.stage1:
            layer.add_spectral_norms()

        for layer in self.stage2:
            layer.add_spectral_norms()

        for layer in self.stage3:
            layer.add_spectral_norms()

        for layer in self.stage4:
            layer.add_spectral_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.downsample1.forward(x)
        z1 = self.checkpoint(self.stage1.forward, z1)

        z2 = self.downsample2.forward(z1)
        z2 = self.checkpoint(self.stage2.forward, z2)

        z3 = self.downsample3.forward(z2)
        z3 = self.checkpoint(self.stage3.forward, z3)

        z4 = self.downsample4.forward(z3)
        z4 = self.checkpoint(self.stage4.forward, z4)

        return z1, z2, z3, z4


class DetectorBlock(Module):
    """A detector block with depth-wise separable convolution and residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(
            num_channels,
            num_channels,
            kernel_size=7,
            padding=3,
            groups=num_channels,
            bias=False,
        )

        self.conv2 = Conv2d(num_channels, hidden_channels, kernel_size=1)
        self.conv3 = Conv2d(hidden_channels, num_channels, kernel_size=1)

        self.silu = SiLU()

    def add_spectral_norms(self) -> None:
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = spectral_norm(self.conv2)
        self.conv3 = spectral_norm(self.conv3)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.conv2.forward(z)
        z = self.silu.forward(z)
        z = self.conv3.forward(z)

        z = x + z  # Local residual connection

        return z


class BinaryClassifier(Module):
    """A simple single-layer binary classification head to preserve positional invariance."""

    def __init__(self, input_features: int):
        super().__init__()

        self.linear = Linear(input_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear.forward(x)

        return z
