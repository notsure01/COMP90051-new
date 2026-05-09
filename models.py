"""
Three model architectures with increasing complexity:

  1. SimpleCNN     – basic convolutional network              (low complexity)
  2. SimpleResNet  – residual network with skip connections   (medium complexity)
  3. ViT           – Vision Transformer (Dosovitskiy et al.,  (high complexity)
                     "An Image is Worth 16x16 Words",
                     ICLR 2021, https://openreview.net/forum?id=YicbFdNTTy)

Each model's constructor accepts the hyperparameters that are tuned:
  CNN    : dropout
  ResNet : (lr/weight_decay are passed to the optimiser, not the model)
  ViT    : patch_size

The factory function `build_model` is the single entry point used by
train_utils.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ===========================================================================
# 1. SimpleCNN
# ===========================================================================

class SimpleCNN(nn.Module):
    """
    Four convolutional stages, each with Conv → BN → ReLU → MaxPool,
    followed by a dropout-regularised fully-connected classifier.

    Tuned hyperparameter: dropout (controls regularisation strength).
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: 128×128 → 64×64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 2: 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 3: 32×32 → 16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 4: 16×16 → 8×8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (B, 256, 1, 1)
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# 2. SimpleResNet
# ===========================================================================

class _ResBlock(nn.Module):
    """Basic residual block: two 3×3 convolutions with a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SimpleResNet(nn.Module):
    """
    Custom ResNet with 4 residual stages (inspired by ResNet-18 but lighter).
    Channels: 64 → 128 → 256 → 512, each stage has 2 residual blocks.

    Tuned hyperparameters: lr, weight_decay (applied to the optimiser).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(64,  64,  2, stride=1)
        self.layer2 = self._make_stage(64,  128, 2, stride=2)
        self.layer3 = self._make_stage(128, 256, 2, stride=2)
        self.layer4 = self._make_stage(256, 512, 2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_stage(in_c: int, out_c: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(_ResBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


# ===========================================================================
# 3. ViT  (Dosovitskiy et al., ICLR 2021)
# ===========================================================================

class _PatchEmbedding(nn.Module):
    """
    Divide the image into non-overlapping patches and project each patch
    into an embedding vector using a single Conv2d with stride == patch_size.
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_channels: int,
                 embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"Image size {img_size} must be divisible by patch_size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)            # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)       # (B, num_patches, embed_dim)
        return x


class _TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block:
      LayerNorm → Multi-head Self-Attention (pre-norm) + residual
      LayerNorm → MLP (pre-norm) + residual
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=attn_drop,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Reference:
      Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale." ICLR 2021.
      https://openreview.net/forum?id=YicbFdNTTy

    Tuned hyperparameters:
      lr         – learning rate of the Adam optimiser
      patch_size – spatial size of each image patch (8, 16, or 32)

    Architecture choices keep computation feasible for nested CV:
      embed_dim = 192, depth = 6, num_heads = 4 (48 dims/head)
    """

    def __init__(self,
                 img_size: int   = config.IMG_SIZE,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = config.NUM_CLASSES,
                 embed_dim: int  = 192,
                 depth: int      = 6,
                 num_heads: int  = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float  = 0.1):
        super().__init__()

        self.patch_embed = _PatchEmbedding(img_size, patch_size,
                                           in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads,
                              mlp_ratio, dropout, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B      = x.shape[0]
        x      = self.patch_embed(x)                         # (B, N, D)
        cls    = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
        x      = torch.cat([cls, x], dim=1)                  # (B, N+1, D)
        x      = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])                            # CLS token output


# ===========================================================================
# Factory
# ===========================================================================

def build_model(name: str,
                num_classes: int = config.NUM_CLASSES,
                **hp) -> nn.Module:
    """
    Instantiate a model by name, injecting the relevant hyperparameters.

    Parameters
    ----------
    name        : "CNN" | "ResNet" | "ViT"
    num_classes : number of output classes
    **hp        : hyperparameter values from the current HP combo
                  (only model-architecture HPs are used here; lr/weight_decay
                   are consumed by build_optimizer in train_utils.py)

    Returns
    -------
    An initialised nn.Module.
    """
    if name == "CNN":
        return SimpleCNN(num_classes=num_classes,
                         dropout=float(hp.get("dropout", 0.4)))
    elif name == "ResNet":
        return SimpleResNet(num_classes=num_classes)
    elif name == "ViT":
        return ViT(num_classes=num_classes,
                   patch_size=int(hp.get("patch_size", 16)))
    else:
        raise ValueError(f"Unknown model name: {name!r}")
