from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18

from .wide_resnet import wrn_28_10_cifar


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR images."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(
    name: str,
    num_classes: int = 10,
    *,
    pretrained: bool = False,
    pretrained_checkpoint: str | None = None,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "resnet18":
        if pretrained or pretrained_checkpoint:
            raise ValueError("Pretrained initialization options are only supported for deit_tiny_patch4_32.")
        return resnet18_cifar(num_classes=num_classes)
    if normalized in {"wrn_28_10", "wide_resnet_28_10", "wide_resnet28_10"}:
        if pretrained or pretrained_checkpoint:
            raise ValueError("Pretrained initialization options are only supported for deit_tiny_patch4_32.")
        return wrn_28_10_cifar(num_classes=num_classes)
    if normalized == "deit_tiny_patch4_32":
        # Import lazily so existing CNN commands do not require timm.
        from vit_12_runs_task.model.deit_cifar import build_deit_tiny_patch4_32

        return build_deit_tiny_patch4_32(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
        )
    if normalized == "deit_tiny_patch4_32_paper":
        # Isolated CIFAR-normalized paper wrapper. Existing CNN and legacy ViT
        # aliases retain their exact behavior.
        from vit_12_runs_task.paper_recipe.model import build_paper_deit_tiny_patch4_32

        return build_paper_deit_tiny_patch4_32(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
        )
    raise ValueError(f"Unsupported model: {name}")
