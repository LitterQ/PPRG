from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from vit_12_runs_task.model.deit_cifar import (
    INPUT_SIZE,
    PATCH_SIZE,
    _create_backbone,
    _extract_state_dict,
    _json_safe,
    initial_trainable_parameters_hash,
)

from .config import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    PAPER_PATCH_EMBED_ADAPTATION,
    PAPER_POS_EMBED_ADAPTATION,
    PAPER_MODEL_ALIAS,
    PINNED_TIMM_VERSION,
    PRETRAINED_MODEL_ID,
    PRETRAINED_PUBLIC_NAME,
    file_sha256,
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _official_block_sum_patch_embed(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Apply the paper repository's exact patch16 -> patch4 checkpoint filter."""

    if source.ndim != 4 or len(target_shape) != 4:
        raise ValueError(f"Patch weights must be OIHW tensors; source={tuple(source.shape)}, target={tuple(target_shape)}")
    if tuple(source.shape[:2]) != tuple(target_shape[:2]):
        raise ValueError(f"Patch channel mismatch: source={tuple(source.shape)}, target={tuple(target_shape)}")
    target_h, target_w = int(target_shape[2]), int(target_shape[3])
    source_h, source_w = int(source.shape[2]), int(source.shape[3])
    if target_h != PATCH_SIZE or target_w != PATCH_SIZE:
        raise ValueError(f"Paper adapter requires a {PATCH_SIZE}x{PATCH_SIZE} target, got {target_h}x{target_w}")
    if source_h % target_h or source_w % target_w:
        raise ValueError(f"Patch size is not block-summable: source={source_h}x{source_w}, target={target_h}x{target_w}")
    return source.reshape(
        *source.shape[:2], target_h, source_h // target_h, target_w, source_w // target_w
    ).sum(dim=(3, 5))


def _official_resize_pos_embed(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tokens: int,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Apply the pinned repository's bilinear position-embedding resize."""

    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(f"Position embeddings must have shape [1,N,C]; source={tuple(source.shape)}, target={tuple(target.shape)}")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError(f"Position embedding width mismatch: source={tuple(source.shape)}, target={tuple(target.shape)}")
    if num_tokens < 0 or source.shape[1] <= num_tokens:
        raise ValueError(f"Invalid prefix-token count {num_tokens} for source shape {tuple(source.shape)}")
    token_embed = source[:, :num_tokens]
    grid_embed = source[0, num_tokens:]
    old_grid = math.isqrt(int(grid_embed.shape[0]))
    if old_grid * old_grid != int(grid_embed.shape[0]):
        raise ValueError(f"Source position grid is not square: {grid_embed.shape[0]} tokens")
    new_h, new_w = (int(grid_size[0]), int(grid_size[1]))
    if target.shape[1] != num_tokens + new_h * new_w:
        raise ValueError(
            f"Target position shape disagrees with prefix/grid sizes: target={tuple(target.shape)}, "
            f"num_tokens={num_tokens}, grid={grid_size}"
        )
    grid_embed = grid_embed.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    # Official pinned code calls F.interpolate(..., mode='bilinear'); these explicit
    # flags preserve that behavior and prevent a future timm default from changing it.
    grid_embed = F.interpolate(
        grid_embed,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    return torch.cat((token_embed, grid_embed), dim=1)


def _load_official_paper_pretrained(backbone: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load ImageNet DeiT weights using the pinned paper repository's adapter."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    state = _extract_state_dict(payload)
    target_state = backbone.state_dict()

    patch_key = "patch_embed.proj.weight"
    pos_key = "pos_embed"
    if patch_key not in state or pos_key not in state:
        raise ValueError(f"Pretrained checkpoint is missing {patch_key!r} or {pos_key!r}.")
    source_patch_shape = list(state[patch_key].shape)
    source_pos_shape = list(state[pos_key].shape)
    state[patch_key] = _official_block_sum_patch_embed(state[patch_key], target_state[patch_key].shape)
    state[pos_key] = _official_resize_pos_embed(
        state[pos_key],
        target_state[pos_key],
        num_tokens=int(getattr(backbone, "num_prefix_tokens", 1)),
        grid_size=tuple(int(value) for value in backbone.patch_embed.grid_size),
    )

    dropped_classifier: list[str] = []
    for key in list(state):
        if key in target_state and state[key].shape != target_state[key].shape:
            if key.startswith(("head.", "head_dist.")):
                dropped_classifier.append(key)
                del state[key]
            else:
                raise ValueError(
                    f"Unexpected pretrained shape mismatch for {key}: "
                    f"source={tuple(state[key].shape)}, target={tuple(target_state[key].shape)}"
                )

    incompatible = backbone.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias", "head_dist.weight", "head_dist.bias"}
    missing = sorted(key for key in incompatible.missing_keys if key not in allowed_missing)
    unexpected = sorted(
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("head_dist.") and key != "dist_token"
    )
    if missing or unexpected:
        raise ValueError(
            "Pretrained checkpoint was not architecture-compatible after official paper adaptation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "kind": "local_checkpoint",
        "path": str(path),
        "sha256": file_sha256(path),
        "adapter": {
            "patch_embed": PAPER_PATCH_EMBED_ADAPTATION,
            "position_embedding": PAPER_POS_EMBED_ADAPTATION,
            "patch_source_shape": source_patch_shape,
            "patch_target_shape": list(state[patch_key].shape),
            "patch_target_sha256": _tensor_sha256(state[patch_key]),
            "position_source_shape": source_pos_shape,
            "position_target_shape": list(state[pos_key].shape),
            "position_target_sha256": _tensor_sha256(state[pos_key]),
        },
        "dropped_classifier_keys": sorted(dropped_classifier),
    }


class PixelSpaceCIFAR10Normalizer(nn.Module):
    """Accept [0,1] pixels and apply official CIFAR normalization exactly once."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("normalization_mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("normalization_std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.normalization_mean.to(dtype=x.dtype)
        std = self.normalization_std.to(dtype=x.dtype)
        return self.backbone((x - mean) / std)


def build_paper_deit_tiny_patch4_32(
    num_classes: int = 10,
    *,
    pretrained: bool = False,
    pretrained_checkpoint: str | Path | None = None,
) -> PixelSpaceCIFAR10Normalizer:
    """Construct the isolated paper model without changing the legacy wrapper."""

    checkpoint = str(pretrained_checkpoint).strip() if pretrained_checkpoint is not None else ""
    if checkpoint:
        backbone = _create_backbone(num_classes=num_classes, pretrained=False)
        source = _load_official_paper_pretrained(backbone, checkpoint)
        pretrained_loaded = True
    elif pretrained:
        raise ValueError(
            "The paper recipe requires --pretrained-checkpoint so the pinned official "
            "patch/position adapter and checkpoint SHA256 are both auditable."
        )
    else:
        backbone = _create_backbone(num_classes=num_classes, pretrained=False)
        source = {
            "kind": "random_initialization",
            "model_id": PRETRAINED_MODEL_ID,
        }
        pretrained_loaded = False

    model = PixelSpaceCIFAR10Normalizer(backbone)
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - _create_backbone already gives the primary error
        raise RuntimeError(f"Paper model requires timm=={PINNED_TIMM_VERSION}") from exc
    model._moo_model_provenance = {  # type: ignore[attr-defined]
        "model_alias": PAPER_MODEL_ALIAS,
        "recipe": "paper",
        "timm_model_id": PRETRAINED_MODEL_ID,
        "timm_public_name": PRETRAINED_PUBLIC_NAME,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "num_classes": int(num_classes),
        "normalization_mode": "once_inside_model_from_pixel_space",
        "normalization_mean": list(CIFAR10_MEAN),
        "normalization_std": list(CIFAR10_STD),
        "pretrained_loaded": pretrained_loaded,
        "pretrained_source": source,
        "pretrained_adaptation": {
            "patch_embed": PAPER_PATCH_EMBED_ADAPTATION,
            "position_embedding": PAPER_POS_EMBED_ADAPTATION,
            "generic_timm_checkpoint_filter": False,
        },
        "pretrained_config": _json_safe(getattr(backbone, "pretrained_cfg", {})),
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }
    return model


def model_provenance(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "_moo_model_provenance", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


__all__ = [
    "PixelSpaceCIFAR10Normalizer",
    "_official_block_sum_patch_embed",
    "_official_resize_pos_embed",
    "build_paper_deit_tiny_patch4_32",
    "initial_trainable_parameters_hash",
    "model_provenance",
]
