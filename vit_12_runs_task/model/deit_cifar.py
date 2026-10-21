from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


DEIT_ALIAS = "deit_tiny_patch4_32"
TIMM_MODEL_ID = "deit_tiny_patch16_224.fb_in1k"
TIMM_PUBLIC_NAME = "deit_tiny_patch16_224"
PINNED_TIMM_VERSION = "1.0.15"
INPUT_SIZE = 32
PATCH_SIZE = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            f"deit_tiny_patch4_32 requires timm=={PINNED_TIMM_VERSION}. Use the configured project "
            "environment; do not silently substitute a different ViT implementation."
        ) from exc
    if timm.__version__ != PINNED_TIMM_VERSION:
        raise RuntimeError(
            f"deit_tiny_patch4_32 was tested with timm=={PINNED_TIMM_VERSION}, "
            f"but the selected environment provides timm=={timm.__version__}. "
            "Do not update the shared environment; report the dependency mismatch."
        )
    return timm


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class PixelSpaceImageNetNormalizer(nn.Module):
    """Keep attacks in pixel space and normalize only at the backbone boundary."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("normalization_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("normalization_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.normalization_mean.to(dtype=x.dtype)
        std = self.normalization_std.to(dtype=x.dtype)
        return self.backbone((x - mean) / std)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("The pretrained checkpoint does not contain a non-empty state dictionary.")

    state: dict[str, torch.Tensor] = {}
    for raw_key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state[key] = value
    if not state:
        raise ValueError("No tensor entries were found in the pretrained checkpoint.")
    return state


def _load_local_pretrained(backbone: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    state = _extract_state_dict(payload)

    try:
        from timm.models.vision_transformer import checkpoint_filter_fn

        state = checkpoint_filter_fn(state, backbone)
    except Exception as exc:
        raise RuntimeError(
            f"timm could not adapt the 224/patch16 checkpoint at {path} to 32/patch4."
        ) from exc

    target_state = backbone.state_dict()
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
        if not key.startswith(("head_dist.",)) and key != "dist_token"
    )
    if missing or unexpected:
        raise ValueError(
            "Pretrained checkpoint was not architecture-compatible after timm adaptation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "kind": "local_checkpoint",
        "path": str(path),
        "sha256": file_sha256(path),
        "dropped_classifier_keys": sorted(dropped_classifier),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_backbone(num_classes: int, pretrained: bool) -> nn.Module:
    timm = _require_timm()
    try:
        pretrained_cfg_overlay = {"hf_hub_id": None} if pretrained else None
        return timm.create_model(
            TIMM_MODEL_ID,
            pretrained=pretrained,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            img_size=INPUT_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Unable to load the official ImageNet-pretrained DeiT-Tiny weights. "
                "If network/cache access is unavailable, first run "
                "vit_12_runs_task/scripts/prepare_pretrained_weights.py and pass "
                "--pretrained_checkpoint PATH. Keep TORCH_HOME/HF_HOME inside "
                "vit_12_runs_task as documented in its README."
            ) from exc
        raise


def build_deit_tiny_patch4_32(
    num_classes: int = 10,
    *,
    pretrained: bool = False,
    pretrained_checkpoint: str | Path | None = None,
) -> PixelSpaceImageNetNormalizer:
    """Build the task's CIFAR-10 DeiT using timm's ViT and resampling code."""

    checkpoint = str(pretrained_checkpoint).strip() if pretrained_checkpoint is not None else ""
    if checkpoint:
        backbone = _create_backbone(num_classes=num_classes, pretrained=False)
        source = _load_local_pretrained(backbone, checkpoint)
        pretrained_loaded = True
    else:
        backbone = _create_backbone(num_classes=num_classes, pretrained=pretrained)
        source = {
            "kind": "timm_pretrained" if pretrained else "random_initialization",
            "model_id": TIMM_MODEL_ID,
        }
        pretrained_loaded = bool(pretrained)

    model = PixelSpaceImageNetNormalizer(backbone)
    timm = _require_timm()
    pretrained_cfg = _json_safe(getattr(backbone, "pretrained_cfg", {}))
    model._moo_model_provenance = {  # type: ignore[attr-defined]
        "model_alias": DEIT_ALIAS,
        "timm_model_id": TIMM_MODEL_ID,
        "timm_public_name": TIMM_PUBLIC_NAME,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "num_classes": int(num_classes),
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "pretrained_loaded": pretrained_loaded,
        "pretrained_source": source,
        "pretrained_config": pretrained_cfg,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }
    return model


def initial_trainable_parameters_hash(model: nn.Module) -> str:
    """Hash names, shapes, dtypes, and initial trainable parameter bytes."""

    digest = hashlib.sha256()
    count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            tensor = parameter.detach().cpu().contiguous()
            header = json.dumps(
                {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            count += tensor.numel()
    digest.update(f"numel={count}".encode("ascii"))
    return digest.hexdigest()


def model_provenance(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "_moo_model_provenance", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}
