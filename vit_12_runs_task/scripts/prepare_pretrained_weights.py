from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vit_12_runs_task.model.deit_cifar import TIMM_MODEL_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_under_task(path: Path) -> Path:
    task_root = Path(__file__).resolve().parents[1]
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(task_root)
    except ValueError as exc:
        raise ValueError(f"Output must stay under {task_root}; got {resolved}") from exc
    return resolved


def validate_cache_environment() -> None:
    required = [
        "TORCH_HOME",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
        "MPLCONFIGDIR",
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "Set all project-local cache variables before downloading weights; missing: " + ", ".join(missing)
        )
    for name in required:
        path = require_under_task(Path(os.environ[name]))
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache an official DeiT-Tiny source checkpoint under the task tree.")
    parser.add_argument(
        "--output",
        default="vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt",
    )
    args = parser.parse_args()
    validate_cache_environment()

    try:
        import timm
    except ImportError as exc:
        raise SystemExit("timm==1.0.15 is required in the selected read-only Python environment.") from exc
    if timm.__version__ != "1.0.15":
        raise SystemExit(f"Expected timm==1.0.15, found {timm.__version__}.")

    output = require_under_task(Path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing checkpoint: {output}")

    # The server environment intentionally has no huggingface_hub package. The
    # official timm config also carries Facebook's URL, so force that source.
    model = timm.create_model(
        TIMM_MODEL_ID,
        pretrained=True,
        pretrained_cfg_overlay={"hf_hub_id": None},
    )
    payload = {
        "state_dict": model.state_dict(),
        "source": TIMM_MODEL_ID,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "pretrained_cfg": getattr(model, "pretrained_cfg", {}),
    }
    torch.save(payload, output)
    metadata = {
        "path": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "source": TIMM_MODEL_ID,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
