from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vit_12_runs_task.paper_recipe.config import METHODS, file_sha256, paper_recipe_hash
from vit_12_runs_task.paper_recipe.integrity import implementation_hash
from vit_12_runs_task.paper_recipe.model import build_paper_deit_tiny_patch4_32, initial_trainable_parameters_hash


def require_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.relative_to(TASK_ROOT)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify matched-seed initialization for all four paper methods.")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="vit_12_runs_task/configs/paper_initialization_hashes.json")
    args = parser.parse_args()
    checkpoint = require_task(args.pretrained_checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Missing pretrained checkpoint: {checkpoint}")
    result: dict[str, object] = {
        "paper_recipe_hash": paper_recipe_hash(),
        "implementation_hash": implementation_hash(),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": file_sha256(checkpoint),
        "seeds": {},
    }
    for seed in args.seeds:
        hashes: dict[str, str] = {}
        for method in METHODS:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = build_paper_deit_tiny_patch4_32(pretrained=False, pretrained_checkpoint=checkpoint)
            hashes[method] = initial_trainable_parameters_hash(model)
            del model
        if len(set(hashes.values())) != 1:
            raise SystemExit(f"Initialization mismatch for seed {seed}: {hashes}")
        result["seeds"][str(seed)] = {"hash": next(iter(hashes.values())), "methods": hashes}  # type: ignore[index]
    output = require_task(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
