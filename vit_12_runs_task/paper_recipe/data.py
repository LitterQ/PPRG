from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms

from pareto_attack_dg.utils.data import (
    _configure_linux_abstract_multiprocessing_sockets,
    shutdown_dataloader_workers,
)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EpochSeededDataset(Dataset[Any]):
    """Make stochastic transforms a pure function of seed, epoch, and index."""

    def __init__(self, dataset: Dataset[Any], *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __getitem__(self, index: int) -> Any:
        epoch = int(self._epoch[0].item())
        value = ((self.seed + 1) * 1_000_003 + epoch * 10_000_019 + int(index) * 101) % (2**31)
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            torch.manual_seed(value)
            np.random.seed(value % (2**32))
            random.seed(value)
            return self.dataset[index]
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class EpochRandomSampler(Sampler[int]):
    """Shuffle deterministically without sharing DataLoader's worker-seed RNG."""

    def __init__(self, dataset: Dataset[Any], *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            ((self.seed + 1) * 1_000_003 + self.epoch * 10_000_019) % (2**63 - 1)
        )
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def set_paper_loader_epoch(loader: DataLoader[Any], epoch: int) -> None:
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def build_paper_dataloaders(
    *,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    use_fake_data: bool = False,
    fake_train_size: int = 8,
    fake_test_size: int = 4,
    download: bool = False,
    eval_max_samples: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    if num_workers > 0:
        _configure_linux_abstract_multiprocessing_sockets()

    # The official source includes Resize(32,32), which is an identity for
    # CIFAR-10. Keeping pixel tensors unnormalized here is numerically
    # equivalent to the official normalized-space attack because the model
    # wrapper applies the same affine transform exactly once.
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=fake_train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
            random_offset=0,
        )
        test_dataset = datasets.FakeData(
            size=fake_test_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
            random_offset=100000,
        )
    else:
        root = Path(data_dir)
        train_dataset = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download)

    train_seed = int(generator.initial_seed())
    train_dataset = EpochSeededDataset(train_dataset, seed=train_seed)
    train_sampler = EpochRandomSampler(train_dataset, seed=train_seed)

    full_test_length = len(test_dataset)
    if eval_max_samples:
        if not 0 < eval_max_samples <= full_test_length:
            raise ValueError(f"eval_max_samples must be in [1,{full_test_length}], got {eval_max_samples}")
        test_dataset = Subset(test_dataset, list(range(eval_max_samples)))

    persistent_workers = num_workers > 0
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=False,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    metadata = {
        "train_dataset_length": len(train_dataset),
        "test_dataset_length": len(test_dataset),
        "full_test_dataset_length": full_test_length,
        "expected_train_batches": len(train_loader),
        "drop_last": False,
        "shuffle_train": True,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "worker_seed_strategy": "torch.initial_seed -> NumPy/Python",
        "augmentation_seed_strategy": "stateless(seed,epoch,index)",
        "sampler_seed_strategy": "stateless(seed,epoch)",
    }
    return train_loader, test_loader, metadata


__all__ = [
    "build_paper_dataloaders",
    "set_paper_loader_epoch",
    "shutdown_dataloader_workers",
]
