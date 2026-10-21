from __future__ import annotations

import os
import sys
import uuid
from multiprocessing import connection as multiprocessing_connection
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_ABSTRACT_SOCKET_PATCHED = False


def _configure_linux_abstract_multiprocessing_sockets() -> None:
    """Keep multiprocessing resource-sharer sockets off NFS-backed TMPDIR."""
    global _ABSTRACT_SOCKET_PATCHED
    if _ABSTRACT_SOCKET_PATCHED or sys.platform != "linux":
        return
    original = multiprocessing_connection.arbitrary_address

    def nfs_safe_arbitrary_address(family: str):
        if family == "AF_UNIX":
            # A leading NUL selects Linux's kernel-managed abstract socket
            # namespace. No listener file is created, so there is nothing for
            # NFS to rename to `.nfs*` or fail to unlink during worker exit.
            return f"\0pda-dataloader-{os.getpid()}-{uuid.uuid4().hex}"
        return original(family)

    multiprocessing_connection.arbitrary_address = nfs_safe_arbitrary_address
    _ABSTRACT_SOCKET_PATCHED = True


def shutdown_dataloader_workers(*loaders: DataLoader) -> None:
    """Explicitly stop persistent DataLoader workers before Python finalizers."""
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
            loader._iterator = None


def num_classes_for_dataset(dataset: str) -> int:
    name = dataset.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_dataloaders(
    dataset: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    use_fake_data: bool = False,
    fake_train_size: int = 512,
    fake_test_size: int = 256,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, int]:
    if num_workers > 0:
        _configure_linux_abstract_multiprocessing_sockets()

    name = dataset.lower()
    num_classes = num_classes_for_dataset(name)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=fake_train_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_transform,
        )
        test_dataset = datasets.FakeData(
            size=fake_test_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=test_transform,
        )
    else:
        root = Path(data_dir)
        dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train_dataset = dataset_cls(root=root, train=True, download=download, transform=train_transform)
        test_dataset = dataset_cls(root=root, train=False, download=download, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    # The experiment workspace is on NFS. Recreating workers for every iterator
    # leaves multiprocessing listener sockets in TMPDIR while NFS tries to
    # unlink them, producing noisy `.nfs*` "Device or resource busy" errors.
    # Persistent workers also avoid repeated startup overhead across epochs and
    # across the several attack evaluations performed with the same loader.
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    return train_loader, test_loader, num_classes
