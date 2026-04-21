from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn

ModelFactory = Callable[..., nn.Module]
PathLike = str | Path


def _resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint_model(
    checkpoint_path: PathLike,
    model_factory: ModelFactory,
    device: torch.device | str | None = None,
) -> nn.Module:
    target_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    model = model_factory(
        in_channels=checkpoint["in_channels"],
        latent_dims=checkpoint["latent_dims"],
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_autoencoder(
    checkpoint_path: PathLike,
    model_factory: ModelFactory,
    device: torch.device | str | None = None,
) -> nn.Module:
    return _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        model_factory=model_factory,
        device=device,
    )


def load_vae(
    checkpoint_path: PathLike,
    model_factory: ModelFactory,
    device: torch.device | str | None = None,
) -> nn.Module:
    return _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        model_factory=model_factory,
        device=device,
    )
