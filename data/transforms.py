"""Image transforms and augmentation pipeline.

Provides composable transforms for preprocessing and augmenting images
for both classification and segmentation tasks.
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms as T


class Resize:
    """Resize image and optionally mask to target size."""

    def __init__(self, size: Tuple[int, int], keep_aspect: bool = False) -> None:
        """Initialize resize transform.

        Args:
            size: Target size as (height, width).
            keep_aspect: If True, pad to maintain aspect ratio instead of stretching.
        """
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, sample: dict) -> dict:
        """Apply resize to image and optionally mask.

        Args:
            sample: Dict with keys 'image' and optionally 'mask' (torch tensors).

        Returns:
            Dict with resized tensors.
        """
        image = sample["image"]
        mask = sample.get("mask", None)

        # Resize image
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Resize mask (if present)
        if mask is not None:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=self.size,
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        sample["image"] = image
        if mask is not None:
            sample["mask"] = mask

        return sample


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize horizontal flip.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, sample: dict) -> dict:
        """Apply random horizontal flip.

        Args:
            sample: Dict with keys 'image' and optionally 'mask'.

        Returns:
            Flipped sample if random condition met.
        """
        if torch.rand(1).item() < self.p:
            image = torch.flip(sample["image"], dims=[-1])
            sample["image"] = image

            if "mask" in sample:
                mask = torch.flip(sample["mask"], dims=[-1])
                sample["mask"] = mask

        return sample


class RandomVerticalFlip:
    """Randomly flip image and mask vertically."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize vertical flip.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, sample: dict) -> dict:
        """Apply random vertical flip.

        Args:
            sample: Dict with keys 'image' and optionally 'mask'.

        Returns:
            Flipped sample if random condition met.
        """
        if torch.rand(1).item() < self.p:
            image = torch.flip(sample["image"], dims=[-2])
            sample["image"] = image

            if "mask" in sample:
                mask = torch.flip(sample["mask"], dims=[-2])
                sample["mask"] = mask

        return sample


class Normalize:
    """Normalize image using mean and std."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize normalization.

        Args:
            mean: Mean for each channel.
            std: Standard deviation for each channel.
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: dict) -> dict:
        """Normalize image.

        Args:
            sample: Dict with key 'image' as [C, H, W] tensor with values in [0, 1].

        Returns:
            Dict with normalized image.
        """
        image = sample["image"]
        image = (image - self.mean) / (self.std + 1e-8)
        sample["image"] = image
        return sample


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]) -> None:
        """Initialize transform composition.

        Args:
            transforms: List of transform callables.
        """
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        """Apply all transforms sequentially.

        Args:
            sample: Input sample dict.

        Returns:
            Transformed sample dict.
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Compose:
    """Get standard training transforms with augmentation.

    Args:
        image_size: Target image size (height, width).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Composed transform pipeline.
    """
    transforms_list = [
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.25),
    ]

    if normalize:
        transforms_list.append(Normalize())

    return Compose(transforms_list)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Compose:
    """Get standard validation transforms (no augmentation).

    Args:
        image_size: Target image size (height, width).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Composed transform pipeline.
    """
    transforms_list = [
        Resize(image_size),
    ]

    if normalize:
        transforms_list.append(Normalize())

    return Compose(transforms_list)
