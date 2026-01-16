"""Generic dataset classes for classification and segmentation.

Provides PyTorch Dataset implementations that accept image paths, labels,
and masks. No hardcoded datasets or assumptions.
"""

from typing import Callable, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ClassificationDataset(Dataset):
    """Generic classification dataset.

    Loads images from file paths and associates them with class labels.
    Supports optional transforms applied at loading time.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize classification dataset.

        Args:
            image_paths: List of paths to image files (PNG, JPG, etc.).
            labels: List of class labels (integers) corresponding to images.
            transforms: Optional transform pipeline to apply to each sample.

        Raises:
            ValueError: If image_paths and labels have different lengths.
        """
        if len(image_paths) != len(labels):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of labels ({len(labels)})"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transforms = transforms

        # Verify files exist
        missing = [p for p in self.image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing[:5]}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys:
            - 'image': Image tensor [C, H, W] with values in [0, 1]
            - 'label': Class label (integer)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label = self.labels[idx]

        sample = {"image": image, "label": label}

        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class SegmentationDataset(Dataset):
    """Generic segmentation dataset.

    Loads image-mask pairs from file paths. Supports optional transforms
    that are applied to both image and mask consistently.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transforms: Optional[Callable] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """Initialize segmentation dataset.

        Args:
            image_paths: List of paths to image files.
            mask_paths: List of paths to mask files (single-channel, pixel-wise labels).
            transforms: Optional transform pipeline (applied to both image and mask).
            num_classes: Number of segmentation classes (for validation).

        Raises:
            ValueError: If paths have different lengths or files missing.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of masks ({len(mask_paths)})"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.transforms = transforms
        self.num_classes = num_classes

        # Verify files exist
        missing_img = [p for p in self.image_paths if not p.exists()]
        missing_msk = [p for p in self.mask_paths if not p.exists()]
        if missing_img or missing_msk:
            raise FileNotFoundError(
                f"Missing image files: {missing_img[:3]} or mask files: {missing_msk[:3]}"
            )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single image-mask pair.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys:
            - 'image': Image tensor [C, H, W] with values in [0, 1]
            - 'mask': Segmentation mask [H, W] with integer class indices
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load mask (single channel, integer labels)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        mask = torch.from_numpy(np.array(mask)).long()

        sample = {"image": image, "mask": mask}

        # Apply transforms (both image and mask)
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Validate class indices if num_classes provided
        if self.num_classes is not None:
            max_class = sample["mask"].max().item()
            if max_class >= self.num_classes:
                raise ValueError(
                    f"Mask contains class {max_class} but num_classes={self.num_classes}"
                )

        return sample


class DummyClassificationDataset(Dataset):
    """Utility dataset for testing (generates random images)."""

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 10,
        image_size: Tuple[int, int] = (224, 224),
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize dummy dataset.

        Args:
            num_samples: Number of random samples to generate.
            num_classes: Number of classes.
            image_size: Image size (height, width).
            transforms: Optional transforms.
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Generate random sample.

        Args:
            idx: Sample index (unused).

        Returns:
            Dict with random image and label.
        """
        image = torch.rand(3, *self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()

        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class DummySegmentationDataset(Dataset):
    """Utility dataset for testing segmentation (generates random images and masks)."""

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 5,
        image_size: Tuple[int, int] = (224, 224),
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize dummy segmentation dataset.

        Args:
            num_samples: Number of random samples.
            num_classes: Number of segmentation classes.
            image_size: Image size (height, width).
            transforms: Optional transforms.
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Generate random sample with mask.

        Args:
            idx: Sample index (unused).

        Returns:
            Dict with random image and segmentation mask.
        """
        image = torch.rand(3, *self.image_size)
        mask = torch.randint(0, self.num_classes, self.image_size)

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
