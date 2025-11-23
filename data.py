from os import path, walk
from warnings import warn

import torch

from torch import Tensor

from torch.utils.data import Dataset

from torchvision.io import decode_image

from torchvision.transforms.v2 import (
    Transform,
    Grayscale,
    ToDtype,
)

from PIL import Image


class ColorCrush(Dataset):
    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        target_resolution: int,
        pre_transform: Transform | None,
    ):
        if target_resolution <= 0:
            raise ValueError(
                f"Target resolution must be positive, {target_resolution} given."
            )

        image_paths = []
        dropped = 0

        for folder_path, _, filenames in walk(root_path):
            for filename in filenames:
                if self.has_image_extension(filename):
                    image_path = path.join(folder_path, filename)

                    image = Image.open(image_path)

                    width, height = image.size

                    if width < target_resolution or height < target_resolution:
                        dropped += 1

                        continue

                    image_paths.append(image_path)

        if dropped > 0:
            warn(
                f"Dropped {dropped} images that were smaller "
                f"than the target resolution of {target_resolution}."
            )

        grayscale_transform = Grayscale(num_output_channels=1)

        to_tensor_transform = ToDtype(torch.float32, scale=True)

        self.image_paths = image_paths
        self.pre_transform = pre_transform
        self.grayscale_transform = grayscale_transform
        self.to_tensor_transform = to_tensor_transform

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        if self.pre_transform:
            image = self.pre_transform.forward(image)

        x = self.grayscale_transform.forward(image)
        x = self.to_tensor_transform.forward(x)

        y = self.to_tensor_transform.forward(image)

        return x, y

    def __len__(self):
        return len(self.image_paths)


class ImagePairs(Dataset):
    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(self, lr_root_path: str, hr_root_path: str):
        lr_image_paths = []
        hr_image_paths = []

        batch = [
            (lr_root_path, lr_image_paths),
            (hr_root_path, hr_image_paths),
        ]

        for root_path, image_paths in batch:
            for folder_path, _, filenames in walk(root_path):
                for filename in filenames:
                    if self.has_image_extension(filename):
                        image_path = path.join(folder_path, filename)

                        image_paths.append(image_path)

        self.lr_image_paths = lr_image_paths
        self.hr_image_paths = hr_image_paths

        self.transformer = ToDtype(torch.float32, scale=True)

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        lr_image_path = self.lr_image_paths[index]
        hr_image_path = self.hr_image_paths[index]

        lr_image = decode_image(lr_image_path, mode=self.IMAGE_MODE)
        hr_image = decode_image(hr_image_path, mode=self.IMAGE_MODE)

        x = self.transformer(lr_image)
        y = self.transformer(hr_image)

        return x, y

    def __len__(self):
        return len(self.lr_image_paths)
