from collections.abc import Callable
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from tqdm import tqdm

from docta.utils.config import Config

DEBUG = False


class ImageDataset(VisionDataset):
    def __init__(
        self,
        image_metadata: pd.DataFrame,
        root: Path,
        image_dir: Path,
        label_name: str,
        target_size: tuple[int, int] = (224, 224),
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.image_metadata = image_metadata
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_name = label_name

        self.root = root
        self.classes = list(self.image_metadata[label_name].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.target_size = target_size
        batch_size = 1000
        batch_number = 0
        self.data = []
        self.batch = []
        batch_loaded = False

        for idx, img_path in enumerate(
            tqdm(self.image_metadata["path"], desc="Processing Images")
        ):
            if idx < batch_number * batch_size and batch_loaded:
                continue
            # Determine batch filename
            elif idx % batch_size == 0:
                batch_filename = self.get_batch_filename(batch_number)
                if idx > 0 and not batch_loaded:
                    self.save_batch(batch_number)
                batch_number += 1
                batch_loaded = False
                self.batch = []
                batch_filename = self.get_batch_filename(batch_number)

                if batch_filename.exists():
                    self.load_batch(batch_filename)
                    batch_loaded = True
                    continue

            # Process image
            img_object = (
                Image.open(f"{self.image_dir}/{img_path}")
                .convert("RGB")
                .resize(target_size)
            )
            self.batch.append(np.array(img_object))

        # Save the last batch if necessary
        if self.batch and not self.get_batch_filename(batch_number).exists():
            self.save_batch(batch_number)

        self.data = np.vstack(self.data)
        # self.data = self.data.transpose((0, 2, 3, 1))

        self.targets = (
            self.image_metadata[label_name].map(lambda x: self.class_to_idx[x]).tolist()
        )
        if DEBUG:
            print("data mean: ", self.data.mean(axis=(0, 1, 2)))
            print("data std: ", self.data.std(axis=(0, 1, 2)))

    def get_batch_filename(self, batch_number):
        filename = (
            self.root.parent / f"{self.root.name}_batches" / f"{batch_number:03d}.npy"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        return filename

    def save_batch(self, batch_number):
        batch_filename = self.get_batch_filename(batch_number)
        batch = np.stack(self.batch, axis=0)
        np.save(batch_filename, batch)
        self.data.append(batch)
        if DEBUG:
            print(f"Saved {batch_filename}")

    def load_batch(self, batch_filename):
        batch = np.load(batch_filename)
        self.data.append(batch)
        if DEBUG:
            print(f"Loaded {batch_filename}")

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, index: int) -> tuple[ImageFile.ImageFile, str]:
        image, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


class MultiLabelImageDataset(ImageDataset):
    mean = (124.42204022, 130.19395847, 98.68810746)
    std = (55.03071655, 52.61038074, 56.0628322)

    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def __init__(self, cfg, train=True, preprocess=None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        root = get_project_root() / cfg.data_root
        image_metadata = pd.read_csv(root.with_suffix(".csv"))
        super().__init__(
            root=root,
            image_dir=cfg.image_dir,
            label_name=cfg.label_name,
            image_metadata=image_metadata,
            transform=preprocess,
            target_transform=None,
        )

        self.cfg = cfg

        self.label = self.targets
        self.feature = self.data

    def __getitem__(self, index: int) -> tuple[ImageFile.ImageFile, int, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, (target, noisy_label), index).
            target: clean label
            noisy_label: loaded/synthesized noisy label
        """
        img, label = super().__getitem__(index)

        return img, label, index


class ImageDataset10Classes(ImageDataset):
    mean = (120.31289279, 136.31811933, 91.3144624)
    std = (52.96044569, 51.03640914, 52.84869974)

    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def __init__(self, cfg, train=True, preprocess=None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        root = get_project_root() / cfg.data_root
        image_metadata = pd.read_csv(root.with_suffix(".csv"))
        super().__init__(
            root=root,
            image_dir=cfg.image_dir,
            label_name=cfg.label_name,
            image_metadata=image_metadata,
            transform=preprocess,
            target_transform=None,
        )

        self.cfg = cfg

        self.label = self.targets
        self.feature = self.data

    def __getitem__(self, index: int) -> tuple[ImageFile.ImageFile, int, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, (target, noisy_label), index).
            target: clean label
            noisy_label: loaded/synthesized noisy label
        """
        img, label = super().__getitem__(index)

        return img, label, index


def get_project_root():
    return Path(__file__).parents[2]


if __name__ == "__main__":
    root = get_project_root() / "data/image_data/2024-08-21_labelled_data"
    image_metadata = pd.read_csv(root.with_suffix(".csv"))
    ds = ImageDataset(
        image_metadata=image_metadata,
        root=root,
        image_dir=root,
        label_name="andjela_labels",
    )
    print(ds[0])
    cfg = Config.fromfile("./config/image_classification_data.py")
    ds2 = MultiLabelImageDataset(cfg=cfg, train=True)
    print(ds2[0][0].shape)

    print(len(ds.class_to_idx))
