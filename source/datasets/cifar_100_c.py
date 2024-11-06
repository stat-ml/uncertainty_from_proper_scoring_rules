import logging
import os
import tarfile
import urllib.request
from ctypes import ArgumentError

import numpy
import torch
import torch.utils.data

import source.source.path_config

LOGGER = logging.getLogger(__name__)
ALLOWED_PERTURBATION_TYPES = {
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
}


# root (str) – Root directory of the datasets.
# transform (callable, optional) – A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop. Defaults to None.
# target_transform (callable, optional) – A function/transform that takes in the target and transforms it. Defaults to None.
# subset (str) – The subset to use, one of all or the keys in cifarc_subsets.
# severity (int) – The severity of the corruption, between 1 and 5.
# download (bool, optional) – If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Defaults to False.
class CIFAR100C(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = f"{source.source.path_config.REPOSITORY_ROOT}/data",
        transform=None,
        target_transform=None,
        subset: str = "brightness",
        severity: int = 1,
        train: bool = False,
        download: bool = False,
    ):
        """
        CIFAR100C dataset https://zenodo.org/records/3555552
        from Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.

        The first 10,000 images in each .npy are the test set images corrupted at severity 1,
        and the last 10,000 images are the test set images corrupted at severity five.



        Allowed perturbation types are:
            brightness, contrast, defocus_blur, elastic_transform, fog, frost, gaussian_blur,
            gaussian_noise, glass_blur, impulse_noise, jpeg_compression, motion_blur, pixelate,
            saturate, shot_noise, snow, spatter, speckle_noise, zoom_blur

        This dataset does not contain train images, so it train is set to True it will return empty dataset.

        """
        if severity < 1 or severity > 5:
            raise ArgumentError(
                "Variable severity can not be smaller the 1 or bigger then 5."
            )

        if train:
            self.dataset = []
            return

        path_to_data_folder = root
        if subset not in ALLOWED_PERTURBATION_TYPES:
            raise RuntimeError(
                f"{subset} is not an allowed perturbatino type."
                f"Allowed types are: {', '.join(ALLOWED_PERTURBATION_TYPES)}."
            )

        self.path_to_folder_with_files = path_to_data_folder
        is_archive_present, is_data_present = self.is_archive_or_data_present(
            path_to_download=path_to_data_folder
        )
        if is_archive_present or is_data_present:
            LOGGER.info(
                f"Downloaded archive or data folder is found at {root}."
                "Skipping downloading."
            )

        if not is_archive_present and not is_data_present:
            if not download:
                raise RuntimeError(
                    f"Folder CIFAR100_C with data or zip can not be found in {path_to_data_folder}.\n"
                    "Most likely it was never downloaded.\n"
                    "To download data initialize CIFAR100C class with download=True."
                )
            else:
                path_to_archive = self.download_cifar_100_c(
                    path_to_download=path_to_data_folder
                )
                self.extract_cifar_100_c(
                    path_to_archive=path_to_archive,
                    path_to_folder_to_extract_archive=path_to_data_folder,
                )

        elif is_archive_present and not is_data_present:
            self.extract_cifar_100_c(
                path_to_archive=f"{path_to_data_folder}/cifar-100-c.tar",
                path_to_folder_to_extract_archive=path_to_data_folder,
            )

        tensor_indexes_start = (severity - 1) * 10000
        tensor_indexes_end = (severity) * 10000

        path_to_corrupted_data_tensor = (
            f"{path_to_data_folder}/CIFAR-100-C/{subset}.npy"
        )
        path_to_labels_tensor = f"{path_to_data_folder}/CIFAR-100-C/labels.npy"
        if not os.path.exists(path_to_corrupted_data_tensor):
            raise RuntimeError(
                f"Something went wrong, data at {path_to_corrupted_data_tensor} is missing."
            )

        # First 10000 are test samples corrupted with level 1
        # Last 10000 are test samples corrupted with level 5
        # Train samples are from 10000 to 40000
        corrupted_data_tensor = torch.from_numpy(
            numpy.load(path_to_corrupted_data_tensor)
        )[tensor_indexes_start:tensor_indexes_end]
        labels_tensor = torch.from_numpy(numpy.load(path_to_labels_tensor))[
            tensor_indexes_start:tensor_indexes_end
        ]

        # It is loaded as int8 [0, 255]
        corrupted_data_tensor = corrupted_data_tensor.float() / 255

        # It has dimension of [batch_size, H, W, C],
        # but for transforms to work we need C, H, W, so we permute
        corrupted_data_tensor = corrupted_data_tensor.permute(0, 3, 1, 2)

        self.dataset = torch.utils.data.TensorDataset(
            corrupted_data_tensor, labels_tensor
        )

        self.transform = transform
        self.target_transform = target_transform

    def is_archive_or_data_present(self, path_to_download: str) -> tuple[bool, bool]:
        return (
            os.path.exists(f"{path_to_download}/cifar-100-c.tar"),
            os.path.exists(f"{path_to_download}/CIFAR-100-C"),
        )

    def show_progress_bar(self, block_num: int, block_size: int, total_size: int):
        percentage_as_float = str(round(block_num * block_size / total_size * 100, 1))
        LOGGER.info(f"{str(percentage_as_float)} %\r")

    def download_cifar_100_c(self, path_to_download: str = "./data") -> str:
        file_url = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
        filename = "cifar-100-c.tar"
        local_path_to_file = f"{path_to_download}/{filename}"
        LOGGER.info(f"Downloading {file_url} to {path_to_download}/{filename}")

        urllib.request.urlretrieve(
            url=file_url, filename=local_path_to_file, reporthook=self.show_progress_bar
        )

        return local_path_to_file

    def extract_cifar_100_c(
        self, path_to_archive: str, path_to_folder_to_extract_archive: str
    ) -> bool:
        LOGGER.info(
            f"Extracting {path_to_archive} to {path_to_folder_to_extract_archive}"
        )
        tar_file = tarfile.open(path_to_archive)
        tar_file.extractall(path_to_folder_to_extract_archive)
        return True

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.dataset)
