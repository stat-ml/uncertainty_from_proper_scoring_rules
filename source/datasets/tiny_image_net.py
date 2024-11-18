import logging

import datasets as hugging_face_datasets
import torch
import torch.utils.data

LOGGER = logging.getLogger(__name__)

class TinyImageNet(torch.utils.data.Dataset):
    def __init__(
        self,
        transform=None,
        target_transform=None,
        train: bool = False,
    ):
        """
        TinyImageNet dataset https://huggingface.co/datasets/zh-plus/tiny-imagenet
        """
        if train:
            self.dataset = hugging_face_datasets.load_dataset(
                "zh-plus/tiny-imagenet",
                split="train"
            )
        else:
            self.dataset = hugging_face_datasets.load_dataset(
                "zh-plus/tiny-imagenet",
                split="valid"
            )

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.dataset[index].values()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.dataset)
