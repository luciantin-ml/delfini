from dolphins_recognition_challenge.datasets import get_dataset, display_batches, ToTensor, Compose
from lib.trafos import *


def get_tensor_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(ColorJitter())
    return Compose(transforms)


def get_data_loader(batch_size=4):

    data_loader, data_loader_test = get_dataset(
        "segmentation", get_tensor_transforms=get_tensor_transforms, batch_size=batch_size
    )

    # display_batches(data_loader, n_batches=4)
    return data_loader, data_loader_test

