import os
import torch
from torchvision import datasets, transforms


def dataset(input_size, data_augmentation=[], download=True):
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.Resize(input_size[2:]),
        transforms.ToTensor()
    ]+data_augmentation), download=download)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.Resize(input_size[2:]),
        transforms.ToTensor()
        ]), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=input_size[0],
                                               shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=input_size[0],
                                              shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    return train_loader, test_loader
