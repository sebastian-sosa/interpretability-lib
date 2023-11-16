import os
import tarfile
import urllib.request
import zipfile

import torch
from torchvision import datasets, transforms


def load_cifar10(batch_size=32, num_workers=0):
    """Load the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def load_mnist(batch_size=32, num_workers=0):
    """Load the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def load_caltech101(data_dir='./data', batch_size=32, num_workers=0):
    """Load the Caltech-101 dataset."""
    # Define the URL and filename of the dataset
    CALTECH_101_URL = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip'
    filename = CALTECH_101_URL.split('/')[-1]

    # Download the dataset if it doesn't exist
    if not os.path.exists(filename):
        print('Downloading dataset...')
        urllib.request.urlretrieve(CALTECH_101_URL, filename)

    # Extract the zip file if it hasn't been extracted
    if not os.path.exists(os.path.join(data_dir, 'caltech-101')):
        print('Extracting zip file...')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Extract the tar.gz file if it hasn't been extracted
    if not os.path.exists(os.path.join(data_dir, 'caltech-101', '101_ObjectCategories')):
        print('Extracting tar.gz file...')
        with tarfile.open(os.path.join(data_dir, 'caltech-101', '101_ObjectCategories.tar.gz'), 'r:gz') as tar:
            tar.extractall(os.path.join(data_dir, 'caltech-101'))

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    dataset_dir = os.path.join(data_dir, 'caltech-101', '101_ObjectCategories')
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader
