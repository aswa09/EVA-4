import torch
import torchvision
import torchvision.transforms as transforms
from albumentation_transforms import albumentations_train_transforms
import numpy as np


## Get Train and Test data

def get_train_test(args):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    train_transform = albumentations_train_transforms(mean, std, p=1.0)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )])

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(args.seed)

    if cuda:
        torch.cuda.manual_seed(args.seed)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.nworkers,
                           pin_memory=True) if cuda else dict(shuffle=args.shuffle,
                                                              batch_size=args.batch_size)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader, test_transform