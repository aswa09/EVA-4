import torch
import torchvision
import torchvision.transforms as transforms
from albumentation_transforms import albumentations_train_transforms
from datasets.tinyimagenet import TinyImageNet
## Get Train and Test data

def get_train_test(args):

    train_transform = albumentations_train_transforms(args, p=1.0)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std=args.std
        )])

    # CUDA?
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

    if args.dataset=='cifar':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
												download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
											   download=True, transform=test_transform)
        dataset=None
    elif args.dataset=='tinyimagenet':
        dataset=TinyImageNet(args,train_transform,test_transform)
        trainset=dataset.train_data
        testset=dataset.val_data

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader, dataset