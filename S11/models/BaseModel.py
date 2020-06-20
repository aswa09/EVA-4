import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.learner import Learner


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # PREP LAYER
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # RESNET BLOCK 1
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # RESNET BLOCK 2
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # MAXPOOL
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )

        # OUTPUT
        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # PREP LAYER
        x = self.input_layer(x)

        # LAYER 1
        x = self.layer1(x)

        # RESNET BLOCK 1
        r1 = self.resblock1(x)
        x = x + r1

        # LAYER 2
        x = self.layer2(x)

        # LAYER 2
        x = self.layer3(x)

        # RESNET BLOCK 2
        r2 = self.resblock2(x)
        x = x + r2

        # MAX POOL
        x = self.pool1(x)
        x = torch.flatten(x, 1)

        # LINEAR
        x = self.linear(x)

        return F.log_softmax(x, dim=-1)

    def fit(self, train_loader, optimizer, criterion, device='cpu', epochs=1, l1_factor=0.0, val_loader=None,
            callbacks=None):
        """Train the model.
        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer for the model.
            criterion (torch.nn): Loss Function.
            device (str or torch.device): Device where the data
                will be loaded.
            epochs (int, optional): Numbers of epochs to train the model. (default: 1)
            l1_factor (float, optional): L1 regularization factor. (default: 0)
            val_loader (torch.utils.data.DataLoader, optional): Validation data
                loader. (default: None)
            callbacks (list, optional): List of callbacks to be used during training.
                (default: None)
            track (str, optional): Can be set to either 'epoch' or 'batch' and will
                store the changes in loss and accuracy for each batch
                or the entire epoch respectively. (default: 'epoch')
        """
        self.learner = Learner(
            self, optimizer, criterion, train_loader, device=device, epochs=epochs,
            val_loader=val_loader, l1_factor=l1_factor, callbacks=callbacks
        )
        self.learner.fit()
