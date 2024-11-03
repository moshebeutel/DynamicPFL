from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_checkpoint(path: Path,
                    model: nn.Module,
                    # optimizer: torch.optim.Optimizer,
                    device: torch.device = torch.device('cpu')):
    assert path.is_file(), f'{path} is not a file'
    assert path.suffix in ['.pt', '.pth'], f'Expected suffix pt or pth, got {path.suffix}'
    assert path.exists(), f'{path} does not exist'

    checkpoint = torch.load(path.as_posix(), map_location=device)

    # Restore the model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore other information
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    best_acc = checkpoint['best_acc']
    return model, epoch, loss, acc, best_acc
    # return model, optimizer, epoch, loss, acc, best_acc


def save_checkpoint(path: Path, model: nn.Module, epoch: int,
                    # optimizer: Optional[torch.optim.Optimizer],
                    acc: float, loss: float, best_acc: float) -> None:
    # assert path.parent.exists(), f'{path.parent} does not exist'
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        print(f'Created {path}')
    assert path.suffix in ['.pt', '.pth'], f'Expected suffix pt or pth, got {path.suffix}'

    if path.exists():
        # This should be very rare. There are floats in file name
        path = path.parent / (path.stem + '_' + path.suffix)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'acc': acc,
        'best_acc': best_acc
    }
    torch.save(checkpoint, path.as_posix())


class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class cifar10Net(nn.Module):
    def __init__(self):
        super(cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32 * 4 * 4)
        self.fc2 = nn.Linear(32 * 4 * 4, 32 * 2 * 2)
        self.fc3 = nn.Linear(32 * 2 * 2, 10)

        init_weights(self)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class femnistNet(nn.Module):
    def __init__(self):
        super(femnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class EMGModel(nn.Module):
    def __init__(self, num_features=192, num_classes=100, use_softmax=False):
        super(EMGModel, self).__init__()

        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self._use_softmax = use_softmax

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=1) if self._use_softmax else x
        return x


class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # SVHN has 3 color channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjusted linear layer size
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the digits 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class cifar10NetGPkernel(nn.Module):
    def __init__(self):
        super(cifar10NetGPkernel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32 * 4 * 4)
        self.fc2 = nn.Linear(32 * 4 * 4, 32 * 2)

        init_weights(self)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.reshape(x, (-1, 32 * 4 * 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x