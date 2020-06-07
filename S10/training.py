import torch
from tqdm import tqdm
import sys


##Training

def train(model, device, train_loader, optimizer, epoch, criterion, l1, l2):
    train_losses = []
    train_acc = []

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    temp_loss = 0
    temp_acc = 0
    temp_proc = 0


    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # loss
        # loss = F.nll_loss(y_pred, target)
        loss = criterion(y_pred, target)
        if l1 > 0:
            loss_l1 = 0
            for param in model.parameters():
                loss_l1 += torch.norm(param, 1)
            loss += l1 * loss_l1
        if l2 > 0:
            loss_l2 = 0
            for param in model.parameters():
                loss_l2 += torch.norm(param, 2)
            loss += l2 * loss_l2

        temp_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        temp_acc += correct
        temp_proc += processed

    temp_loss /= len(train_loader)
    temp_acc = 100 * temp_acc / temp_proc
    train_losses.append(temp_loss)
    train_acc.append(temp_acc)

    return train_losses, train_acc