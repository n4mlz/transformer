"""
参考文献
https://take-tech-engineer.com/pytorch-quickstart/
https://dreamer-uma.com/pytorch-mlp-mnist/

"""

import torch
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# numpy配列からdatasetを作る関数を書く
# 自作datasetを作る関数を書く 

def train(model, dataloader, criterion, optimizer, acc=False, device=None):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    model.to(device)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if acc:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="")
        elif batch == len(dataloader) - 1:
            loss, current = loss.item(), size
            print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="")
            print('\r', end='')
    
    train_loss /= num_batches
    correct /= size

    if acc:
        return train_loss, correct
    else:
        return train_loss, None


def valid(model, dataloader, criterion, acc=False, device=None):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            val_loss += criterion(pred, y).item()
            if acc:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    val_loss /= num_batches
    correct /= size
    if acc:
        return val_loss, correct
    else:
        return val_loss, None


def fit(model, train_loader, valid_loader, criterion, optimizer, epochs, acc=False, device=None, call=None):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        _train_loss, _train_acc = train(model, train_loader, criterion, optimizer, acc, device)
        _valid_loss, _valid_acc = valid(model, valid_loader, criterion, acc, device)

        if acc:
            print(f"Epoch {epoch+1:>3} | train_loss: {_train_loss:.5f}, train_acc: {_train_acc:.5f}, val_loss: {_valid_loss:.5f}, val_acc: {_valid_acc:.5f}")
        else:
            print(f"Epoch {epoch+1:>3} | train_loss: {_train_loss:.5f}, val_loss: {_valid_loss:.5f}")
        
        train_loss_list.append(_train_loss)
        valid_loss_list.append(_valid_loss)

        if acc:
            print("="*90)
        else:
            print("="*52)
    
        if call != None:
            call()
    return train_loss_list, valid_loss_list


def show_train_and_valid(train_loss_list, valid_loss_list):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    ax.plot(range(len(valid_loss_list)), valid_loss_list, c='r', label='valid loss')

    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('loss', fontsize='20')
    ax.set_title('training and validation loss', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')

    plt.show()
