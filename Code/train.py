import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN
from dataset import train_loader, test_loader

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCH = 30
BATCH_SIZE = 128
LR = 1E-3

# Instantiate the CNN model and move it to the appropriate device (GPU or CPU)
model = CNN().to(device)

# Define the optimizer and loss function
optim = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()

def calc(data_loader):
    """
    Calculate loss and accuracy for a given DataLoader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
        tuple: Loss and accuracy.
    """
    
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss += lossf(output, targets)
            correct += (output.argmax(1) == targets).sum()
            total += data.size(0)
    loss = loss.item()/len(data_loader)
    acc = correct.item()/total
    return loss, acc


def show():
    """
    Display training and validation metrics.
    Save the model if the validation accuracy improves.

    Args:
        None

    Returns:
        None
    """
    if epoch == 0:
        global model_saved_list
        global temp
        temp = 0
        
    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}'
    ]
    header_show = ' '.join(header_list)
    print(header_show, end=' ')
    
    # Calculate and display training metrics
    loss, acc = calc(train_loader)
    train_list = [
        f'LOSS: {loss:.4f}',
        f'ACC: {acc:.4f}'
    ]
    train_show = ' '.join(train_list)
    print(train_show, end=' ')
    
    # Calculate and display validation metrics
    val_loss, val_acc = calc(test_loader)
    test_list = [
        f'VAL-LOSS: {val_loss:.4f}',
        f'VAL-ACC: {val_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')
    
    
    # Save the model if validation accuracy improves
    if val_acc > temp:
        model_saved_list = header_list+train_list+test_list
        torch.save(model.state_dict(), 'model.pt')
        temp = val_acc


# Training loop        
for epoch in range(EPOCH):
    start_time = time.time()
    
    # Iterate through batches in the training DataLoader
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        loss = lossf(output, targets)
        acc = (output.argmax(1) == targets).sum().item()/BATCH_SIZE
        loss.backward()
        optim.step()
        
        # Display real-time training information
        print(
            f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
            f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            end='\r'
        )
        
    # Display aggregated training and validation metrics
    show()
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}')
    
# Display and print the best model information
model_saved_show = ' '.join(model_saved_list)
print('| BEST-PERFORMING-MODEL | '+model_saved_show)