import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set the batch size for training and testing
BATCH_SIZE = 128

# Define the training dataset
train_file = datasets.MNIST(
    root='/Users/nairabarseghyan/Desktop/cv/Data/train/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

# Define the testing dataset
test_file = datasets.MNIST(
    root='/Users/nairabarseghyan/Desktop/cv/Data/validation/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# Display a sample of the training data
train_data = train_file.data
train_targets = train_file.targets
print(train_data.size())  # [60000, 28, 28]
print(train_targets.size())  # [60000]
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(train_targets[i].numpy())
    plt.axis('off')
    plt.imshow(train_data[i], cmap='gray')
plt.show(block=False)


# Display a sample of the testing data
test_data = test_file.data
test_targets = test_file.targets
print(test_data.size())  # [10000, 28, 28]
print(test_targets.size())  # [10000]
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(test_targets[i].numpy())
    plt.axis('off')
    plt.imshow(test_data[i], cmap='gray')
plt.show(block=False)


# Create DataLoader instances for training and testing
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)