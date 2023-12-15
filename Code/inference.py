import os
import torch
from PIL import Image
from model import CNN
import matplotlib.pyplot as plt
from torchvision import transforms
import time

# Set the path to the test data
path = '/Users/nairabarseghyan/Desktop/cv/Data/test/'

# Lists to store images and labels
imgs = []
labels = []

# Read and preprocess each image in the test directory
for name in sorted(os.listdir(path)):
    img = Image.open(path+name).convert('L')
    img = transforms.ToTensor()(img)
    imgs.append(img)
    labels.append(int(name[0]))
    
# Convert the list of images to a torch tensor
imgs = torch.stack(imgs, 0)


# Instantiate the CNN model
model = CNN()

# Load the trained model parameters
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Make predictions on the test images
with torch.no_grad():
    output = model(imgs)

# Extract predicted labels
pred = output.argmax(1)

# Convert true labels to torch tensor
true = torch.LongTensor(labels)
print(pred)
print(true)

# Display predicted and true labels along with images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title(f'pred {pred[i]} | true {true[i]}')
    plt.axis('off')
    plt.imshow(imgs[i].squeeze(0), cmap='gray')
    
# Save and display the figure
plt.savefig('test.png')
plt.show(block=False)