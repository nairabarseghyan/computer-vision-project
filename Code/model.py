import torch.nn as nn


class CNN(nn.Module):
    """
        Initializes the Convolutional Neural Network (CNN) model.

        This model is designed for image classification, specifically for the MNIST dataset.

        Architecture:
        - Two Convolutional Layers with ReLU activation and Max Pooling.
        - One Fully Connected Layer for classification.

        Args:
            None

        Returns:
            None
        """
        
    def __init__(self):
        super(CNN, self).__init__()
        # Define the convolutional layers in a sequential block
        self.conv = nn.Sequential(
            # First Convolutional Layer
            # Input: [BATCH_SIZE, 1, 28, 28]
            # Output: [BATCH_SIZE, 32, 28, 28]
            nn.Conv2d(1, 32, 5, 1, 2),
            
            # Apply Rectified Linear Unit (ReLU) activation
            nn.ReLU(),
            
            # Perform 2x2 Max Pooling
            # Output: [BATCH_SIZE, 32, 14, 14]
            nn.MaxPool2d(2),
            
            
            # Second Convolutional Layer
            # Input: [BATCH_SIZE, 32, 14, 14]
            # Output: [BATCH_SIZE, 64, 14, 14]
            nn.Conv2d(32, 64, 5, 1, 2),
            
            # Apply ReLU activation
            nn.ReLU(),
            
            # Perform 2x2 Max Pooling
            # Output: [BATCH_SIZE, 64, 7, 7]
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)


    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [BATCH_SIZE, 1, 28, 28].

        Returns:
            torch.Tensor: Output tensor representing class predictions.
        """
        
        # Forward pass through the convolutional layers
        x = self.conv(x)
        
         # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Forward pass through the fully connected layer
        y = self.fc(x)
        
        return y
