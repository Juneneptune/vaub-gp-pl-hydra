import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Download and load the training dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test dataset
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define a more flexible CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10, input_height=8, input_width=8):
        super(CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the feature map after the convolution and pooling layers
        def conv2d_size_out(size, kernel_size = 3, stride = 1, padding = 1):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1
        
        def pool2d_size_out(size, kernel_size = 2, stride = 2, padding = 0):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1
        
        conv1_out_height = conv2d_size_out(input_height)
        conv1_out_width = conv2d_size_out(input_width)
        
        pool1_out_height = pool2d_size_out(conv1_out_height)
        pool1_out_width = pool2d_size_out(conv1_out_width)
        
        conv2_out_height = conv2d_size_out(pool1_out_height)
        conv2_out_width = conv2d_size_out(pool1_out_width)
        
        pool2_out_height = pool2d_size_out(conv2_out_height)
        pool2_out_width = pool2d_size_out(conv2_out_width)
        
        conv3_out_height = conv2d_size_out(pool2_out_height)
        conv3_out_width = conv2d_size_out(pool2_out_width)
        
        pool3_out_height = pool2d_size_out(conv3_out_height)
        pool3_out_width = pool2d_size_out(conv3_out_width)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * pool3_out_height * pool3_out_width, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Define a simple ResNet-like architecture for the classifier
# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN, self).__init__()
#         # Convolutional layer (sees 1x16x16 image tensor)
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.dropout = nn.Dropout(0.25)
#         # Fully connected layer
#         self.fc1 = nn.Linear(128 * 2 * 2, 256)  # assuming the input is (1, 16, 16)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 2 * 2)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

