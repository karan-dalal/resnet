import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm.notebook import tqdm

# Hyperparameters
batch_size = 16
data_root = './data/cifar10'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 40_000
val_size = 10_000
num_epochs = 10

# Loading Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform,
)
assert train_size + val_size <= len(dataset), "Trying to sample too many elements!" \
    "Please lower the train or validation set sizes."
train_set, val_set, _ = torch.utils.data.random_split(
    dataset, [train_size, val_size, len(dataset) - train_size - val_size]
)
test_set = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform,
)
classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)

# Data Loaders
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

# Parameter Counter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ResNet Function
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Create a residual block for our ResNet18 architecture.

        Here is the expected network structure:
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=stride
        - batchnorm layer (Batchnorm2D)
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=1
        - batchnorm layer (Batchnorm2D)
        - shortcut layer:
            if either the stride is not 1 or the out_channels is not equal to in_channels:
                the shortcut layer is composed of two steps:
                - conv layer with
                    in_channels=in_channels, out_channels=out_channels, 1x1 kernel, stride=stride
                - batchnorm layer (Batchnorm2D)
            else:
                the shortcut layer should be an no-op

        All conv layers will have a padding of 1 and no bias term. To facilitate this, consider using
        the provided conv() helper function.
        When performing a forward pass, the ReLU activation should be applied after the first batchnorm layer
        and after the second batchnorm gets added to the shortcut.
        '''
        # YOUR CODE HERE
        # TODO: Initialize the block with a call to super and make your conv and batchnorm layers.
        super(ResNetBlock, self).__init__()
        self.conv1 = self.conv(in_channels, out_channels,
                               kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = self.conv(out_channels, out_channels,
                               kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Use some conditional logic when defining your shortcut layer
        # For a no-op layer, consider creating an empty nn.Sequential()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                self.conv(in_channels,
                          out_channels,
                          kernel_size=3, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        # END YOUR CODE

    def conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False)

    def forward(self, x):
        '''
        Compute a forward pass of this batch of data on this residual block.

        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        '''
        # YOUR CODE HERE
        # TODO: Call the first convolution, batchnorm, and activation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # TODO: Call the second convolution and batchnorm

        # Also call the shortcut layer on the original input
        out += self.shortcut(x)
        # TODO: Sum the result of the shortcut and the result of the second batchnorm
        # and apply your activation
        out = F.relu(out)
        return out
        # END YOUR CODE


class ResNet18(nn.Module):
    def __init__(self):
        # Read the following, and uncomment it when you understand it, no need to add more code
        num_classes = 10
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=64, 
                               kernel_size=3,
                               stride=1, 
                               padding=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, stride):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Read the following, and uncomment it when you understand it, no need to add more code
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

# Calculate Loss
@torch.no_grad()
def compute_accuracy(model, val_loader):
    total_correct = 0
    model = model.to(device)
    for inputs, labels in tqdm(val_loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(1)
        correct = (outputs == labels)
        total_correct += correct.sum()
    return total_correct / len(val_loader.dataset)


model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
print('Beginning to train model')
model = model.to(device)
for epoch in range(num_epochs):
    print("Hi")
    total_loss = 0
    start_time = time.perf_counter()
    for inputs, labels in train_loader:
        print("Bpo")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
    end_time = time.perf_counter()
    duration = end_time - start_time
    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)

    print(f'epoch {epoch:2}',
            f'loss: {total_loss:.3f}',
            f'time: {duration:.3f}',
            f'train acc: {train_acc:.4f}',
            f'val acc: {val_acc:.4f}', sep='\t')