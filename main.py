import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm.notebook import tqdm

# Set hyperparameters
batch_size = 16
data_root = './data/cifar10'
train_size = 40_000
val_size = 10_000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load in data
transform = transforms.Compose([
    transforms.ToTensor(),
    # scales pixel values to range [-1, 1]
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

# Create data loads
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


# Set loss function
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

# Set training function
def train(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    print('Beginning to train model')
    model = model.to(device)
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        start_time = time.perf_counter()
        for inputs, labels in tqdm(train_loader, leave=False):
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

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create residual block for ResNet18 architecture
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__() # Module initialization
        self.conv1 = self.conv(in_channels, out_channels, kernel_size=3, stride=stride) # First conv layer
        self.bn1 = nn.BatchNorm2d(out_channels) # Batchnorm layer
        self.conv2 = self.conv(out_channels, out_channels, kernel_size=3, stride=1) # Second conv layer
        self.bn2 = nn.BatchNorm2d(out_channels) # Batchnorm layer

        # Transforms input to match output if they don't have the same dimensions for a residual connection 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                self.conv(in_channels,
                     out_channels,
                     kernel_size=3, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
    
    # Conv method to set padding equal to 1 and no bias
    def conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False)

    # Forward pass on this residual block
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Calling first conv, batchnorm, and activation
        out = self.bn2(self.conv2(out)) # Calling second conv and batchnorm
        out += self.shortcut(x) # Calling shortcut method and adding result
        out = F.relu(out) # Final activation
        return out

class ResNet18(nn.Module):
    def __init__(self):
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
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

num_epochs = 10
resnet = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)

train(resnet, train_loader, val_loader, num_epochs, criterion, optimizer)