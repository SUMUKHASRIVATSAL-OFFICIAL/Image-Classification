import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations (normalize images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

print(f"Train Dataset Size: {len(trainset)}")
print(f"Test Dataset Size: {len(testset)}")



import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Load a pretrained ResNet18 model and modify for CIFAR-100
model = models.resnet18(weights="IMAGENET1K_V1")  # Use pretrained weights
model.fc = nn.Linear(512, 100)  # Change output layer for 100 classes

device = torch.device("cpu")  # Use CPU
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 5  # Adjust based on speed

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print loss every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

print("Training complete!")

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully as 'model.pth' ✅")

correct = 0
total = 0
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
