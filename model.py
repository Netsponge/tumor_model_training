import torch
import torch.nn as nn
import torch.optim as optim

# Définition du modèle CNN
class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # 224x224 devient 56x56 après 2 maxpools
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Aplatir
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Instancier le modèle
num_classes = 4  # 4 classes : glioma, meningioma, pituitary, no tumor
model = TumorClassifier(num_classes)

print(model)
