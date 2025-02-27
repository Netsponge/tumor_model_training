import torch
import torch.nn as nn
import torch.optim as optim
from model import TumorClassifier
from dataset import train_loader, test_loader

# Vérifier si GPU dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancier le modèle
num_classes = 4
model = TumorClassifier(num_classes).to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Époque {epoch+1}/{num_epochs}, Perte : {running_loss/len(train_loader)}")

print("Entraînement terminé ! 🎉")

correct = 0
total = 0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"✅ Exactitude sur l'ensemble de test : {accuracy:.2f}%")

# Sauvegarde du modèle
torch.save(model.state_dict(), "tumor_model.pth")
print("Modèle sauvegardé sous tumor_model.pth ✅")
