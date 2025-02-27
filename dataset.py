import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Chemin vers le dossier Training
train_path = "./data/image_dataset/Training"

# Choisir une classe au hasard
class_name = os.listdir(train_path)[2]  # Prend la première classe

# Prendre une image dans cette classe
image_path = os.path.join(train_path, class_name, os.listdir(os.path.join(train_path, class_name))[0])

# Charger et afficher l'image
img = Image.open(image_path)
plt.imshow(img)
plt.axis("off")
plt.title(f"Exemple de {class_name}")
plt.show()



# Définition des transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize les images à 224x224
    transforms.ToTensor(),  # Convertit en tenseur PyTorch
    transforms.Normalize([0.5], [0.5])  # Normalisation des pixels entre -1 et 1
])

# Charger les datasets
train_path = "./data/image_dataset/Training"
test_path = "./data/image_dataset/Testing"

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Vérifier les classes détectées
print("Classes du dataset :", train_dataset.classes)
