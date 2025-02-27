import torch
import torchvision.transforms as transforms
from PIL import Image
from model import TumorClassifier

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # 4 classes : glioma, meningioma, pituitary, no tumor
model = TumorClassifier(num_classes).to(device)
model.load_state_dict(torch.load("tumor_model.pth", map_location=device))
model.eval()

# Définir les classes
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Définir les transformations (comme pour l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    """Prend une image en entrée et prédit la classe"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Ajouter une dimension batch

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()
    
    print(f"Prédiction : {class_names[class_index]} ✅")
    return class_names[class_index]

# Tester avec une image
image_path = "./data/image_dataset/single_prediction/image(1).jpg"
predict_image(image_path)
