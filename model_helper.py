# model_helper.py

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

model = None
class_names = ["Front Breakage", "Front Crushed", "Front Normal", "Rear Breakage", "Rear Crushed", "Rear Normal"]

class CarClassifierResNet(nn.Module):
  def __init__(self, num_classes, dropout_rate=0.5):
    super().__init__()
    self.model = models.resnet50(weights='DEFAULT')

    for param in self.model.parameters():
      param.requires_grad = False

    for param in self.model.layer4.parameters():
      param.requires_grad = True

    in_features = self.model.fc.in_features

    self.model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )

  def forward(self, x):
    x = self.model(x)
    return x

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0) # preprocess(image) returns a tensor of shape [3, 224, 224], unsqueeze(0) adds a batch dimension to make it [1, 3, 224, 224] because the model expects a batch of images.


    global model

    if model is None:
        model = CarClassifierResNet(num_classes=6)
        model.load_state_dict(torch.load("model/saved_model_final.pth", map_location=torch.device('cpu')))
        model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        return class_names[class_idx]


