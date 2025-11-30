import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm

# ---------------------------
# Load class names
# ---------------------------
import json

# You MUST save class names from training:
# Example code: 
# import json
# json.dump(train_dataset._data.classes, open("classes.json", "w"))
#
# Then place classes.json beside app.py
class_names = json.load(open("classes.json"))

# ---------------------------
# Model definition
# ---------------------------
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------
# Load model
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCardClassifier(num_classes=53)
model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------------------
# Prediction Function
# ---------------------------
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    return probs

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🃏 Playing Card Classifier (EfficientNet-B0, PyTorch)")
st.write("Upload a playing card image to classify it!")

uploaded_file = st.file_uploader("Upload card image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    probs = predict(img)

    # Top-5 predictions
    top_k = 5
    top_indices = np.argsort(probs)[-top_k:][::-1]

    st.subheader("🔮 Top Predictions")
    for idx in top_indices:
        st.write(f"**{class_names[idx]}** — {probs[idx]*100:.2f}%")

    # Bar chart
    st.bar_chart({class_names[i]: probs[i] for i in top_indices})