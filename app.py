import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
import json

# Load class labels for CIFAR-100
url = "https://raw.githubusercontent.com/yoavram/Deep-Learning-Course/master/data/cifar-100-classes.json"
response = requests.get(url)

if response.status_code == 200:
    classes = response.json()
else:
    classes = {str(i): f"Class {i}" for i in range(100)}  # Default labels if request fails

# Load trained model
model = models.resnet18()
model.fc = torch.nn.Linear(512, 100)  # Adjust for CIFAR-100
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Streamlit UI
st.title("CIFAR-100 Image Classifier 🖼️")
st.write("Upload an image to classify using a ResNet18 model trained on CIFAR-100.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[str(predicted.item())]
    
    st.success(f"Predicted Class: **{predicted_class}** 🎯")
