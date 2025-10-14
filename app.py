import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- 1. Define the Model Architecture ---
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True
        for param in self.cnn.layer3.parameters():
            param.requires_grad = True
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.unsqueeze(1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out

# --- 2. Load the Trained Model ---
@st.cache_resource
def load_model(model_path="best_model.pth"):
    device = torch.device("cpu")
    model = CNN_LSTM(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# --- 3. Define Image Transforms ---
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Prediction Function ---
def predict(image, model, device, class_names):
    image = preprocess_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item() * 100

    return predicted_class_name, confidence, probabilities.cpu().numpy().flatten()

# --- 5. Streamlit App Layout ---
st.set_page_config(page_title="Wall Crack Detection", layout="centered")

st.title("🧱 Wall Crack Detection AI")
st.markdown("Upload an image of a wall to predict if it has cracks and get suggestions.")

class_names = ["Cracked", "Non-cracked"]
model, device = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    st.subheader("Prediction:")
    with st.spinner("Analyzing image..."):
        predicted_class, confidence, all_probabilities = predict(image, model, device, class_names)

        if predicted_class == "Cracked":
            st.error(f"Prediction: **{predicted_class}**")
            st.markdown(f"Confidence: **{confidence:.2f}%**")

            # --- Suggestions and Causes ---
            st.markdown("### 🔍 Possible Causes of Air Cracks:")
            st.markdown("""
            - **Thermal expansion and contraction** due to temperature changes.
            - **Poor construction practices** or use of substandard materials.
            - **Shrinkage** during curing of concrete or plaster.
            - **Structural movement** or settlement of the foundation.
            - **Moisture ingress** leading to material degradation.
            """)

            st.markdown("### 🛠️ Suggested Actions:")
            st.markdown("""
            - Conduct a **detailed structural inspection**.
            - Apply **crack sealants** or **epoxy injections** for minor cracks.
            - Improve **drainage and waterproofing** to prevent moisture-related damage.
            - Consult a **civil engineer** for persistent or widening cracks.
            - Monitor the crack over time using **periodic imaging**.
            """)

        else:
            st.success(f"Prediction: **{predicted_class}**")
            st.markdown(f"Confidence: **{confidence:.2f}%**")
            st.markdown("✅ No visible cracks detected. Keep monitoring periodically.")

        st.markdown("---")
        st.subheader("All Class Probabilities:")
        prob_df = {
            "Class": class_names,
            "Probability": [f"{p*100:.2f}%" for p in all_probabilities]
        }
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.markdown("Developed with ❤️ for crack detection and structural safety.")