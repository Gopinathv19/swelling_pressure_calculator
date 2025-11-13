import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ---------------------------
# 🧩 TabTransformer Definition
# ---------------------------
class TabTransformer(nn.Module):
    def __init__(self, num_features, embed_dim=64, num_heads=4, num_layers=3):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ---------------------------
# 🧠 Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = TabTransformer(num_features=7, embed_dim=64, num_heads=4, num_layers=3)
    checkpoint = torch.load("tab2.pt", map_location=torch.device('cpu'))
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

model = load_model()
device = torch.device("cpu")

# ---------------------------
# 🔍 Prediction Function
# ---------------------------
def predict_swelling_pressure(input_features):
    """
    Predict swelling pressure from soil properties.
    input_features: list or array of 7 numeric values in this order:
        [Gs, OMC, LL, PL, PI, Clay Content, SL]
    """
    with torch.no_grad():
        x = np.array(input_features).reshape(1, -1)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        pred = model(x_tensor).cpu().item()
        return pred

# ---------------------------
# 🌍 Streamlit UI
# ---------------------------
st.title("🌍 Swelling Pressure Prediction using TabTransformer")
st.write("Enter soil parameters below to predict the swelling pressure (kN/m²).")

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    gs = st.number_input("Specific Gravity (Gs)", min_value=2.0, max_value=3.0, step=0.01)
    omc = st.number_input("Optimum Moisture Content (OMC) (%)", min_value=0.0, max_value=100.0, step=0.1)
    ll = st.number_input("Liquid Limit (LL) (%)", min_value=0.0, max_value=100.0, step=0.1)

with col2:
    pl = st.number_input("Plastic Limit (PL) (%)", min_value=0.0, max_value=100.0, step=0.1)
    pi = st.number_input("Plasticity Index (PI) (%)", min_value=0.0, max_value=100.0, step=0.1)

with col3:
    clay = st.number_input("Clay Content (%)", min_value=0.0, max_value=100.0, step=0.1)
    sl = st.number_input("Shrinkage Limit (SL) (%)", min_value=0.0, max_value=100.0, step=0.1)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict Swelling Pressure"):
    input_features = [gs, omc, ll, pl, pi, clay, sl]  # User inputs
    predicted_sp = predict_swelling_pressure(input_features)
    st.success(f"**Predicted Swelling Pressure:** {predicted_sp:.2f} kN/m²")
