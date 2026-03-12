import os
import sys
import time
import numpy as np
from PIL import Image
import streamlit as st
import joblib
import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Ensure custom modules can be imported
sys.path.insert(0, os.path.dirname(__file__))
from src.models.train_cnn import CNN, get_val_transforms, IMAGE_SIZE, NUM_CLASSES
from src.models.train_efficientnet import build_efficientnet

# Set Streamlit page configuration
st.set_page_config(
    page_title="Plastic Bottle Classifier",
    page_icon="♻️",
    layout="centered"
)

def load_config(config_path="config.toml"):
    """Load TOML configuration file locally without depending on src.utils."""
    if not os.path.exists(config_path):
        # Fallback if running from a different directory
        config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    
    with open(config_path, "rb") as f:
        return tomllib.load(f)

config = load_config()

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
CLASSES = config["classes"]["names"]

MODELS = {
    "CNN": os.path.join(config["paths"]["model_save_dir"], "best_cnn.pth"),
    "EfficientNet-B0": os.path.join(config["paths"]["model_save_dir"], "best_efficientnet.pth"),
    "Logistic Regression": os.path.join(config["paths"]["model_save_dir"], "best_logistic_regression.pkl"),
    "SVM": os.path.join(config["paths"]["model_save_dir"], "best_svm.pkl")
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_pytorch_model(model_name):
    """Load PyTorch models (CNN, EfficientNet) into memory and cache them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODELS[model_name]
    
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"
        
    try:
        if model_name == "CNN":
            model = CNN(num_classes=NUM_CLASSES)
        elif model_name == "EfficientNet-B0":
            model = build_efficientnet(num_classes=NUM_CLASSES, freeze_backbone=False)
            
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_sklearn_model(model_name):
    """Load Scikit-Learn models (LogReg, SVM) into memory and cache them."""
    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"
        
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def process_image_for_sklearn(pil_image):
    """Process PIL Image for sklearn models (LogReg, SVM)."""
    img_resized = pil_image.resize(IMAGE_SIZE)
    img_normalize = np.asarray(img_resized, dtype=np.float32) / 255.0
    img_flatten = img_normalize.reshape(1, -1)
    return img_flatten


def process_image_for_pytorch(pil_image):
    """Process PIL Image for PyTorch models (CNN, EfficientNet)."""
    val_transform = get_val_transforms(IMAGE_SIZE)
    tensor = val_transform(pil_image)
    return tensor.unsqueeze(0) # Add batch dimension -> (1, 3, 128, 128)


def predict(image, model_name):
    """Run prediction on the image using the selected model."""
    if model_name in ["CNN", "EfficientNet-B0"]:
        model, err = load_pytorch_model(model_name)
        if err: return None, 0.0, err
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = process_image_for_pytorch(image).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx]
        
    else: # LogReg, SVM
        model, err = load_sklearn_model(model_name)
        if err: return None, 0.0, err
        
        X_flat = process_image_for_sklearn(image)
        pred_idx = model.predict(X_flat)[0]
        
        # Get probability if available (SVM might not output proba without specific config, but pipeline usually handles it if enabled)
        try:
            proba = model.predict_proba(X_flat)[0]
            confidence = proba[pred_idx]
        except (AttributeError, RuntimeError):
            # Fallback if probability is not available
            confidence = 1.0 if hasattr(model, 'decision_function') else "N/A"
            
    return CLASSES[pred_idx], confidence, None


# ==========================================
# UI COMPONENTS
# ==========================================
def main():
    st.title("♻️ Plastic Bottle Classifier")
    st.markdown("Upload an image to classify whether it contains a **Plastic Bottle** or **Others**.")
    
    # 1. Model Selection
    st.sidebar.header("Settings")
    
    # Check what models exist
    available_models = [m for m, p in MODELS.items() if os.path.exists(p)]
    
    if not available_models:
        st.error("No trained models found! Please run the training script first.")
        st.stop()
        
    # Default to CNN if available
    default_idx = available_models.index("CNN") if "CNN" in available_models else 0
    selected_model = st.sidebar.selectbox(
        "Choose Model", 
        available_models, 
        index=default_idx
    )
    
    st.sidebar.markdown("""
    ---
    **Model Information:**
    - **CNN**: Custom architecture trained from scratch.
    - **EfficientNet-B0**: Transfer learning with ImageNet weights.
    - **Logistic Regression**: Linear model on flattened pixels.
    - **SVM**: Support Vector Machine on flattened pixels.
    """)

    # 2. Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Display Image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # 3. Analyze Button with Spinner
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing image with {selected_model}..."):
                    # Add artificial delay for UX (to show the spinner clearly)
                    time.sleep(1.0) 
                    
                    # Run Prediction
                    prediction, confidence, err = predict(image, selected_model)
                    
                    if err:
                        st.error(f"Error during prediction: {err}")
                    else:
                        st.success("Analysis Complete!")
                        st.markdown("### Result")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", prediction)
                        with col2:
                            if isinstance(confidence, float):
                                st.metric("Confidence", f"{confidence * 100:.2f}%")
                            else:
                                st.metric("Confidence", str(confidence))
                                
                        if prediction == "Plastic Bottle":
                            st.info("♻️ Please recycle plastic bottles in the correct bin!")
                        else:
                            st.info("🗑️ Please dispose of this item appropriately.")
                            
        except Exception as e:
            st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()
