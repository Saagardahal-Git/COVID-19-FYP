import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("covid19_cnn_model_debug.h5")

# -----------------------------
# 1. Image preprocessing
# -----------------------------
def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 1)
    return img

# -----------------------------
# 2. Check if the image looks like a chest X-ray
# -----------------------------
def is_chest_xray(image):
    """
    Heuristic check to verify if the image appears to be a chest X-ray.
    - X-rays are mostly grayscale.
    - They have mid-range contrast.
    - There is a lot of edge detail (bones, lungs).
    """

    # Convert to grayscale and normalize
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_norm = gray / 255.0

    # Compute image statistics
    mean_intensity = np.mean(gray_norm)
    std_intensity = np.std(gray_norm)
    
    # Compute color variation (X-rays have very little color)
    color_diff = np.mean(np.abs(image[:, :, 0] - image[:, :, 1])) + \
                 np.mean(np.abs(image[:, :, 1] - image[:, :, 2]))

    # Compute edge strength (X-rays have visible lung/bone edges)
    edges = cv2.Canny((gray_norm * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Define heuristic thresholds
    is_graylike = color_diff < 10         # almost no color variation
    is_balanced_brightness = 0.2 < mean_intensity < 0.8
    is_reasonable_contrast = std_intensity > 0.05
    has_edges = edge_density > 0.02       # at least some visible structures

    return all([is_graylike, is_balanced_brightness, is_reasonable_contrast, has_edges])

# -----------------------------
# 3. Prediction function
# -----------------------------
def predict_covid(image):
    if not is_chest_xray(image):
        return "⚠️ This image does not appear to be a chest X-ray. Please upload a valid X-ray image."

    img = preprocess(image)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100
    label = "COVID-19 Positive" if class_idx == 1 else "Normal"
    return f"{label} ({confidence:.2f}% confidence)"

# -----------------------------
# 4. Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict_covid,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩻 COVID-19 Chest X-ray Detection",
    description="Upload a chest X-ray image to detect COVID-19 using a CNN model. Non–X-ray images will be automatically rejected."
)

# -----------------------------
# 5. Launch App
# -----------------------------
if __name__ == "__main__":
    interface.launch()
