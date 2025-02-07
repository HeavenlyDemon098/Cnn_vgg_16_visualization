import math
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from skimage import transform
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# ----------------------
# Utility Functions
# ----------------------
def tensor_summary(tensor: np.ndarray) -> None:
    """Display shape, min, and max values of a tensor."""
    st.write(f"**Shape:** {tensor.shape} | **Min:** {tensor.min()} | **Max:** {tensor.max()}")

def normalize_feature_map(image: np.ndarray) -> np.ndarray:
    """Normalize feature map for better visualization."""
    image -= image.mean()  # Center at zero
    image /= (image.std() + 1e-5)  # Scale to unit variance
    image *= 64
    image += 128  # Bring to visible range
    return np.clip(image, 0, 255).astype(np.uint8)  # Clip values and convert

def read_layer(model, x, layer_name):
    """Return the activation values for the specified layer."""
   
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    outputs = intermediate_model.predict(x)[0]  # Remove batch dimension
    return outputs

def view_layer(model, x, layer_name, cols=5):
    """Visualize feature maps of a given layer in a structured format."""
    outputs = read_layer(model, x, layer_name)
    num_features = outputs.shape[-1]  # Number of feature maps
    num_rows = np.ceil(num_features / cols).astype(int)  # Auto-adjust rows

    plt.figure(figsize=(15, num_rows * 3))  # Adjust size dynamically

    for i in range(min(num_features, 20)):  # Show only first 20 channels
        plt.subplot(num_rows, cols, i + 1)
        feature_map = normalize_feature_map(outputs[:, :, i])
        plt.imshow(feature_map, cmap="viridis", interpolation="nearest")
        plt.axis("off")
        plt.title(f"Filter {i+1}", fontsize=8)

    plt.tight_layout()
    st.pyplot(plt)

def describe_layer(layer) -> str:
    """
    Returns a description string for a given layer.
    This uses the layer's configuration details.
    """
    desc = f"**Layer Name:** {layer.name}\n\n"
    desc += f"**Layer Type:** {layer.__class__.__name__}\n\n"
    config = layer.get_config()
    if layer.__class__.__name__ == "Conv2D":
        desc += f"This is a convolutional layer with {config.get('filters', 'unknown')} filters, "
        desc += f"kernel size: {config.get('kernel_size', 'unknown')}, activation: {config.get('activation', 'none')}.\n\n"
    elif layer.__class__.__name__ in ["MaxPooling2D", "AveragePooling2D"]:
        desc += f"This is a pooling layer with pool size: {config.get('pool_size', 'unknown')}.\n\n"
    elif layer.__class__.__name__ == "Dense":
        desc += f"This is a fully-connected (dense) layer with {config.get('units', 'unknown')} units, activation: {config.get('activation', 'none')}.\n\n"
    else:
        desc += f"Configuration: {config}\n\n"
    return desc

# ----------------------
# Load VGG16 model (cache the model for performance)
# ----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = VGG16(weights="imagenet")
    return model


model = load_model()

# Get list of layers for visualization
layer_options = [(i, layer.name) for i, layer in enumerate(model.layers)]
layer_dict = {layer.name: layer for layer in model.layers}

# ----------------------
# Streamlit Layout
# ----------------------
st.title("VGG16 Visualization App")
st.write("""
Upload an image, view the VGG16 architecture, and click on a layer to see its description and output feature maps.
""")

# Sidebar: Upload image and select a layer
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("Upload an image (jpg or png)", type=["jpg", "jpeg", "png"])

selected_layer_name = st.sidebar.selectbox(
    "Select a layer to inspect",
    [layer.name for layer in model.layers if "conv" in layer.name],  # Only convolutional layers
    index=0,
    key="selected_layer_name"  # Set a session key
)


# Main: Show model architecture (list of layers)
st.header("VGG16 Architecture")
st.write("Below is the list of layers in VGG16:")
for idx, layer in layer_options:
    st.write(f"**{idx}.** {layer}")

# If an image is uploaded, process and show results.
if uploaded_file is not None:
    # Read image via PIL
    image = Image.open(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # Convert image to numpy array and resize to (224,224)
    image = image.convert("RGB")
    image_np = np.array(image)
    image_resized = transform.resize(image_np, (224, 224), anti_aliasing=True)
    x = image_resized.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # add batch dimension
    x = preprocess_input(x)  # preprocess for VGG16

    # Display summary for the uploaded image.
    st.write("### Image Tensor Summary")
    tensor_summary(x)

    # ----------------------
    # When a layer is selected:
    # ----------------------
    st.header(f"Layer Inspection: {selected_layer_name}")
    layer = layer_dict[selected_layer_name]
    st.markdown(describe_layer(layer))

    # Build an intermediate model to get the output of the selected layer.
    intermediate_model = Model(inputs=model.input, outputs=layer.output)
    activation = intermediate_model.predict(x)

    st.write("#### Layer Output Tensor Summary:")
    tensor_summary(activation)

    # If the activation is 4D, visualize feature maps.
    if activation.ndim == 4:
        st.write("#### Feature Maps Visualization:")
        view_layer(model, x, selected_layer_name)
    else:
        st.write("Layer output is not a 4D feature map. Cannot visualize as images.")
else:
    st.write("Please upload an image using the sidebar to begin.")
