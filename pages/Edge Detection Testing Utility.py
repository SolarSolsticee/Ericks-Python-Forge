import streamlit as st
import cv2
import numpy as np
from skimage import filters, feature, color
from PIL import Image


# --- Edge Detection Functions ---
# (These are mostly the same as your original script, but adapted to take an image array directly)

def apply_canny(image, threshold1, threshold2, sigma):
    """Apply Canny edge detection with Gaussian smoothing"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges


def apply_sobel(image, ksize):
    """Apply Sobel edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    return sobel_combined


def apply_laplacian(image, ksize):
    """Apply Laplacian edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian


def apply_prewitt(image):
    """Apply Prewitt edge detection using scikit-image"""
    gray = color.rgb2gray(image)
    edges = filters.prewitt(gray)
    return (edges * 255).astype(np.uint8)


def apply_roberts(image):
    """Apply Roberts edge detection using scikit-image"""
    gray = color.rgb2gray(image)
    edges = filters.roberts(gray)
    return (edges * 255).astype(np.uint8)


def apply_scharr(image):
    """Apply Scharr edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
    scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))
    return scharr_combined


def apply_log(image, sigma):
    """Apply Laplacian of Gaussian edge detection"""
    gray = color.rgb2gray(image)
    # Applying Gaussian filter first, then Laplacian
    blurred = filters.gaussian(gray, sigma=sigma)
    edges = filters.laplace(blurred)
    # Normalize for display
    edges = np.absolute(edges)
    edges = np.uint8((edges / edges.max()) * 255) if edges.max() > 0 else edges
    return edges


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🖼️ Interactive Edge Detection Utility")
st.write("Upload an image and select an edge detection algorithm to see the results in real-time.")

# Sidebar for controls
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    # Load image using PIL and convert to NumPy array
    image = Image.open(uploaded_file).convert("RGB")
    original_image = np.array(image)

    # --- Algorithm Selection ---
    algorithm = st.sidebar.selectbox(
        "Select Edge Detection Algorithm",
        [
            "Canny",
            "Sobel",
            "Laplacian",
            "Prewitt",
            "Roberts",
            "Scharr",
            "Laplacian of Gaussian (LoG)",
            "Compare All",
        ],
    )

    processed_image = None
    params_str = ""

    # --- Main processing and display logic ---
    if algorithm != "Compare All":
        # Display parameter sliders based on selected algorithm
        if algorithm == "Canny":
            st.sidebar.subheader("Canny Parameters")
            t1 = st.sidebar.slider("Lower Threshold", 0, 500, 30)
            t2 = st.sidebar.slider("Upper Threshold", 0, 500, 80)
            sigma = st.sidebar.slider("Sigma (Gaussian Blur)", 0.0, 5.0, 1.0, 0.1)
            processed_image = apply_canny(original_image, t1, t2, sigma)
            params_str = f"Lower Threshold: {t1}, Upper Threshold: {t2}, Sigma: {sigma:.1f}"

        elif algorithm in ["Sobel", "Laplacian"]:
            st.sidebar.subheader(f"{algorithm} Parameters")
            ksize = st.sidebar.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
            if algorithm == "Sobel":
                processed_image = apply_sobel(original_image, ksize)
            else:
                processed_image = apply_laplacian(original_image, ksize)
            params_str = f"Kernel Size: {ksize}"

        elif algorithm == "Laplacian of Gaussian (LoG)":
            st.sidebar.subheader("LoG Parameters")
            sigma_log = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
            processed_image = apply_log(original_image, sigma_log)
            params_str = f"Sigma: {sigma_log:.1f}"

        else:  # For Prewitt, Roberts, Scharr
            if algorithm == "Prewitt":
                processed_image = apply_prewitt(original_image)
            elif algorithm == "Roberts":
                processed_image = apply_roberts(original_image)
            elif algorithm == "Scharr":
                processed_image = apply_scharr(original_image)
            params_str = "No adjustable parameters"

        # Display the results in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.header("Original Image")
            st.image(original_image, use_column_width=True)
        with col2:
            st.header(f"{algorithm} Edges")
            st.image(processed_image, use_column_width=True, clamp=True)
            st.caption(f"**Parameters:** {params_str}")

    else:  # Compare All
        st.header("Comparison of All Edge Detection Methods")
        st.write("Default parameters are used for this comparison.")

        # Display original image at the top
        st.subheader("Original Image")
        st.image(original_image, width=400)
        st.markdown("---")

        # Use columns for a grid layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Canny")
            st.image(apply_canny(original_image, 30, 80, 1.0), use_column_width=True)
            st.caption("thresh1=30, thresh2=80, sigma=1.0")

            st.subheader("Prewitt")
            st.image(apply_prewitt(original_image), use_column_width=True)

        with col2:
            st.subheader("Sobel")
            st.image(apply_sobel(original_image, ksize=3), use_column_width=True)
            st.caption("ksize=3")

            st.subheader("Roberts")
            st.image(apply_roberts(original_image), use_column_width=True)

            st.subheader("Laplacian of Gaussian")
            st.image(apply_log(original_image, sigma=1.4), use_column_width=True)
            st.caption("sigma=1.4")

        with col3:
            st.subheader("Laplacian")
            st.image(apply_laplacian(original_image, ksize=3), use_column_width=True)
            st.caption("ksize=3")

            st.subheader("Scharr")
            st.image(apply_scharr(original_image), use_column_width=True)

else:
    st.info("Please upload an image to get started.")