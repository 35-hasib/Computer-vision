import cv2
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
import os
import pickle

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Resize the image (assuming face detection and alignment is already done)
    image = cv2.resize(image, (256, 256))

    # Apply Difference of Gaussians (DoG) filtering
    gaussian1 = cv2.GaussianBlur(image, (5, 5), 0)
    gaussian2 = cv2.GaussianBlur(image, (9, 9), 0)
    dog = cv2.subtract(gaussian1, gaussian2)

    return dog


def extract_features(image):
    # Gabor features
    gabor_features = []
    for theta in range(8):  # 8 orientations
        theta_rad = theta / 8. * np.pi
        for sigma in (1, 3, 5, 7, 9):  # Different scales
            freq = 0.6 / sigma
            gabor_real, gabor_imag = gabor(image, frequency=freq, theta=theta_rad)
            gabor_features.append(np.mean(gabor_real))  # Mean of real part
            gabor_features.append(np.mean(gabor_imag))  # Mean of imaginary part

    # LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize histogram

    # HOG features
    hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)

    # Concatenate all features into one vector
    features = np.hstack([gabor_features, lbp_hist, hog_features])
    return features


def save_background_features(background_folder, save_file='background_features.npy'):
    all_files = os.listdir(background_folder)
    background_image_paths = [
        os.path.join(background_folder, file)
        for file in all_files
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Extracting features from {len(background_image_paths)} background images...")

    background_features = [
        extract_features(preprocess_image(path)) for path in background_image_paths
    ]

    np.save(save_file, background_features)
    print(f"Background features saved to {save_file}.")


# === 2. (Optional) Save background features once ===
        # Run this ONCE to generate the background feature file
background_folder = "background_images"
save_background_features(background_folder, save_file='background_features.npy')
