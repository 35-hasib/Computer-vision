import cv2
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
import os
import pickle

# === Step 1: Data Preprocessing ===
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

# === Step 2: Feature Extraction ===
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

# === Step 3: Save Background Features (Run Once) ===

# === Step 4: Load Background Features ===
def load_background_features(save_file='background_features.npy'):
    background_features = np.load(save_file)
    print(f"Loaded {len(background_features)} background feature vectors.")
    return background_features

# === Step 5: Dimensionality Reduction with PLS ===
def pls_reduction(X, y, n_components=None):
    if n_components is None:
        n_components = min(11, X.shape[0], X.shape[1])

    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y.reshape(-1, 1))  # Reshape y to a column vector
    return pls

# === Step 6: One-Shot Similarity Model ===
def one_shot_similarity(features1, features2, background_features):
    # Prepare training data
    y = np.array([1] + [-1] * len(background_features))
    X = np.vstack([features1, background_features])

    # Train PLS model
    pls = pls_reduction(X, y)

    # Predict similarity score for features2
    score = pls.predict(features2.reshape(1, -1))

    return score[0][0]  # Return scalar similarity score

# === Step 7: Face Verification Function ===
def face_verification(image1, image2, background_features, threshold=0.5):
    # Extract features for both query images
    features1 = extract_features(image1)
    features2 = extract_features(image2)

    # Compute similarity scores both ways for robustness
    score1 = one_shot_similarity(features1, features2, background_features)
    score2 = one_shot_similarity(features2, features1, background_features)

    avg_score = (score1 + score2) / 2
    print(f"Similarity scores: {score1:.4f}, {score2:.4f}, average: {avg_score:.4f}")

    return avg_score > threshold

# === Step 8: Main Script ===
if __name__ == "__main__":
    try:
        # === 1. Preprocess query images ===
        image1_path = "data_set/p3/p31.jpeg"
        image2_path = "data_set/p3/p31.jpeg"

        image1 = preprocess_image(image1_path)
        image2 = preprocess_image(image2_path)

        
        # === 3. Load precomputed background features ===
        background_features = load_background_features('background_features.npy')

        # === 4. Perform face verification ===
        result = face_verification(image1, image2, background_features)

        print("Are the images of the same person?", "Yes ✅" if result else "No ❌")

    except Exception as e:
        print(f"Error: {e}")
