import cv2
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor

# Step 1: Data Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
        
    # Resize image (assuming face detection and alignment is already done)
    image = cv2.resize(image, (256, 256))
    
    # Apply Difference of Gaussians (DoG) filtering
    gaussian1 = cv2.GaussianBlur(image, (5, 5), 0)
    gaussian2 = cv2.GaussianBlur(image, (9, 9), 0)
    dog = cv2.subtract(gaussian1, gaussian2)
    
    return dog

# Step 2: Feature Extraction
def extract_features(image):
    # Gabor features
    gabor_features = []
    for theta in range(8):  # 8 orientations
        theta_rad = theta / 8. * np.pi
        for sigma in (1, 3, 5, 7, 9):  # Different scales
            freq = 0.6 / sigma
            gabor_real, gabor_imag = gabor(image, frequency=freq, theta=theta_rad)
            gabor_features.append(np.mean(gabor_real))  # Mean instead of whole array
            gabor_features.append(np.mean(gabor_imag))  # Mean instead of whole array

    # LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize histogram

    # HOG features
    hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)

    # Concatenate all features into a single feature vector
    features = np.hstack([gabor_features, lbp_hist, hog_features])
    return features

# Step 3: Dimensionality Reduction with PLS
def pls_reduction(X, y, n_components=None):
    if n_components is None:
        n_components = min(11, X.shape[0], X.shape[1])  # Automatically adjust to data shape
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y.reshape(-1, 1))  # Reshape y to be a column vector
    return pls

# Step 4: One-Shot Similarity (OSS) Model
def one_shot_similarity(image1, image2, background_images):
    # Extract features for image1 and image2
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    
    # Extract features for background images
    background_features = np.array([extract_features(img) for img in background_images])
    
    # Prepare training data
    y = np.array([1] + [-1] * len(background_features))  # +1 for image1, -1 for background
    X = np.vstack([features1, background_features])
    
    # Train PLS model
    pls = pls_reduction(X, y)
    
    # Project image2 feature onto PLS model to get similarity score
    score = pls.predict(features2.reshape(1, -1))
    
    return score[0][0]  # Return scalar score

# Step 5: Face Verification
def face_verification(image1, image2, background_images, threshold=0.5):
    # Compute similarity scores both ways for robustness
    score1 = one_shot_similarity(image1, image2, background_images)
    score2 = one_shot_similarity(image2, image1, background_images)
    
    avg_score = (score1 + score2) / 2
    print(f"Similarity scores: {score1:.4f}, {score2:.4f}, average: {avg_score:.4f}")
    
    # Decide if the images are of the same person based on threshold
    return avg_score > threshold

# === Example usage ===
try:
    # Preprocess query images
    image1 = preprocess_image("data_set/p3/p31.jpeg")

    image2 = preprocess_image("data_set/p3/p32.jpeg")

    # Preprocess background images (images not of the same person)
    # background_images = [preprocess_image(f"p1{i}.jpeg") for i in range(1, 3)]
    import os
    background_folder = "background_images"  # <-- Replace with your actual path

    # Get image file paths
    all_files = os.listdir(background_folder)
    background_image_paths = [os.path.join(background_folder, file)
                            for file in all_files
                            if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(background_image_paths)} images.")

    # Preprocess each background image
    background_images = [preprocess_image(path) for path in background_image_paths]

    # Perform face verification
    result = face_verification(image1, image2, background_images)

    print("Are the images of the same person?", "Yes ✅" if result else "No ❌")

except Exception as e:
    print(f"Error: {e}")
