# import os

# def load_image_paths(dataset_dir):
#     image_paths = []
#     labels = []
#     for person_id, person_folder in enumerate(os.listdir(dataset_dir)):
#         person_dir = os.path.join(dataset_dir, person_folder)
#         if os.path.isdir(person_dir):
#             for image_name in os.listdir(person_dir):
#                 image_path = os.path.join(person_dir, image_name)
#                 image_paths.append(image_path)
#                 labels.append(person_id)
#     return image_paths, labels


# import numpy as np
# import random

# def create_pairs(image_paths, labels):
#     positive_pairs = []
#     negative_pairs = []
    
#     num_people = len(set(labels))
#     people_indices = [np.where(np.array(labels) == i)[0] for i in range(num_people)]
    
#     # Create positive pairs
#     for indices in people_indices:
#         if len(indices) >= 2:
#             for i in range(len(indices) - 1):
#                 for j in range(i + 1, len(indices)):
#                     positive_pairs.append((image_paths[indices[i]], image_paths[indices[j]]))
    
#     # Create negative pairs
#     for i in range(len(image_paths)):
#         for j in range(i + 1, len(image_paths)):
#             if labels[i] != labels[j]:
#                 negative_pairs.append((image_paths[i], image_paths[j]))
    
#     return positive_pairs, negative_pairs

# # Load dataset
# dataset_dir = "D:\github\Computer-vision\data_set"
# image_paths, labels = load_image_paths(dataset_dir)

# # Create pairs
# positive_pairs, negative_pairs = create_pairs(image_paths, labels)

# # Balance the dataset (optional)
# min_pairs = min(len(positive_pairs), len(negative_pairs))
# positive_pairs = positive_pairs[:min_pairs]
# negative_pairs = negative_pairs[:min_pairs]

# # Combine pairs and labels
# all_pairs = positive_pairs + negative_pairs
# all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

# # Shuffle pairs and labels
# combined = list(zip(all_pairs, all_labels))
# random.shuffle(combined)
# all_pairs, all_labels = zip(*combined)

# import cv2

# def preprocess_image(image_path, target_size=(160, 160)):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, target_size)
#     image = image.astype('float32') / 255.0  # Normalize to [0, 1]
#     return image

# def load_and_preprocess_pairs(pairs):
#     pair_images = []
#     for pair in pairs:
#         img1 = preprocess_image(pair[0])
#         img2 = preprocess_image(pair[1])
#         pair_images.append((img1, img2))
#     return pair_images

# # Preprocess pairs
# train_pair_images = load_and_preprocess_pairs(all_pairs)
# train_pair_images = np.array(train_pair_images)
# train_labels = np.array(all_labels)


# print('How was That')


from tensorflow.keras import layers, models
import tensorflow as tf

# Custom layer to compute Euclidean distance
class EuclideanDistance(layers.Layer):
    def call(self, inputs):
        embedding_a, embedding_b = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(embedding_a - embedding_b), axis=-1, keepdims=True))

# Base network (shared weights)
def build_base_network(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128)  # Embedding vector
    ])
    return model

# Siamese Network
def build_siamese_network(input_shape):
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    base_network = build_base_network(input_shape)
    
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    # Compute the Euclidean distance using the custom layer
    distance = EuclideanDistance()([embedding_a, embedding_b])
    
    # Output a similarity score (e.g., sigmoid for binary classification)
    output = layers.Dense(1, activation='sigmoid')(distance)
    
    siamese_network = models.Model(inputs=[input_a, input_b], outputs=output)
    return siamese_network

# Build the model
input_shape = (160, 160, 3)  # Example input shape
siamese_model = build_siamese_network(input_shape)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
siamese_model.summary()


print('Hasib')