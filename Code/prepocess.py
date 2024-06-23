import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt    
IMG_HEIGHT = 224
IMG_WIDTH =224
CHANNELS = 3

class_labels = {
    'spider' : 0,
    'horse': 1,
    'butterfly': 2,
    'dog':3,
    'chicken' : 4,
    'elephant':5,
    'sheep': 6,
    'cow' : 7,
    'squirrel': 8,
    'cat': 9
}

def load_data(paths, max_images=10, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    labels = []
    folders = os.listdir(paths)
    img_paths = []
    
    for name in folders:
        class_path = os.path.join(paths, name)
        img_names = os.listdir(class_path)
        for img_name in img_names:
            img_paths.append((os.path.join(class_path, img_name), class_labels[name]))

    random.shuffle(img_paths)  # Shuffle to ensure randomness
    img_paths = img_paths[:max_images]  # Limit to max_images

    for img_path, label in img_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def plot_correct_predictions(model, X_test, Y_test, n=20):
    """
    Plot only the correct predictions of the model on the test data.

    Args:
    - model: Trained Keras model.
    - X_test: Test images.
    - Y_test: True labels of test images (not one-hot encoded).
    - class_labels: Dictionary of class labels.
    - n: Number of images to display.
    """
    inverted_classes = {v: k for k, v in class_labels.items()}
    plt.figure(figsize=(5, 5))

    
    # Get predictions
    predictions = np.argmax(model.predict(X_test), axis=1)
    true_labels = Y_test  # Use Y_test directly since it's not one-hot encoded
    correct_indices = [i for i in range(len(predictions)) if predictions[i] == true_labels[i]]
    random_indices = np.random.choice(correct_indices, min(n, len(correct_indices)), replace=False)


    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        pred = predictions[idx]
        actual = true_labels[idx]
        title = f"{inverted_classes[pred]}"
        plt.title(title, color="g", fontsize=8)
        plt.imshow(X_test[idx])

    plt.show()
