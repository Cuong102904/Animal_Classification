import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import random
from prepocess import class_labels
from prepocess import load_data
from prepocess import plot_correct_predictions

inverted_classes = {v: k for k, v in class_labels.items()}

model = load_model('Final\90percentsmodel.keras')
def prepocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img /255.0
    img = np.expand_dims(img, axis=0)
    return img


def random_image():
    # Select 10 random images from the test data
    X_random , Y_random = load_data(paths = '../Data/test', max_images= 20)
    plot_correct_predictions(model, X_random, Y_random, n=20)



def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = prepocess_image(file_path)
        prediction = model.predict(img)
        display_image(file_path, prediction)

# Function to display the image and prediction
def display_image(file_path, prediction):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img  # Keep a reference to avoid garbage collection
    predicted_class = inverted_classes[np.argmax(prediction, axis=1)[0]]  # Use inverted_classes to get class name
    result_label.config(text=f'Prediction: {predicted_class}')  # Update with predicted class name

window = tk.Tk()
window.title("Image Classifier")

window.geometry("400x420")  # Width x Height


# Create and pack widgets
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=(5, 0))

random_button = tk.Button(window, text="Random Image", command=random_image)
random_button.pack(pady=(5, 0))

panel = tk.Label(window)  # Label to display the image
panel.pack(pady=(5, 0))

result_label = tk.Label(window, text="Prediction: ")
result_label.pack(pady=(5, 0))

# Start the Tkinter event loop
window.mainloop()