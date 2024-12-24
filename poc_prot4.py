# Proof of Concept 4

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

# return resized and normalized image
def preprocess_img(path):
    image = cv2.imread(path)
    resized = cv2.resize(image, (224, 224))
    return image / 255.0

def load_data(input_path, label_path):
    images = []
    labels = []
    with open(labels_file, 'r') as file:
        for line in file.readlines():
            name, y_coord = line.strip().split()
            path = os.path.join

if __name__ == "__main__":
    img = preprocess_img("assets/v2/left.JPEG")
    print(img)
    cv2.imwrite("preprocess.JPEG", img)