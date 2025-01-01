# Proof of Concept 2

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

SIZE = 224

# mask and resize image
def preprocess_img(path):
    image = cv2.imread(path)
    height, _, _ = image.shape
    resized = cv2.resize(image, (SIZE, SIZE))

    green_lower_bound = np.array([35, 40, 40])
    green_upper_bound = np.array([85, 255, 255])
    img_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, green_lower_bound, green_upper_bound)
    img_mask_3ch = cv2.merge([img_mask, img_mask, img_mask])

    return img_mask_3ch.astype('float32') / 255.0, height

# load normalized images and labels
def load_data(input_path, label_path):
    images = []
    labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            name, value = line.strip().split()
            path = os.path.join(input_path, name)
            image, height = preprocess_img(path)
            images.append(image)
            labels.append(int(value))

    return np.array(images), np.array(labels)

# build feature extraction model
def build():
    base = MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# main
if __name__ == "__main__":
    images, labels = load_data("assets/images/", "assets/labels.txt")
    
    model = build()

    history = model.fit(
        images, labels,
        validation_data=(images, labels),
        epochs=20,
        batch_size=32
    )

    loss, acc = model.evaluate(images, labels)
    print(f"Evaluation = Loss: {loss}, Accuracy: {acc}")

    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype("int32")
    for i, pred in enumerate(predictions):
        print(f"Predicted: {pred[0]}, Actual: {labels[i]}")