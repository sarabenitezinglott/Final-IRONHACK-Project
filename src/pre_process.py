import warnings
warnings.filterwarnings("ignore")

import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras import regularizers 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import densenet, EfficientNetB0

def createfolders(data_path,folder_names):
    data_path = "../Final-IRONHACK-Project/data/original/"
    folder_names= ["train_folder", "val_folder1"]

    for folders in folder_names:
        folder_path = os.path.join(data_path, folders)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder {folders} created at: {folder_path}")
        else: 
            print(f"Folder {folders} already exists at: {folder_path}")

def split_data(data_path):
    all_images = [file for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file))]
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    print(train_images)
    print(val_images)

    train_folder = "../Final-IRONHACK-Project/data/original/train_folder"
    val_folder = "../Final-IRONHACK-Project/data/original/val_folder1"
    
    for image in train_images:
        src_path_t = os.path.join(data_path, image)
        dst_path_t = os.path.join(train_folder, image)
        shutil.move(src_path_t, dst_path_t)

    for image in val_images:
        src_path_v = os.path.join(data_path, image)
        dst_path_v = os.path.join(val_folder, image)
        shutil.move(src_path_v, dst_path_v)

# Tried use mean subtraction, normalization, and standards to scale pixels, 
# however each of these methods affected the colors of the images. Found an 
# alternate approach in the "ImageDataGenerator" function. 


def image_generator(original_data_dir, img_width, img_height, batch_size):
    data_datagen = ImageDataGenerator(
        validation_split=0.2,  
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True, 
        fill_mode="nearest")

    train_generator = data_datagen.flow_from_directory(
        original_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=None, subset="training")

    validation_generator = data_datagen.flow_from_directory(
        original_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=None, subset="validation")

    return train_generator, validation_generator

def plot_augmented_images(train_generator, num_images=5):
    original_images = next(train_generator)
    original_image = original_images[0]

    original_image = np.expand_dims(original_image, axis=0)
    augmented_iterator = train_generator
    augmented_images = [next(augmented_iterator)[0][0].astype(np.uint8) for _ in range(num_images)]

    plt.figure(figsize=(15, 5))

    for i, augmented_image in enumerate(augmented_images):
        plt.subplot(1, num_images + 1, i + 2)
        plt.imshow(augmented_image)
        plt.title(f'Augmented {i + 1}')

    plt.show()




