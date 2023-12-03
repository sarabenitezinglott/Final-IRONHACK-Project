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

def train_valid_split(df):
    train = df.drop(columns= ["Y"])
    valid = df["Y"]
    train_x, valid_x, train_y, valid_y = train_test_split(train,valid, test_size=0.2, stratify=df['label'], random_state=50)
    return train_x, valid_x, train_y, valid_y

def createfolders(data_path,folder_names):
    for folder in folder_names:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder {folder} created at: {folder_path}")
        else: 
            print(f"Folder {folder} already exists at: {folder_path}")

def move_images(df, folder_path):
    for _, i in df.iterrows():
        src = i['file_path']
        shutil.move(src, folder_path)

def createsubfolders(data_path,folder_names):
    for folder in folder_names:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder {folder} created at: {folder_path}")
        else: 
            print(f"Folder {folder} already exists at: {folder_path}")

def images_class(df, folder_path_CE, folder_path_LAA):
    for _, i in df.iterrows():
        if i["label"] == "CE":
            src = i['new_file_path']
            shutil.move(src, folder_path_CE)
        else:
            src = i['new_file_path']
            shutil.move(src, folder_path_LAA)

# Tried use mean subtraction, normalization, and standards to scale pixels, 
# however each of these methods affected the colors of the images. Found an 
# alternate approach in the "ImageDataGenerator" function. 

def image_generator(train_dir, val_dir, img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True, 
        fill_mode="nearest")
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=None, subset="training")

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
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





