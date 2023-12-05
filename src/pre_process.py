import warnings
warnings.filterwarnings("ignore")

import numpy as np
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
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet

# Data split, so we have train and validation datasets out of the train folder.
def train_valid_split(df):
    train = df.drop(columns= ["Y"])
    valid = df["Y"]
    train_x, valid_x, train_y, valid_y = train_test_split(train,valid, test_size=0.2, stratify=df['label'], random_state=50)
    return train_x, valid_x, train_y, valid_y

# Creation of new folders, necesary for ImageDataGenerator image detection
def createfolders(data_path,folder_names):
    for folder in folder_names:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder {folder} created at: {folder_path}")
        else: 
            print(f"Folder {folder} already exists at: {folder_path}")

# Move images using shutil library
def move_images(df, folder_path):
    for _, i in df.iterrows():
        src = i['file_path']
        shutil.move(src, folder_path)

# Create subfolders for each label, inside of each folder
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

# An image generator for each CNN. Tried to join this 
# function using conditionals but it was giving errors constantly. 

def image_generator():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "D:/bootcamp/original/try_train/",  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=5,
        class_mode='binary')  
    
    validation_generator = test_datagen.flow_from_directory(
        "D:/bootcamp/original/try_val/",
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary')
    
    return train_generator, validation_generator

def image_generator_for_B0():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "D:/bootcamp/original/try_train/",  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=5,
        class_mode='binary', shuffle=False)  
    
    validation_generator = test_datagen.flow_from_directory(
        "D:/bootcamp/original/try_val/",
        target_size=(224, 224),
        batch_size=5,
        class_mode='binary', shuffle=False)
    
    return train_generator, validation_generator

# Plot the images to see how it looks like the rescaling and standarization of pixels.

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

# Tried to use ImageDataGenerator for the test images, BUT as it has no subfolders (images are 
# not labeled), it detected 0 images. So I preprocessed the images in each NN notebook



