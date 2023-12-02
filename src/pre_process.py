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

def split_images(dataset_path,train_path,test_path):
    all_images = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]

    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

    for image in train_images:
        shutil.copy(os.path.join(dataset_path, image), os.path.join(train_path, image))

    for image in test_images:
        shutil.copy(os.path.join(dataset_path, image), os.path.join(test_path, image))


# Tried use mean subtraction, normalization, and standards to scale pixels, 
# however each of these methods affected the colors of the images. Found an 
# alternate approach in the "ImageDataGenerator" function. 
   
def image_generator1(train_data_dir, test_data_dir, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(
        validation_split = 0.2,  
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True, 
        fill_mode = "nearest")

    test_imggen = ImageDataGenerator(rescale=1. / 255)

    test_df = pd.DataFrame({
        'filename': os.listdir(test_data_dir), 'class': 'test_class'})

    train_generator = data_generator.flow_from_directory(  
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = data_generator.flow_from_dataframe(  
        test_df,
        directory=test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, test_generator

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




