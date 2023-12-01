import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras import regularizers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import densenet

# Tried use mean subtraction, normalization, and standards to scale pixels, 
# however each of these methods affected the colors of the images. Found an 
# alternate approach in the "ImageDataGenerator" function. 

def image_generator1(train_data_dir, test_data_dir, img_width, img_height, batch_size):
    train_imggen = ImageDataGenerator(  
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True, 
        fill_mode = "nearest")

    test_imggen = ImageDataGenerator(rescale=1. / 255)

    test_df = pd.DataFrame({
        'filename': os.listdir(test_data_dir), 'class': 'test_class'})

    train_generator = train_imggen.flow_from_directory(  
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_imggen.flow_from_dataframe(  
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


#This CNN has three convolutiona layers -"Conv2D"- and 
#two fully connected layers -"Dense"

def get_weights(train_generator, test_generator, img_width, img_height, epochs):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs, validation_data=test_generator)

    # In a DenseNet layer, the weights are stored in get_weights()[0], 
    # and the biases are stored in get_weights()[1].
    model_weights = model.get_weights()[0]
    np.save("data/weights.npy", model_weights)

    return model






