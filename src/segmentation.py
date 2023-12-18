import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image_for_segmentation(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.densenet.preprocess_input(img)
    return img

def segment_image(image_path, model):
    img = preprocess_image_for_segmentation(image_path)
    img = np.expand_dims(img, axis=0)

    # Segmentation mask
    predictions = model.predict(img)
    mask = predictions[0][:, :, 0]

    # Binary segmentation
    threshold = 0.5
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

# Function to add segmentation to the image generator
def image_generator_with_segmentation(segmentation_model=None):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "D:/bootcamp/original/try_train/",
        target_size=(500, 500),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    validation_generator = test_datagen.flow_from_directory(
        "D:/bootcamp/original/try_val/",
        target_size=(500, 500),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    if segmentation_model:
        # Add segmentation to the generators
        train_generator.segmentation_masks = [segment_image(image_path, segmentation_model) for image_path in train_generator.filepaths]
        validation_generator.segmentation_masks = [segment_image(image_path, segmentation_model) for image_path in validation_generator.filepaths]

    return train_generator, validation_generator

# Load the DeepLabv3+ model
segmentation_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
segmentation_model.trainable = False

# Usage example
train_generator, validation_generator = image_generator_with_segmentation(segmentation_model=segmentation_model)

# Access the segmentation masks
first_segmentation_mask = train_generator.segmentation_masks[0]
