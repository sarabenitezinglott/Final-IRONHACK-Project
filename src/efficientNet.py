import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import keras
import efficientnet.keras as efn 
import efficientnet.tfkeras
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Model, load_model  
from keras import layers
from keras.preprocessing import image
from keras.layers import Activation, Dense  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import categorical_crossentropy
from keras import regularizers 
import matplotlib.pyplot as plt

"EfficientNet model:"

class EfficientNet:
    def __init__(self):
        self.model = self.efficient_model()

    def efficient_model(self):
        #Image size has to be 224 because we are using the EficcientNet B0 type.
        inputs = layers.Input(shape=(224, 224, 3))
        NUM_CLASSES = 2
        model = EfficientNetB0(include_top=False, input_tensor=inputs, 
                               weights=None,classes=NUM_CLASSES)
    
        model.trainable = False

        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
        model = keras.Model(inputs, outputs, name="EfficientNet")
        
        return model

    def compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)  
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                          metrics=["accuracy"])
        
    def train(self, train_generator, validation_generator, epochs):
        epochs = 1
        verbose = 1
        efficientnet_history = self.model.fit(train_generator,validation_data=validation_generator,
                                              epochs = epochs, verbose = verbose)

        return efficientnet_history
    
    def get_weights(self):
        # In a DenseNet layer, the weights are stored in get_weights()[0], 
        # and the biases are stored in get_weights()[1].
        model_weights = self.model.get_weights()[0]
        np.save("data/Efficient_weights.h5", model_weights)


    def predict_efficientNet(self, validation_generator, class_names):  
        # Assuming `validation_generator` is already set up correctly
        image_batch, classes_batch = next(validation_generator)

        # Preprocess input images
        processed_images = efficientnet.preprocess_input(image_batch.copy())

        # Predict using the preprocessed images
        predicted_batch = self.model.predict(processed_images)

        for k in range(0, image_batch.shape[0]):
            image = image_batch[k]
            pred = predicted_batch[k]
            the_pred = np.argmax(pred)
            predicted = class_names[the_pred]
            val_pred = max(pred)
            the_class = np.argmax(classes_batch[k])
            value = class_names[np.argmax(classes_batch[k])]

            plt.figure(k)
            plt.title('Class: ' + value + ' - ' + 'Prediction ratio of: ' + predicted + '[' + str(val_pred) + ']')
            plt.imshow(image)


    def load_weights(self,weights_path):
        self.model.load_weights(weights_path, allow_pickle=True)

    def predict_efficientNet(self, validation_generator, class_names):
        image_batch, classes_batch = next(validation_generator)
        predicted_batch = self.model.predict(image_batch)
        for k in range(0,image_batch.shape[0]):
            image = image_batch[k]
            pred = predicted_batch[k]
            the_pred = np.argmax(pred)
            predicted = class_names[the_pred]
            val_pred = max(pred)
            the_class = np.argmax(classes_batch[k])
            value = class_names[np.argmax(classes_batch[k])]
        plt.figure(k)
        plt.title( 'Class: ' + value + ' - ' + 'Prediction ratio of: ' + predicted + '[' + str(val_pred) + ']')
        plt.imshow(image)


    def load_and_preprocess_images(image_folder, target_size=(224, 224)):
        image_list = []
        for filename in os.listdir(image_folder):
            img_path = os.path.join(image_folder, filename)
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            image_list.append(img_array)

        return np.array(image_list)

