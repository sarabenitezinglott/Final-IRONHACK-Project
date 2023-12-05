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
    
    def get_weights(self,weights_path):
        self.model.save_weights(weights_path)

    def predict_efficientNet(self, validation_generator, class_names):  
        image_batch,classes_batch = next(validation_generator)
        processed_images = efficientnet.keras.preprocess_input(image_batch.copy())
        predicted_batch = self.model.predict(processed_images)


        fig, axes = plt.subplots(1, image_batch.shape[0], figsize=(12, 4))
        for k in range(0, image_batch.shape[0]):
            image = image_batch[k]
            pred = predicted_batch[k]
            the_pred = np.argmax(pred)
            predicted = class_names[the_pred]
            val_pred = max(pred)
            the_class = np.argmax(classes_batch[k])
            value = class_names[the_class]

            axes[k].imshow(image)
            axes[k].set_title('Class:' + value + ' - ' + 'Pred ratio of:' + predicted + '[' + str(val_pred) + ']', fontsize=6)

            plt.show()

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict_on_data(self, test_data, class_names):  
        processed_images = efficientnet.keras.preprocess_input(test_data)
        predicted_batch = self.model.predict(processed_images)

        return predicted_batch

