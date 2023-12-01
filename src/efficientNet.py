import numpy as np
import keras
import efficientnet.keras as efn 
import efficientnet.tfkeras
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Model, load_model  
from keras import layers
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
        inputs = layers.Input(shape=(150, 150, 3))
        model = EfficientNetB0(include_top=False, input_tensor=inputs, 
                               weights="imagenet")
    
        model.trainable = False

        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(2, activation="softmax", name="pred")(x)

        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", 
            metrics=["accuracy"])
        
        return model

    
    def train(self, train_generator, test_generator, epochs=2):
        efficientnet_history = self.model.fit(train_generator, epochs=epochs,
                                               validation_data=test_generator)

        return efficientnet_history
    
    def unfreeze_model(self, model): #maybe ponerlo arriba jjeje
    # We unfreeze the top 20 layers while leaving BatchNorm layers 
    # frozen
        for layer in model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            optimizer=optimizer, 
            loss="categorical_crossentropy", 
            metrics=["accuracy"])
