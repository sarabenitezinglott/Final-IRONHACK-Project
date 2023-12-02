import warnings
warnings.filterwarnings("ignore")

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
        outputs = layers.Dense(2, activation="softmax", name="pred")(x)
        model = keras.Model(inputs, outputs, name="EfficientNet")
        
        return model

    def compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)  
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", 
                          metrics=["accuracy"])
        
    def train(self, train_generator, validation_generator, epochs):
        epochs = 2
        verbose = 2
        efficientnet_history = self.model.fit(train_generator,validation_data=validation_generator,
                                              epochs = epochs, verbose = verbose)

        return efficientnet_history
    
    def get_weights(self):
        # In a DenseNet layer, the weights are stored in get_weights()[0], 
        # and the biases are stored in get_weights()[1].
        model_weights = self.model.get_weights()[0]
        np.save("data/Efficient_weights.npy", model_weights)
    
    def evaluation_B0(self, test_generator_x,test_generator_y):
        evaluate = self.model.evaluate(test_generator_x,test_generator_y)
        return evaluate


    # def unfreeze_model(self, model): #maybe ponerlo arriba jjeje
    # # We unfreeze the top 20 layers while leaving BatchNorm layers 
    # # frozen
    #     for layer in model.layers[-20:]:
    #         if not isinstance(layer, layers.BatchNormalization):
    #             layer.trainable = True

    #     optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    #     model.compile(
    #         optimizer=optimizer, 
    #         loss="categorical_crossentropy", 
    #         metrics=["accuracy"])
