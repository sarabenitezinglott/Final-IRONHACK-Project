import numpy as np
from keras.applications import densenet
from keras.models import Model, load_model  
from keras.layers import Activation, Dense  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from keras import regularizers 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  


"DenseNet model:"
class DenseNet_model:
    def __init__(self):
        self.model = self.densenet_model()

    def densenet_model(self):  
        base_model = densenet.DenseNet121(input_shape=(150, 150, 3),
                                        weights="imagenet",
                                        include_top=False,
                                        pooling='avg')

        x = base_model.output
        x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
        x = Activation('relu')(x)
        x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
        x = Activation('relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = Adam()
        loss = categorical_crossentropy
        metrics = ["accuracy", "mse"]
        model.compile(optimizer = optimizer, loss = loss, 
                      metrics = metrics)

        return model
    
    def train(self, train_generator, test_generator, epochs =2):
        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)  
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)  
        callbacks_list = [early_stop, reduce_lr]  
        nb_validation_samples = 4
        batch_size = 2

        densenet_history = self.model.fit(train_generator,
                                epochs=epochs,
                                validation_data=test_generator,
                                validation_steps=nb_validation_samples // batch_size,
                                callbacks=callbacks_list) 

    def predict_densenet(self, test_generator, class_names):  
        image_batch, classes_batch = next(test_generator)
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

        