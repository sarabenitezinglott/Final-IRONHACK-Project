import numpy as np
from keras.applications import densenet
from keras.models import Model, load_model  
from keras.layers import Activation, Dense  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from keras import regularizers 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import densenet, EfficientNetB0


"DenseNet model:"
class DenseNet_model:
    def __init__(self, train_generator=None, test_generator=None, img_width=150, img_height=150, epochs=2):
        self.model = self.densenet_model(train_generator, test_generator, img_width, img_height, epochs)

    def densenet_model(self, train_generator, test_generator, img_width, img_height, epochs):  
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

        return model
    
    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_generator, test_generator, epochs =2):
        nb_validation_samples = 4
        batch_size = 2

        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
        callbacks_list = [early_stop, reduce_lr]

        history = self.model.fit(train_generator,epochs=epochs,validation_data=test_generator,
                                          validation_steps=nb_validation_samples // batch_size,
                                          callbacks=callbacks_list)

        return history

    def get_weights(self):
        # In a DenseNet layer, the weights are stored in get_weights()[0], 
        # and the biases are stored in get_weights()[1].
        model_weights = self.model.get_weights()[0]
        np.save("data/weights.npy", model_weights)

    def evaluation(self, test_generator, class_names, batch_size):
        batch_size = 2
        verbose = 2
        evaluate = self.model.evaluate(test_generator, class_names, batch_size, verbose)
        return evaluate

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

        