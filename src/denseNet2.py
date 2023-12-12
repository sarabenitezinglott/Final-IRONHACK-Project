import os
import keras
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

class YourDenseNetModel:
    def __init__(self):
        self.model = self.build_densenet_model()
        self.compile_model()
        self.train_generator = None
        self.validation_generator = None

    def build_densenet_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model
    
    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def set_generators(self, train_generator, validation_generator):
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def train(self, x_train, epochs=3, batch_size=32):
        nb_validation_samples = 159
        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
        callbacks_list = [early_stop, reduce_lr]

        # Calculate steps per epoch and validation steps
        steps_per_epoch = len(x_train)//batch_size
        validation_steps = int(np.ceil(nb_validation_samples / batch_size))

        history = self.model.fit(self.train_generator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    validation_data=self.validation_generator,
                                    validation_steps=validation_steps,
                                    callbacks=callbacks_list)

        return history


    def get_weights(self,weights_path):
        self.model.save_weights(weights_path)

    def evaluation(self, batch_size):
        verbose = 1
        evaluate = self.model.evaluate(
            self.validation_generator,
            batch_size=batch_size,
            verbose=verbose)
        return evaluate

    def predict_densenet(self, class_names):
        image_batch, classes_batch = next(self.validation_generator)
        predicted_batch = self.model.predict(image_batch)
        for k in range(0, image_batch.shape[0]):
            image = image_batch[k]
            pred = predicted_batch[k]
            the_pred = np.argmax(pred)
            predicted = class_names[the_pred]
            val_pred = max(pred)
            the_class = np.argmax(classes_batch[k])
            value = class_names[the_class]

            plt.imshow(image)
            plt.title(f"{value} - Pred ratio: {predicted} [{val_pred}]", fontsize=8)
            
            # Save images separately
            save_file = os.path.join("./images/denseNet/", f'prediction_{k}.png')
            plt.savefig(save_file)
            plt.close()  


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict_on_data(self, test_data, class_names):  
        processed_images = densenet_preprocess_input(test_data)
        predicted_batch = self.model.predict(processed_images)

        return predicted_batch