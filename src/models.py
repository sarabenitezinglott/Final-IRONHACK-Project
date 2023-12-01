import numpy as np
from keras.applications import densenet
from keras.models import Model, load_model  
from keras.layers import Activation, Dense  
from keras import regularizers 
import matplotlib.pyplot as plt
import efficientnet.keras as efn 
import efficientnet.tfkeras
from tensorflow.keras.applications import EfficientNetB6

"DenseNet model:"

def densenet_model():  
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

    return model

def predict_densenet(model, test_generator, class_names):  
    image_batch, classes_batch = next(test_generator)
    predicted_batch = model.predict(image_batch)
    for k in range(0,image_batch.shape[0]):
      image = image_batch[k]
      pred = predicted_batch[k]
      the_pred = np.argmax(pred)
      predicted = class_names[the_pred]
      val_pred = max(pred)
      the_class = np.argmax(classes_batch[k])
      value = class_names[np.argmax(classes_batch[k])]
      plt.figure(k)
      isTrue = (the_pred == the_class)
      plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
      plt.imshow(image)


"EfficientNet model:"
