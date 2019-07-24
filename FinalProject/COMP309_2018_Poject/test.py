
# coding: utf-8

# In[1]:


#input tensorflow and keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Convolution2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers

import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import random

#Online code
# from google.colab import drive
# drive.mount('/content/drive')


# In[5]:


#Test model
data_path = 'Test_data'
model_path = 'models/VGG16_adam_prepor_FULL_loss 0.5362'

#Input keras applications
from tensorflow.keras.applications import VGG16#, InceptionResNetV2
vgg16 = VGG16(include_top=False,input_shape=(300, 300, 3),classes=3)

#Build the model (VCG-16)
model = Sequential()
for layer in vgg16.layers:
  model.add(layer)
model.add(Flatten())
model.add(Dense(256, activation='relu',kernel_regularizer = regularizers.l2(0.001 )))
#model.add(Dropout(0.25))
model.add(Dense(256, activation='relu',kernel_regularizer = regularizers.l2(0.001 )))
#model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax',activity_regularizer=regularizers.l1(0.001)))
model.summary()


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=True)

#model compile
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[categorical_accuracy])
model.load_weights(model_path)

test_datagen = ImageDataGenerator(
    
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale = 1./255, 
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2

)

test_data = test_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300), 
    batch_size=30
)
print("Testing...")
loss_and_metrics = model.evaluate_generator(test_data,steps=len(test_data))
print('Test loss:{}\nTest accuracy:{}'.format(loss_and_metrics[0], loss_and_metrics[1]))

