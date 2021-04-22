import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
import random
import cv2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

#Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow==2.4.1`

# Split out features and labels
#X_train, y_train = train_data['features'], train_data['labels']
#X_val, y_val = val_data['features'], val_data['labels']
#X_test, y_test = test_data['features'], test_data['labels']

#preprocess
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


######################


def nn_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(43, activation='softmax'))
  
  model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = nn_model()
#print(model.summary())

def run_model():
  history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=10,epochs=500,validation_data=(X_val, y_val), shuffle = 1)
  return history


def ResNet50_model():
  model_ResNet50 = keras.applications.resnet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  return model_ResNet50

#model_2 = ResNet50_model()
def run_model_ResNet50():
  history_2 = model_2.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=10,epochs=500,validation_data=(X_val, y_val), shuffle = 1)
  return history_2


def plot_Loss_epoch():
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss')
  plt.xlabel('epoch')

def plot_accuracy_epoch():
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.legend(['training','test'])
  plt.title('Accuracy')
  plt.xlabel('epoch')