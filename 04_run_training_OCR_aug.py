import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
    ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random


def add_noise(img):
    VARIABILITY = 50
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


if __name__ == '__main__':
    # IMPORT (EMNIST : 28x28 images)
    training_letter = pd.read_csv('./data/EMNIST/emnist-letters-train.csv')

    # PREPROCESSING
    y1 = np.array(training_letter.iloc[:, 0].values)
    x1 = np.array(training_letter.iloc[:, 1:].values)
    train_images = x1 / 255.0
    TRAIN_SIZE = train_images.shape[0]
    IMAGE_SIZE = 28

    train_images = train_images.reshape(TRAIN_SIZE, IMAGE_SIZE,
                                        IMAGE_SIZE, 1)

    # TRANSFORM LABELS
    N_CLASSES = 37

    y1 = tf.keras.utils.to_categorical(y1, N_CLASSES)

    # create data generator
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.0],
                                 preprocessing_function=add_noise)
    datagen.fit(train_images)

    # Min size of image for resnet: 32x32, hence resizing layer
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(
            32, 32, interpolation="bilinear", name=None),
        tf.keras.applications.ResNet50(
            weights=None, input_shape=(32, 32, 1), classes=N_CLASSES
        )]
    )

    it = datagen.flow(train_images, y1)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(it,
                        epochs=10)

    model.save('./Model_RESNET_AUG.h5')
