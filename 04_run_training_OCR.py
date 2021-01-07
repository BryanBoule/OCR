import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
    ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    # IMPORT (EMNIST : 28x28 images)
    training_letter = pd.read_csv('./data/EMNIST/emnist-balanced-train.csv')

    # PREPROCESSING
    y1 = np.array(training_letter.iloc[:, 0].values)
    x1 = np.array(training_letter.iloc[:, 1:].values)

    # Invert image to fit our situation
    x1_inverted = 255*np.ones(x1.shape[0]) - x1
    train_images = x1_inverted / 255.0

    TRAIN_SIZE = train_images.shape[0]
    IMAGE_SIZE = 28

    train_images = train_images.reshape(TRAIN_SIZE, IMAGE_SIZE,
                                        IMAGE_SIZE, 1)

    # TRANSFORM LABELS
    N_CLASSES = 47

    y1 = tf.keras.utils.to_categorical(y1, N_CLASSES)

    # CNN
    train_x, test_x, train_y, test_y = train_test_split(train_images, y1,
                                                        test_size=0.2,
                                                        random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(
            32, 32, interpolation="bilinear", name=None),
        tf.keras.layers.Conv2D(32, 3, input_shape=(32, 32, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    MCP = ModelCheckpoint('Model_VCNN.h5',
                          verbose=1,
                          save_best_only=True,
                          monitor='val_accuracy',
                          mode='max')

    ES = EarlyStopping(monitor='val_accuracy',
                       min_delta=0,
                       verbose=0,
                       restore_best_weights=True,
                       patience=3,
                       mode='max')

    RLP = ReduceLROnPlateau(monitor='val_loss',
                            patience=3,
                            factor=0.2,
                            min_lr=0.0001)

    history = model.fit(train_x, train_y,
                        epochs=10,
                        validation_data=(test_x, test_y),
                        callbacks=[MCP, ES, RLP])

    # VISUALIZATION TRAINING HISTORY
    q = len(history.history['accuracy'])

    plt.figsize = (10, 10)
    sns.lineplot(x=range(1, 1 + q), y=history.history['accuracy'],
                 label='Accuracy')
    sns.lineplot(x=range(1, 1 + q), y=history.history['val_accuracy'],
                 label='Val_Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('./OCR_model/graphs/loss_history.png')
