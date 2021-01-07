import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
    ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE = 28
N_CLASSES = 47


def preprocess_images(dataframe, inverted=True, normalize=True):
    # PREPROCESSING
    labels = np.array(dataframe.iloc[:, 0].values)
    features = np.array(dataframe.iloc[:, 1:].values)

    if inverted:
        # Invert image to fit our situation
        features = 255 - features

    if normalize:
        features = features / 255.0

    return features, labels


def build_model(train,
                train_labels,
                test,
                test_labels,
                epochs):
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

    history = model.fit(train,
                        train_labels,
                        epochs=epochs,
                        validation_data=(test, test_labels),
                        callbacks=[MCP, ES, RLP])
    return model, history


def save_record(model_history):
    q = len(model_history.history['accuracy'])

    plt.figsize = (10, 10)
    sns.lineplot(x=range(1, 1 + q), y=model_history.history['accuracy'],
                 label='Accuracy')
    sns.lineplot(x=range(1, 1 + q), y=model_history.history['val_accuracy'],
                 label='Val_Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('./graphs/loss_history.png')


if __name__ == '__main__':
    # IMPORT (EMNIST : 28x28 images)
    training_letter = pd.read_csv('./data/EMNIST/emnist-balanced-train.csv')
    TRAIN_SIZE = training_letter.shape[0]

    # PREPROCESSING
    train_images, labels = preprocess_images(training_letter,
                                             inverted=True,
                                             normalize=True)

    # Get right shape of data
    train_images = train_images.reshape((TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                                         1))

    # Transform labels
    labels = tf.keras.utils.to_categorical(labels, N_CLASSES)

    # Build model
    train_x, test_x, train_y, test_y = train_test_split(train_images,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

    _, history = build_model(train=train_x,
                             train_labels=train_y,
                             test=test_x,
                             test_labels=test_y,
                             epochs=10)

    # Visualization of training history
    save_record(history)
