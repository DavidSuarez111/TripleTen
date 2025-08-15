
import pandas as pd

import scipy
import tensorflow as tf
import sys
sys.modules['scipy'] = scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):
    
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    Usa ImageDataGenerator con aumentos.
    """
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))

    datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=os.path.join(path, 'final_files'),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=42
    )

    return train_gen


def load_test(path):
    
    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta.
    """
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))

    datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=os.path.join(path, 'final_files'),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=42
    )

    return val_gen


def create_model(input_shape):
    """
    Define el modelo con base en ResNet50.
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Congelar pesos base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Salida continua (edad)
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Entrena el modelo dados los parámetros.
    """
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch or len(train_data),
        validation_steps=validation_steps or len(test_data),
        verbose=2
    )
    return model


