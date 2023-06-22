"""VGG-16 implementation"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from networks.base import NetworkInterface


class Network(NetworkInterface):
    """VGG-16 implementation"""

    @staticmethod
    def name() -> str:
        """Network name"""
        return "VGG16"

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return 8

    @property
    def epochs(self) -> int:
        """Train epochs"""
        return 100

    def create_model(self, augment: bool = True) -> tf.keras.Model:
        """Return a new model"""
        model = Sequential(name="VGG-16")
        model.add(tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1))
        if augment:
            for layer in self._get_augment_layers():
                model.add(layer)

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="sigmoid")
        )
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="sigmoid"))
        model.add(Dense(units=4096, activation="sigmoid"))
        model.add(Dense(units=3, activation="softmax"))

        return model

    def callbacks(self) -> Tuple[tf.keras.callbacks.Callback, ...]:
        """Adds ReduceLROnPlateau  and EarlyStopping to callbacks"""
        origin_callbacks = super().callbacks()
        return origin_callbacks
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=3,
            verbose=1,
            cooldown=5,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", verbose=1, patience=3, start_from_epoch=10
        )
        return origin_callbacks
