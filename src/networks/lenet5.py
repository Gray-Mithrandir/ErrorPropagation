"""LeNet-5 implementation"""
from typing import Tuple

import tensorflow as tf

from networks.base import NetworkInterface


class Network(NetworkInterface):
    """LeNet-5 implementation"""
    @staticmethod
    def name() -> str:
        return "LeNet5"

    @property
    def batch_size(self) -> int:
        return 32

    @property
    def epochs(self) -> int:
        return 80

    def create_model(self, augment: bool = True) -> tf.keras.Model:
        """Create LeNet-5 model

            Parameters
            ----------
            augment: bool = True
                If set adds augmentation layers

            Returns
            ------
            Sequential
                New model
            """
        model = tf.keras.models.Sequential(name="LeNet-5")
        model.add(tf.keras.layers.Rescaling(1.0 / 255))
        if augment:
            for layer in self._get_augment_layers():
                model.add(layer)
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=120, activation="relu"))
        model.add(tf.keras.layers.Dense(units=84, activation="relu"))
        model.add(tf.keras.layers.Dense(units=3, activation="softmax"))
        return model

    def callbacks(self) -> Tuple[tf.keras.callbacks.Callback, ...]:
        """Adds ReduceLROnPlateau  and EarlyStopping to callbacks"""
        origin_callbacks = super().callbacks()
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=3,
            verbose=1,
            cooldown=3,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", verbose=1, patience=5, start_from_epoch=10
        )
        return origin_callbacks + (reduce_lr_callback, early_stop_callback)
