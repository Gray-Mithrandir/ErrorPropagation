"""LeNet-5 implementation"""
from typing import Tuple

import tensorflow as tf


def create_model() -> tf.keras.models.Sequential:
    """Create LeNet-5 model

    Returns
    ------
    Sequential
        New model
    """
    model = tf.keras.models.Sequential(name="LeNet-5")
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation="relu"))
    model.add(tf.keras.layers.Dense(units=84, activation="relu"))
    model.add(tf.keras.layers.Dense(units=3, activation="softmax"))
    return model


def callbacks() -> Tuple[tf.keras.callbacks.Callback, ...]:
    """Return additional callback for model as Reduce learning rate and early stop

    Returns
    -------
    Tuple[tf.keras.callbacks.Callback, ...]
        Callbacks
    """
    return (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.1, patience=4, verbose=1, cooldown=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", verbose=1, patience=6, start_from_epoch=5
        ),
    )


if __name__ == '__main__':
    from runner import runner
    runner(create_model, callbacks(), 16, 40)
