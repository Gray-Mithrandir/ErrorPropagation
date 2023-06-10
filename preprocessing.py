"""Create augmentation and normalization layers"""
import tensorflow as tf

from config import AugmentationSettings, PreProcessing


def get_pre_processing() -> tf.keras.models.Sequential:
    """Return preprocessing model to attach to main model.
    The model contain augmentation and normalization layers and

    Returns
    -------
    tf.keras.models.Sequential
        Pre-processing model
    """
    settings = AugmentationSettings()
    image_sizes = PreProcessing().image_size
    model = tf.keras.models.Sequential(name="Pre-processing")
    model.add(
        tf.keras.layers.RandomTranslation(
            height_factor=settings.height_shift_range / 100.0,
            width_factor=settings.width_shift_range / 100.0,
            fill_mode="constant",
            fill_value=0,
            input_shape=[*image_sizes, 1],
        )
    )
    model.add(
        tf.keras.layers.RandomZoom(
            height_factor=settings.zoom_range / 100.0,
            fill_mode="constant",
            fill_value=0,
        )
    )
    model.add(
        tf.keras.layers.RandomRotation(
            factor=settings.rotation_angle / 100.0,
            fill_mode="constant",
            fill_value=0,
        )
    )
    model.add(tf.keras.layers.Rescaling(1.0 / 255))

    return model
