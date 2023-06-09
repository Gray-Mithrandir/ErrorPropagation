"""Load dataset for training, validation, and testing"""
import logging
from pathlib import Path

import tensorflow as tf

from config import PreProcessing


def get_dataset(reduction: float, train: bool, balance: bool) -> tf.data.Dataset:
    """Load dataset for training or validation

    Parameters
    ----------
    reduction: float
        Dataset reduction fraction (0-1)
    train: bool
        If set return training dataset, otherwise validation
    balance: bool
        If set the dataset will be balanced using over-sample.

    """
    logger = logging.getLogger("raido.dataset")
    _settings = PreProcessing()
    origin_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f'{Path("data", "cooked").absolute()}',
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=None,
        image_size=_settings.image_size,
        shuffle=True,
        seed=2820285082,
        validation_split=0.2,
        subset="training" if train else "validation",
    )
    # Reduce
    class_item_count = {}
    class_dataset = {}
    for class_index, class_name in enumerate(origin_ds.class_names):
        _filter = _create_class_filter(class_index)
        class_dataset[class_name] = origin_ds.filter(_filter)
        _counter = _create_counter()
        class_item_count[class_name] = (
            class_dataset[class_name].reduce(0, _counter).numpy()
        )
    logger.info("Total images before reduction %s", class_item_count)
    for class_name in origin_ds.class_names:
        original_size = class_item_count[class_name]
        class_item_count[class_name] = int(original_size * reduction)
        class_dataset[class_name] = (
            class_dataset[class_name]
            .shuffle(original_size)
            .take(class_item_count[class_name])
        )
    logger.info("Total images after reduction %s", class_item_count)
    if balance:
        largest_set = max(list(count for count in class_item_count.values()))
        logger.info("Balancing dataset. Each class size %s", largest_set)
        for class_name in origin_ds.class_names:
            class_dataset[class_name] = (
                class_dataset[class_name].repeat().take(largest_set)
            )
    logger.info("Combing datasets")
    return tf.data.Dataset.sample_from_datasets(
        list(dataset for dataset in class_dataset.values())
    )

def get_test_dataset() -> tf.data.Dataset:
    """Load test dataset

    Returns
    -------
    tf.data.Dataset
        Test dataset
    """
    _settings = PreProcessing()
    return tf.keras.utils.image_dataset_from_directory(
        directory=f'{Path("data", "test").absolute()}',
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=None,
        image_size=_settings.image_size,
        shuffle=True,
        seed=2820285082,
    )


def _create_class_filter(metric):
    """Creating callable for class filter"""

    def _filter(data, label):  # pylint: disable = unused-argument
        return tf.math.argmax(label) == metric

    return _filter


def _create_counter():
    """Creating callable for class counter"""

    def _filter(data, label):  # pylint: disable = unused-argument
        return data + 1

    return _filter


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(get_dataset(0.5, True, True).reduce(0, _create_counter()).numpy())
    print(get_dataset(0.5, False, False).reduce(0, _create_counter()).numpy())
    print(get_test_dataset().reduce(0, _create_counter()).numpy())
