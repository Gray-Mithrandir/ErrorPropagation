"""Load dataset for training, validation, and testing"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict

import numpy as np
import tensorflow as tf

from config import PreProcessing


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_dataset(
    reduction: float,
    train: bool,
    balance: bool,
    swap_labels: Tuple[str, ...],
    swap_probability: float,
    export_path: Optional[Path] = None,
) -> tf.data.Dataset:
    """Load dataset for training or validation

    Parameters
    ----------
    reduction: float
        Dataset reduction fraction [0-1]
    train: bool
        If set return training dataset, otherwise validation
    balance: bool
        If set the dataset will be balanced using over-sample.
    swap_labels: Tuple[str, ...]
        Two label to swap between (Label corruption)
    swap_probability: float
        Swap probability [0-1]
    export_path: Optional[Path]
        Folder to save dataset info

    Returns
    -------
    tf.data.Dataset
        Loaded dataset
    """
    logger = logging.getLogger("raido.dataset")
    _settings = PreProcessing()
    ds_status = {}
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

    class_item_count = {}
    class_dataset = {}
    for class_index, class_name in enumerate(origin_ds.class_names):
        _filter = create_class_filter(class_index)
        class_dataset[class_name] = origin_ds.filter(_filter)
        _counter = _create_counter()
        class_item_count[class_name] = (
            class_dataset[class_name].reduce(0, _counter).numpy()
        )
        if "source" not in ds_status:
            ds_status["source"] = {}
        ds_status["source"][class_name] = class_item_count[class_name]
    logger.info("Total images in dataset %s", class_item_count)
    # Reduce
    for class_name in origin_ds.class_names:
        original_size = class_item_count[class_name]
        class_item_count[class_name] = int(original_size * (1 - reduction))
        class_dataset[class_name] = (
            class_dataset[class_name]
            .shuffle(original_size)
            .take(class_item_count[class_name])
        )
        if "reduced" not in ds_status:
            ds_status["reduced"] = {}
        ds_status["reduced"][class_name] = (
            class_dataset[class_name].reduce(0, _create_counter()).numpy()
        )
    logger.info("Total images after reduction %s", class_item_count)
    # Corrupt
    _rev_labels = list(swap_labels)
    _rev_labels.reverse()
    for source, target in zip(swap_labels, _rev_labels):
        _, new_label = next(iter(class_dataset[target].take(1)))
        class_dataset[source] = class_dataset[source].map(
            map_func=swap_with_probability(
                probability=swap_probability, new_label=new_label
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        corrupted_labels = (
            class_dataset[source]
            .filter(create_class_filter(list(origin_ds.class_names).index(target)))
            .reduce(0, _create_counter())
            .numpy()
        )
        if "corrupted" not in ds_status:
            ds_status["corrupted"] = {}
        if source not in ds_status["corrupted"]:
            ds_status["corrupted"][source] = {}
        ds_status["corrupted"][source]["changed"] = corrupted_labels
        ds_status["corrupted"][source]["unchanged"] = (
            class_dataset[source]
            .filter(create_class_filter(list(origin_ds.class_names).index(source)))
            .reduce(0, _create_counter())
            .numpy()
        )
        logger.info("Corrupted labels in class %s %s", source, corrupted_labels)

    if balance:
        largest_set = max(list(count for count in class_item_count.values()))
        logger.info("Balancing dataset. Each class size %s", largest_set)
        for class_name in origin_ds.class_names:
            class_dataset[class_name] = (
                class_dataset[class_name].repeat().take(largest_set)
            )
            if "balanced" not in ds_status:
                ds_status["balanced"] = {}
            ds_status["balanced"][class_name] = (
                class_dataset[class_name].reduce(0, _create_counter()).numpy()
            )
    logger.info("Combing datasets")
    combined = tf.data.Dataset.sample_from_datasets(
        list(dataset for dataset in class_dataset.values())
    )
    ds_status["combined"] = combined.reduce(0, _create_counter()).numpy()
    if export_path is not None:
        if train:
            ds_status["weights"] = {}
            ds_status["weights"]["origin"] = get_train_weights(combined)
            for class_name, weight in zip(origin_ds.class_names, ds_status["weights"]["origin"].values()):
                ds_status["weights"][class_name] = weight
            _file = export_path / "train_dataset.json"
        else:
            _file = export_path / "validation_dataset.json"
        with open(_file, mode="w", encoding="utf-8") as json_fh:
            json.dump(ds_status, json_fh, indent=4, cls=_NpEncoder)
    return combined


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


def get_train_weights(dataset: tf.data.Dataset) -> Dict[int, float]:
    """Calculate train weights of given dataset

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to calculate weights


    Returns
    -------
    Dict[int, float]
        Weights dictionary
    """
    total = dataset.reduce(0, _create_counter()).numpy()
    weights = {}
    class_index = 0
    while True:
        _filter = create_class_filter(class_index)
        _counter = _create_counter()
        count = dataset.filter(_filter).reduce(0, _counter).numpy()
        if count == 0:
            break
        weights[class_index] = count
        class_index += 1
    for class_index in list(weights.keys()):
        weights[class_index] = (1.0 / weights[class_index]) * (total / len(weights.keys()))
    return weights




def create_class_filter(metric):
    """Creating callable for class filter"""

    def _filter(data, label):  # pylint: disable = unused-argument
        return tf.math.argmax(label) == metric

    return _filter


def prefetch_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """Optimize dataset set for fast reading

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to load
    batch_size: int
        Dataset batch size

    Returns
    -------
    dataset: tf.data.Dataset
        Prefetched dataset
    """
    return dataset.batch(
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(buffer_size=tf.data.AUTOTUNE)


def _create_counter():
    """Creating callable for class counter"""

    def _filter(data, label):  # pylint: disable = unused-argument
        return data + 1

    return _filter


def swap_with_probability(
    probability: float, new_label: tf.Tensor
) -> Callable[[Any, Any], Any]:
    """Corrupt label with probability"""

    @tf.function
    def _swap(data, label):
        return tf.cond(
            tf.random.uniform(()) > probability,
            lambda: (data, label),
            lambda: (data, new_label),
        )

    return _swap
