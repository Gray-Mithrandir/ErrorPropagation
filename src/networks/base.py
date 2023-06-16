"""Basing network"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import tensorflow as tf

import visualization
from config import AugmentationSettings, PreProcessing
from load_dataset import (
    get_dataset,
    get_test_dataset,
    get_train_weights,
    prefetch_dataset,
)
from tracker import TrainPoint, TrainType


class NetworkInterface(ABC):
    """Neural network interface"""

    def __init__(self):
        self.logger = logging.getLogger("raido.network")
        self.pre_processing = PreProcessing()
        self._corruption = 0
        self._reduction = 0
        self._train_type = TrainType.NORMAL
        self._corrupted_labels = []

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Network name"""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Batch size"""

    @property
    @abstractmethod
    def epochs(self) -> int:
        """Number of train epochs"""

    @abstractmethod
    def create_model(self, augment: bool = True) -> tf.keras.Model:
        """Create a new network

        Parameters
        ----------
        augment: bool
            If set add augmentation layers

        Returns
        -------
        tf.keras.Model
            New model
        """

    @property
    def reduction(self) -> float:
        """Dataset reduction fraction"""
        return self._reduction

    @reduction.setter
    def reduction(self, value: float):
        if value > 1 or value < 0:
            raise ValueError("Reduction fraction must be in range [0-1]")
        self.logger.info("Updating dataset reduction to %s", value)
        self._reduction = value

    @property
    def corruption(self) -> float:
        """Dataset label corruption fraction"""
        return self._corruption

    @corruption.setter
    def corruption(self, value: float):
        if value > 1 or value < 0:
            raise ValueError("Corruption fraction must be in range [0-1]")
        self.logger.info("Updating dataset label corruption to %s", value)
        self._corruption = value

    @property
    def train_type(self) -> TrainType:
        """Model train option"""
        return self._train_type

    @train_type.setter
    def train_type(self, value: TrainType):
        self.logger.info("Updating train type to %s", value)
        self._train_type = value

    @property
    def corrupted_labels(self) -> Tuple[str, ...]:
        """Tuple of labels to swap during corruption process"""
        return tuple(self._corrupted_labels)

    @corrupted_labels.setter
    def corrupted_labels(self, value: Tuple[str, ...]):
        test_ds = get_test_dataset()
        for label in value:
            if label not in test_ds.class_names:
                raise ValueError(f"Label {label} not exist in dataset")
        self._corrupted_labels = value

    @property
    def train_point(self) -> TrainPoint:
        """Return train point"""
        return TrainPoint(
            reduction=self.reduction,
            corruption=self.corruption,
            train_type=self.train_type,
        )

    @property
    def history_path(self) -> Path:
        """Return history path depends on train parameters

        Returns
        -------
        Path
            History folder
        """
        _path = Path(
            "history",
            self.name(),
            f"{self.train_type.value}_c{self.corruption * 100:03.0f}_r{self.reduction * 100:03.0f}",
        )
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def report_path(self) -> Path:
        """Return report path depends on train parameters for train information

        Returns
        -------
        Path
            Report folder
        """
        _path = Path(
            "reports",
            self.name(),
            f"{self.train_type.value}_c{self.corruption * 100:03.0f}_r{self.reduction * 100:03.0f}",
        )
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def checkpoint_path(self) -> Path:
        """Return best model weights path

        Returns
        -------
        Path
            Checkpoint folder
        """
        _path = Path(self.history_path, "restore", "weights")
        Path(self.history_path, "restore").mkdir(exist_ok=True, parents=True)
        return _path

    def plot_model(self):
        """Plot model"""
        model = self.create_model(augment=False)
        model.build(input_shape=[None, *self.pre_processing.image_size, 1])
        export_path = Path("reports", self.name())
        export_path.mkdir(parents=True, exist_ok=True)
        visualization.plot_model(model, export_path)

    def train(self) -> tf.keras.callbacks.History:
        """Single train loop

        Returns
        -------
        tf.keras.callbacks.History
            Train history
        """
        model = self.create_model(augment=True)
        self.logger.info(
            "Starting training on dataset reduction of %05.1f%% with corruption %05.1f%%",
            self.reduction * 100.0,
            self.corruption * 100.0,
        )
        train_ds = get_dataset(
            reduction=self.reduction,
            train=True,
            balance=self.train_type == TrainType.BALANCED,
            swap_labels=("covid19", "pneumonia"),
            swap_probability=self.corruption,
            export_path=self.report_path,
        )
        prefetched_ds = prefetch_dataset(train_ds, batch_size=self.batch_size)
        validation_ds = get_dataset(
            reduction=self.reduction,
            train=False,
            balance=False,
            swap_labels=("covid19", "pneumonia"),
            swap_probability=self.corruption,
            export_path=self.report_path,
        )

        self.logger.info("Compile model")
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        self.logger.info("Starting model training!")
        if self.train_type is TrainType.BALANCED:
            class_weights = get_train_weights(train_ds)
            self.logger.info("Training weights %s", class_weights)
        else:
            class_weights = None

        history = model.fit(
            prefetched_ds,
            epochs=self.epochs,
            verbose=1,
            validation_data=prefetch_dataset(validation_ds, batch_size=self.batch_size),
            callbacks=self.callbacks(),
            class_weight=class_weights,
        )
        self.logger.info("Loading best weights")
        model.load_weights(self.checkpoint_path).expect_partial()
        self.logger.info("Plotting history")
        visualization.plot_history(history, self.report_path / "train_history.png")
        return history

    def evaluate(self) -> tf.keras.Model:
        """Load trained model and evaluate performance on test and validation dataset

        Returns
        -------
        tf.keras.Model
            Loaded model
        """
        self.logger.info("Loading model and weights")
        model = self.create_model(augment=True)
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        model.load_weights(self.checkpoint_path)
        validation_ds = get_dataset(
            reduction=self.reduction,
            train=False,
            balance=False,
            swap_labels=self.corrupted_labels,
            swap_probability=self.corruption,
            export_path=None,
        )
        test_ds = get_test_dataset()
        self.logger.info("Plotting performance plots")
        visualization.plot_performance(
            model=model,
            dataset=validation_ds,
            labels=test_ds.class_names,
            num_images=3,
            export_path=self.report_path / "performance_on_validation_ds.png",
        )
        visualization.plot_performance(
            model=model,
            dataset=test_ds,
            labels=test_ds.class_names,
            num_images=3,
            export_path=self.report_path / "performance_on_test_ds.png",
        )
        self.logger.info("Plotting confusion matrix")
        visualization.plot_confusion_matrix(
            model=model,
            dataset=validation_ds,
            labels=test_ds.class_names,
            export_path=self.report_path / "confusion_matrix_on_validation_ds.png",
        )
        visualization.plot_confusion_matrix(
            model=model,
            dataset=test_ds,
            labels=test_ds.class_names,
            export_path=self.report_path / "confusion_matrix_on_test_ds.png",
        )
        self.logger.info("Saving performance metrics")
        visualization.save_classification_report(
            model=model,
            dataset=validation_ds,
            labels=test_ds.class_names,
            export_path=self.report_path / "classification_report_on_validation_ds",
        )
        visualization.save_classification_report(
            model=model,
            dataset=test_ds,
            labels=test_ds.class_names,
            export_path=self.report_path / "classification_report_on_test_ds",
        )
        self.logger.info("Saving evaluation report")
        return model

    def callbacks(self) -> Tuple[tf.keras.callbacks.Callback, ...]:
        """Return list of train callbacks

        Returns
        -------
        Tuple[tf.keras.callbacks.Callback, ...]
            Callbacks
        """
        csv_epoch_path = self.report_path / "epoch.csv"
        tensor_board_path = self.history_path / "fit"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensor_board_path,
            histogram_freq=1,
            write_images=True,
            write_graph=True,
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=3,
            verbose=1,
            cooldown=3,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", verbose=1, patience=3, start_from_epoch=5
        )

        csv_logger = tf.keras.callbacks.CSVLogger(f"{csv_epoch_path}")
        return (
            model_checkpoint_callback,
            tensorboard_callback,
            csv_logger,
            reduce_lr_callback,
            early_stop_callback,
        )

    @staticmethod
    def _get_augment_layers() -> Tuple[tf.keras.layers.Layer, ...]:
        """Return augmentation layers"""
        settings = AugmentationSettings()
        image_sizes = PreProcessing().image_size
        return (
            tf.keras.layers.RandomTranslation(
                height_factor=settings.height_shift_range / 100.0,
                width_factor=settings.width_shift_range / 100.0,
                fill_mode="constant",
                fill_value=0,
                input_shape=[*image_sizes, 1],
            ),
            tf.keras.layers.RandomZoom(
                height_factor=settings.zoom_range / 100.0,
                fill_mode="constant",
                fill_value=0,
            ),
            tf.keras.layers.RandomRotation(
                factor=settings.rotation_angle / 100.0,
                fill_mode="constant",
                fill_value=0,
            ),
        )