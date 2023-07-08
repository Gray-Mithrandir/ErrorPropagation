"""Basing network"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import tensorflow as tf

import visualization
from config import AugmentationSettings, PreProcessing
from dataset import cook_dataset, save_dataset_statistics, get_dataset, get_test_dataset
from tracker import TrainInfo, TrainType


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
    def reduction(self) -> int:
        """Dataset reduction fraction"""
        return self._reduction

    @reduction.setter
    def reduction(self, value: int):
        self.logger.info("Updating dataset reduction to %s", value)
        self._reduction = value

    @property
    def corruption(self) -> int:
        """Dataset label corruption fraction"""
        return self._corruption

    @corruption.setter
    def corruption(self, value: int):
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
    def train_point(self) -> TrainInfo:
        """Return train point"""
        return TrainInfo(
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
            f"{self.train_type.value}_c{self.corruption}_r{self.reduction}",
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
            f"{self.train_type.value}_c{self.corruption}_r{self.reduction}",
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
        class_weights = cook_dataset(
            reduction=self.reduction / 100.0,
            corruption={
                ("covid19", "pneumonia"): self.corruption / 100.0,
                ("pneumonia", "covid19"): self.corruption / 100.0,
            },
            balance=self.train_type is TrainType.BALANCED
        )
        save_dataset_statistics(self.report_path)
        model = self.create_model(augment=True)
        self.logger.info(
            "Starting training on dataset reduction of %d%% with corruption %d%%",
            self.reduction,
            self.corruption,
        )
        train_ds = get_dataset(training=True, batch=self.batch_size)
        validation_ds = get_dataset(training=False, batch=self.batch_size)

        self.logger.info("Compile model")
        model.compile(
            loss=tf.keras.metrics.categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(),
            metrics=["accuracy"],
        )

        self.logger.info("Starting model training!")
        history = model.fit(
            train_ds,
            epochs=self.epochs,
            verbose=1,
            validation_data=validation_ds,
            callbacks=self.callbacks(),
            class_weight=dict(enumerate(class_weights)) if self.train_type is TrainType.WEIGHTED else None,
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
        self.logger.info("Plotting performance plots")
        visualization.plot_performance(
            model=model,
            export_path=self.report_path
        )
        self.logger.info("Plotting confusion matrix")
        visualization.plot_confusion_matrix(
            model=model,
            export_path=self.report_path,
        )
        self.logger.info("Saving performance metrics")
        visualization.save_classification_report(
            model=model,
            export_path=self.report_path,
        )
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

        csv_logger = tf.keras.callbacks.CSVLogger(f"{csv_epoch_path}")
        return (
            model_checkpoint_callback,
            tensorboard_callback,
            csv_logger,
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
