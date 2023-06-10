"""Model train, validation and test procedures"""
import logging
from typing import Callable, Tuple
import tensorflow as tf
from pathlib import Path

from logger import init_logger
from recorder import TrainStatistics
from visualization import plot_model
from config import TrainInfo, TrainType, PreProcessing
from load_dataset import get_dataset, prefetch_dataset
from preprocessing import get_pre_processing


def runner(
    create_model: Callable[[], tf.keras.models.Model],
    callbacks: Tuple[tf.keras.callbacks.Callback, ...],
    batch_size: int,
    epochs: int,
):
    model = create_model()
    model.build(
        input_shape=[None, *PreProcessing().image_size, 1]
    )
    base_info = TrainInfo(
        name=model.name, corruption=0, reduction=0, train_type=TrainType.NORMAL
    )
    init_logger(Path("logs", base_info.safe_name))
    logger = logging.getLogger("raido.dataset")
    statistics = TrainStatistics(base_info.safe_name)
    plot_model(model, base_info)
    logger.info("Starting model %s", base_info.name)
    for train_type in TrainType:
        base_info = TrainInfo(
            name=model.name, corruption=0, reduction=0, train_type=train_type
        )
        for reduction in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for corruption in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
                model = create_model()  # TODO add preprocessing layers
                train_info = TrainInfo(
                    name=base_info.name,
                    corruption=corruption,
                    reduction=100-reduction,
                    train_type=train_type,
                )

            if statistics.is_already_trained(train_info):
                logger.warning(
                    "Point with reduction %s and corruption %s already exist. Skipping",
                    reduction,
                    corruption,
                )
                continue
            logger.info(
                "Starting training on dataset reduction of %s%% with corruption %s%%",
                reduction,
                corruption,
            )
            train_ds = prefetch_dataset(
                get_dataset(
                    reduction=reduction,
                    train=True,
                    balance=train_type == TrainType.BALANCED,
                ),
                batch_size=batch_size,
            )
            validation_ds = get_dataset(reduction=reduction / 100.0, train=False, balance=False)

            logger.info("Compile model")
            model.compile(
                loss=tf.keras.metrics.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"],
            )

            csv_epoch_path = train_info.history_path / "epoch.csv"
            tensor_board_path = train_info.history_path / "fit"

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensor_board_path,
                histogram_freq=1,
                write_images=True,
                write_graph=True,
            )
            csv_logger = tf.keras.callbacks.CSVLogger(f"{csv_epoch_path}")
            register_callback = [tensorboard_callback, csv_logger, *callbacks]

            logger.info("Starting model training!")

            history = model.fit(
                train_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=prefetch_dataset(validation_ds, batch_size=batch_size),
                callbacks=register_callback,
            )
            statistics.add_train_metrics(train_info, history)
            statistics.evaluate_point(train_info, model, validation_ds)
        statistics.plot_general_summary(base_info)
