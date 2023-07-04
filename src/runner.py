"""Dynamicly load model and run train loops"""
import logging
import os
from importlib import import_module
from pathlib import Path

import numpy as np

from logger import init_logger
from networks.base import NetworkInterface
from recorder import TrainStatistics
from tracker import RunTracker, TrainType
from dataset import prepare_dataset


def main():
    """Main train loop"""
    model_class = getattr(
        import_module(f"networks.{os.getenv('RAIDO_MODEL')}"), "Network"
    )  # type: NetworkInterface
    init_logger(Path("logs") / model_class.name())
    logger = logging.getLogger("raido")
    network = model_class()  # type: NetworkInterface
    logger.info("Plotting model")
    network.plot_model()
    train_tracker = RunTracker(Path("logs", network.name(), "tracker.json"))
    excel_stats = TrainStatistics(Path("reports", network.name()))
    prepare_dataset(Path("reports", network.name()))

    for train_type in TrainType:
        logger.info("Starting train type: %s", train_type.value)
        for corruption in np.arange(0.0, 60.0, 5):
            logger.info("Label corruption - %.0f%%", corruption)
            for reduction in np.arange(0.0, 100.0, 10):
                logger.info("Dataset reduction - %.0f%%", reduction)
                network = model_class()
                network.train_type = train_type
                network.corruption = corruption
                network.reduction = reduction
                if train_tracker.is_point_trained(network.train_point):
                    logger.warning("Model already trained on this point. Skipping")
                else:
                    logger.info("Starting model training")
                    history = network.train()
                    excel_stats.save_train_history(network.train_point, history)
                    train_tracker.set_train_complete(network.train_point)

                if train_tracker.is_point_evaluated(network.train_point):
                    logger.warning("Model already evaluated on this point skip")
                else:
                    model = network.evaluate()
                    excel_stats.save_evaluation(model, network.train_point)
                    train_tracker.set_evaluation_complete(network.train_point)
    excel_stats.plot_class_summary()
    excel_stats.plot_train_summary()


if __name__ == "__main__":
    main()
