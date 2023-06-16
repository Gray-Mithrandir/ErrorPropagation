"""Dynamicly load model and run train loops"""
import logging
from importlib import import_module
from recorder import TrainStatistics
from tracker import TrainType, RunTracker
from pathlib import Path

from logger import init_logger
import os
from networks.base import NetworkInterface
import numpy as np


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

    for train_type in TrainType:
        logger.info("Starting train type: %s", train_type.value)
        for corruption in np.arange(0.0, 0.6, 0.05):
            logger.info("Label corruption - %.0f%%", corruption * 100)
            for reduction in np.arange(0.0, 1.0, 0.1):
                logger.info("Dataset reduction - %.0f%%", reduction * 100)
                network = model_class()
                network.train_type = train_type
                network.corruption = corruption
                network.reduction = reduction
                if train_tracker.is_point_trained(**network.train_point.dict()):
                    logger.warning("Model already trained on this point. Skipping")
                else:
                    logger.info("Starting model training")
                    history = network.train()
                    excel_stats.save_train_history(network.train_point, history)
                    train_tracker.mark_train_complete(**network.train_point.dict())

                if train_tracker.is_point_evaluated(**network.train_point.dict()):
                    logger.warning("Model already evaluated on this point skip")
                else:
                    model = network.evaluate()
                    excel_stats.save_evaluation(model, network.train_point)
                    train_tracker.mark_evaluation_complete(**network.train_point.dict())


if __name__ == '__main__':
    main()
