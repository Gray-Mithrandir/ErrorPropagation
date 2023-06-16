"""Common utilities for plotting models"""
import json
import logging
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import use
from sklearn.metrics import classification_report, confusion_matrix

from config import PlotSettings
from load_dataset import create_class_filter


def plot_model(model: tf.keras.models.Model, export_path: Path) -> None:
    """Save model info and graphs

    Parameters
    ----------
    model: Model
        Model to save
    export_path: Path
        Where to export model plots
    """
    logger = logging.getLogger("raido.dataset")
    settings = PlotSettings()
    # Save model
    logger.info("Saving model plots - %s", export_path)
    with open(
        export_path / "model.txt",
        mode="w",
        encoding="utf-8",
    ) as summary_fh:
        with redirect_stdout(summary_fh):
            model.summary(line_length=128)
    tf.keras.utils.plot_model(
        model,
        to_file=str(export_path / "model.png"),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=settings.plot_dpi,
        layer_range=None,
        show_layer_activations=True,
    )


def plot_history(history: tf.keras.callbacks.History, export_path: Path) -> None:
    """Save accuracy and validation plots

    Parameters
    ----------
    history: History
        History object from 'model.fit'
    export_path: Path
        Where to save plot
    """
    plot_settings = PlotSettings()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    use("Agg")
    _, big_axes = plt.subplots(
        figsize=plot_settings.plot_size,
        dpi=plot_settings.plot_dpi,
        nrows=2,
        ncols=1,
        sharex=True,
    )
    big_axes[0].plot(epochs_range, acc, label="Training Accuracy")
    big_axes[0].plot(epochs_range, val_acc, label="Validation Accuracy")
    big_axes[0].legend(loc="lower right")
    big_axes[0].set_title("Training and Validation Accuracy")

    big_axes[1].plot(epochs_range, loss, label="Training Loss")
    big_axes[1].plot(epochs_range, val_loss, label="Validation Loss")
    big_axes[1].legend(loc="upper right")
    big_axes[1].set_title("Training and Validation Loss")

    plt.savefig(export_path)
    plt.close()


def plot_performance(
    model: tf.keras.models.Model,
    dataset: tf.data.Dataset,
    labels: Tuple[str, ...],
    num_images: int,
    export_path: Path,
) -> None:
    """Plot sample images and model prediction

    Parameters
    ----------
    model: Model
        Model to evaluate
    dataset: tf.data.Dataset
        Dataset to evaluate on
    labels: Tuple[str, ...]
        Labels to plot
    num_images: int
        Number of images to plot
    export_path: Path
        Filepath to save
    """
    plot_settings = PlotSettings()

    use("Agg")
    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()
    subfigs = fig.subfigures(nrows=len(labels), ncols=1)
    for class_index, (class_name, subfig) in enumerate(zip(labels, subfigs)):
        subfig.suptitle(class_name.capitalize())
        axs = subfig.subplots(nrows=1, ncols=2 * len(labels))

        class_ds = dataset.filter(create_class_filter(class_index)).take(num_images)
        for index, (image, class_tensor) in enumerate(iter(class_ds.batch(1))):
            true_label = labels[np.argmax(class_tensor.numpy())]
            predictions = _numpy_softmax(model.predict(image), axis=1)
            predict_label = labels[int(np.argmax(predictions, axis=1))]

            axs[2 * index].grid(False)
            axs[2 * index].axes.get_xaxis().set_ticks([])
            axs[2 * index].axes.get_yaxis().set_ticks([])
            axs[2 * index].imshow(image[0].numpy(), cmap=plt.get_cmap("binary"))
            if true_label == predict_label:
                color = "blue"
            else:
                color = "red"

            axs[2 * index].set_xlabel(
                f"{predict_label} {100 * np.max(predictions):.02f}%",
                color=color,
                fontsize=6,
            )

            axs[2 * index + 1].grid(False)
            axs[2 * index + 1].axes.get_xaxis().set_ticks(range(len(labels)))
            axs[2 * index + 1].axes.get_yaxis().set_ticks(
                list(tick for tick in np.arange(0, 1, 0.1))
            )
            this_plot = axs[2 * index + 1].bar(
                range(len(labels)),
                predictions.flatten(),
                color="#777777",
            )
            axs[2 * index + 1].axes.get_yaxis()._set_lim(0, 1, auto=False)
            this_plot[class_index].set_color(color)
            axs[2 * index + 1].tick_params(axis="both", which="major", labelsize=6)

    plt.savefig(export_path)
    plt.close()


def plot_confusion_matrix(
    model: tf.keras.models.Model,
    dataset: tf.data.Dataset,
    labels: Tuple[str, ...],
    export_path: Path,
) -> None:
    """Plot confusion matrix

    Parameters
    ----------
    model: Model
        Model to evaluate
    dataset: tf.data.Dataset
        Dataset to evaluate on
    labels: Tuple[str, ...]
        Labels to plot
    export_path: Path
        Filepath to save
    """
    plot_settings = PlotSettings()
    y_true = []
    y_pred = []
    for image, class_tensor in iter(dataset.batch(1)):
        y_true.append(np.argmax(class_tensor))
        y_pred.append(np.argmax(model.predict(image), axis=1))
    result = confusion_matrix(y_true, y_pred, normalize="pred")

    accuracy = np.trace(result) / np.sum(result).astype("float")
    misclass = 1 - accuracy

    use("Agg")
    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()
    plt.imshow(result, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    result = result.astype("float") / result.sum(axis=1)[:, np.newaxis]

    thresh = result.max() / 1.5
    for i, j in product(range(result.shape[0]), range(result.shape[1])):
        plt.text(
            j,
            i,
            f"{result[i, j]:0.4f}",
            horizontalalignment="center",
            color="white" if result[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    plt.savefig(export_path)
    plt.close()


def save_classification_report(
    model: tf.keras.models.Model,
    dataset: tf.data.Dataset,
    labels: Tuple[str, ...],
    export_path: Path,
) -> None:
    """Export classification report

    Parameters
    ----------
    model: Model
        Model to evaluate
    dataset: tf.data.Dataset
        Dataset to evaluate on
    labels: Tuple[str, ...]
        Labels to plot
    export_path: Path
        Filepath to save
    """
    y_true = []
    y_pred = []
    for image, class_tensor in iter(dataset.batch(1)):
        y_true.append(np.argmax(class_tensor))
        y_pred.append(np.argmax(model.predict(image), axis=1))
    additional_metrics = classification_report(y_true, y_pred, target_names=labels)
    export_path.with_suffix(".txt").write_text(additional_metrics, encoding="utf-8")
    additional_metrics = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    with open(export_path.with_suffix(".json"), "w", encoding="utf-8") as metric_fh:
        json.dump(additional_metrics, metric_fh, indent=4)


def _numpy_softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)