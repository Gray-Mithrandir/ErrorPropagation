"""Common utilities for plotting models"""
import json
import logging
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import use
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf
from sklearn.metrics import classification_report, confusion_matrix
from dataset import class_interator, image_iterator, get_dataset, get_test_dataset

from config import PlotSettings, PreProcessing


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
    export_path: Path,
) -> None:
    """Plot sample images and model prediction

    Parameters
    ----------
    model: Model
        Model to evaluate
    export_path: Path
        Filepath to save
    """
    _settings = PreProcessing()
    use("Agg")
    labels = [
        class_path.name.capitalize()
        for class_path in class_interator(Path("data", "processed").absolute())
    ]
    plot_settings = PlotSettings()
    for plot_name, dataset_folder in zip(
            ("performance_on_validation_ds.png", "performance_on_test_ds.png"),
            (Path("data", "cooked").absolute(), Path("data", "test").absolute()),
    ):
        fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
        fig.tight_layout()
        subfigs = fig.subfigures(nrows=len(labels), ncols=1)
        for class_index, (class_path, subfig) in enumerate(zip(class_interator(dataset_folder), subfigs)):
            subfig.suptitle(class_path.name.capitalize())
            axs = subfig.subplots(nrows=1, ncols=2 * len(labels))
            class_ds = tf.keras.utils.image_dataset_from_directory(
                directory=f'{class_path}',
                label_mode=None,
                color_mode="grayscale",
                batch_size=1,
                image_size=_settings.image_size,
                shuffle=True,
                seed=2820285082,
            )
            for index, image in enumerate(iter(class_ds)):
                if index == 3:
                    break
                true_label = labels[class_index]
                predictions = _numpy_softmax(model.predict(image, verbose=0), axis=1)
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

        plt.savefig(export_path / plot_name)
        plt.close()


def plot_confusion_matrix(
    model: tf.keras.models.Model,
    export_path: Path,
) -> None:
    """Plot confusion matrix

    Parameters
    ----------
    model: Model
        Model to evaluate
    export_path: Path
        Filepath to save
    """
    use("Agg")
    plot_settings = PlotSettings()
    labels = [
        class_path.name.capitalize()
        for class_path in class_interator(Path("data", "processed").absolute())
    ]
    for plot_name, dataset in zip(
        ("confusion_matrix_on_validation_ds.png", "confusion_matrix_on_test_ds.png"),
        (get_dataset(training=False, batch=None), get_test_dataset(batch=None)),
    ):
        y_true = []
        y_pred = []
        for image, class_tensor in iter(dataset.batch(1)):
            y_true.append(np.argmax(class_tensor))
            y_pred.append(np.argmax(model.predict(image, verbose=0), axis=1))
        result = confusion_matrix(y_true, y_pred, normalize="pred")

        accuracy = np.trace(result) / np.sum(result).astype("float")
        misclass = 1 - accuracy

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
        plt.xlabel(
            f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}"
        )
        plt.savefig(export_path / plot_name)
        plt.close()


def save_classification_report(
    model: tf.keras.models.Model,
    export_path: Path,
) -> None:
    """Export classification report

    Parameters
    ----------
    model: Model
        Model to evaluate
    export_path: Path
        Filepath to save
    """
    labels = [
        class_path.name.capitalize()
        for class_path in class_interator(Path("data", "processed").absolute())
    ]
    for export_name, dataset in zip(
            ("performance_metrics_on_validation_ds", "performance_metrics_on_test_ds"),
            (get_dataset(training=False, batch=None), get_test_dataset(batch=None)),
    ):
        y_true = []
        y_pred = []
        for image, class_tensor in iter(dataset.batch(1)):
            y_true.append(np.argmax(class_tensor))
            y_pred.append(np.argmax(model.predict(image, verbose=0), axis=1))
        additional_metrics = classification_report(y_true, y_pred, target_names=labels)
        (export_path / export_name).with_suffix(".txt").write_text(additional_metrics, encoding="utf-8")
        additional_metrics = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
        with open((export_path / export_name).with_suffix(".json"), "w", encoding="utf-8") as metric_fh:
            json.dump(additional_metrics, metric_fh, indent=4)


def plot_summary(
    x_axis: Iterable[float],
    y_axis: Iterable[float],
    values: Iterable[Iterable[float]],
    titles: Iterable[str],
    invert_bar: bool,
    export_path: Path,
) -> None:
    """Plot run summary

    Parameters
    ----------
    x_axis:Iterable[float]
        X axis values (data reduction percent)
    y_axis: Iterable[float]
        Y axis values (label corruption percent)
    values: Iterable[Iterable[float]]
        List of performance metrics
    titles: Iterable[str]
        List of metrics names. Must be same length as `values`
    invert_bar: bool
        If set inverse color bar
    export_path: Path
        Filename to save plot
    """
    _settings = PlotSettings()
    use("Agg")
    fig = plt.figure(figsize=_settings.plot_size, dpi=_settings.plot_dpi)
    fig.tight_layout()
    subfigs = fig.subplots(nrows=len(titles), ncols=1, sharex=True)

    x = np.array(x_axis)
    y = np.array(y_axis)
    for index, (z_vals, title, subfig) in enumerate(zip(values, titles, subfigs)):
        z = np.array(z_vals)
        if index == len(titles) - 1:
            subfig.set_xlabel("Reduction")
        subfig.set_ylabel("Corruption")
        subfig.set_title(title)
        xi, yi = np.linspace(x.min(), x.max(), np.unique(x).size), np.linspace(
            y.min(), y.max(), np.unique(y).size
        )
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = Rbf(x, y, z, function="linear")
        zi = rbf(xi, yi)

        surf = subfig.imshow(
            zi,
            vmin=z.min(),
            vmax=z.max(),
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=plt.get_cmap("RdBu_r") if invert_bar else plt.get_cmap("RdBu"),
        )
        divider = make_axes_locatable(subfig)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(surf, cax=cax)
    plt.savefig(export_path)
    plt.close()


def _numpy_softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
