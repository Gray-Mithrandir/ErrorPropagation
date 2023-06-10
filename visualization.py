"""Common utilities for plotting models"""
import json
import logging
from contextlib import redirect_stdout
from itertools import product

from collections.abc import Collection
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import use
from numpy import ndarray
from sklearn.metrics import classification_report, confusion_matrix

from config import PlotSettings, TrainInfo
from load_dataset import create_class_filter, get_test_dataset


def plot_model(model: tf.keras.models.Model, train_info: TrainInfo) -> None:
    """Save model info and graphs

    Parameters
    ----------
    model: Model
        Model to save
    train_info: TrainInfo
        Train information
    """
    logger = logging.getLogger("raido.dataset")
    settings = PlotSettings()
    # Save model
    logger.info(
        "Saving model plots - %s", train_info.report_path / f"{train_info.safe_name}"
    )
    with open(
        train_info.report_path / f"{train_info.safe_name}.txt",
        mode="w",
        encoding="utf-8",
    ) as summary_fh:
        with redirect_stdout(summary_fh):
            model.summary(line_length=256)
    tf.keras.utils.plot_model(
        model,
        to_file=str(train_info.report_path / f"{train_info.safe_name}.png"),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=settings.plot_dpi,
        layer_range=None,
        show_layer_activations=True,
    )


def plot_history(history: tf.keras.callbacks.History, train_info: TrainInfo) -> None:
    """Save accuracy and validation plots

    Parameters
    ----------
    history: History
        History object from 'model.fit'
    train_info: TrainInfo
        Model train information
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

    plt.savefig(train_info.report_path / "train_history.png")
    plt.close()


def plot_test_dataset_evaluation(
    model: tf.keras.models.Model, train_info: TrainInfo
) -> None:
    """Evaluate model on test set

    Parameters
    ----------
    model: Model
        Model to evaluate
    train_info: TrainInfo
        Model train information
    """
    sample_images = 3
    plot_settings = PlotSettings()
    probability_model = tf.keras.Model([model, tf.keras.layers.Softmax()], inputs=(1, 128, 128, 1)4)
    test_ds = get_test_dataset()
    class_names = test_ds.class_names

    use("Agg")
    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()
    subfigs = fig.subfigures(nrows=len(class_names), ncols=1)
    for class_index, (class_name, subfig) in enumerate(zip(class_names, subfigs)):
        subfig.suptitle(class_name.capitalize())
        axs = subfig.subplots(nrows=1, ncols=2 * sample_images)

        class_ds = test_ds.filter(create_class_filter(class_index)).take(sample_images)
        for index, (image, class_tensor) in enumerate(iter(class_ds)):
            true_label = class_names[np.argmax(class_tensor.numpy())]
            predictions = probability_model.predict(tf.data.Dataset.from_tensors(image).batch(1), verbose=0).numpy()
            predict_label = class_names[int(np.argmax(predictions))]

            axs[2 * index].grid(False)
            axs[2 * index].axes.get_xaxis().set_ticks([])
            axs[2 * index].axes.get_yaxis().set_ticks([])
            axs[2 * index].imshow(image.numpy(), cmap=plt.get_cmap("binary"))
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
            axs[2 * index + 1].axes.get_xaxis().set_ticks(range(len(class_names)))
            axs[2 * index + 1].axes.get_yaxis().set_ticks(
                list(tick for tick in np.arange(0, 1, 0.1))
            )
            this_plot = axs[2 * index + 1].bar(
                range(len(class_names)),
                predictions.flatten(),
                color="#777777",
            )
            axs[2 * index + 1].axes.get_yaxis()._set_lim(0, 1, auto=False)
            this_plot[class_index].set_color(color)
            axs[2 * index + 1].tick_params(axis="both", which="major", labelsize=6)

    plt.savefig(train_info.report_path / "evaluation.png")
    plt.close()


def plot_confusion_matrix(
    y_true: Collection[ndarray],
    y_pred: Collection[ndarray],
    labels: Collection[str],
    train_info: TrainInfo,
    prefix: str,
) -> None:
    """Save confusion matrix

    Parameters
    ----------
    y_true: list[ndarray[int]]
        True labels
    y_pred: list[ndarray[int]]
        Predicted labels
    labels: Iterable[str]
        Classes names
    train_info: TrainInfo
        Model training information
    prefix: str
        Filename prefix
    """
    plot_settings = PlotSettings()
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
    plt.savefig(train_info.report_path / f"confusion_matrix_{prefix}.png")
    plt.close()
    metric_path = train_info.report_path / f"extra_metrics_{prefix}.txt"
    additional_metrics = classification_report(y_true, y_pred, target_names=labels)
    metric_path.write_text(additional_metrics, encoding="utf-8")
    metric_json_path = train_info.report_path / f"extra_metrics_{prefix}.json"
    additional_metrics = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    with open(metric_json_path, "w", encoding="utf-8") as metric_fh:
        json.dump(additional_metrics, metric_fh, indent=4)
