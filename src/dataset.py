"""Dataset processing utils"""
from __future__ import annotations

import logging
import shutil
from multiprocessing import Event, Process, Queue, cpu_count
from pathlib import Path
from queue import Empty
from random import sample
from time import monotonic
from typing import Generator, NoReturn, Dict, Optional, Tuple
from itertools import chain
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use
from PIL import Image

from config import PlotSettings, PreProcessing
import tensorflow as tf


def image_processor(queue: Queue, shutdown: Event) -> NoReturn:
    """Consumer process
    Load, resize, normalize and save image

    Parameters
    ----------
    queue: Queue
        Message queue with image tuple path (source, destination) to process
    shutdown: Event
        Shutdown notification event. If queue is empty and event set exit from process

    References
    ----------
    Yaman, S., Karakaya, B. & Erol, Y.
     A novel normalization algorithm to facilitate pre-assessment of Covid-19 disease by improving accuracy of CNN
     and its FPGA implementation. Evolving Systems (2022). https://doi.org/10.1007/s12530-022-09419-3
    """
    _settings = PreProcessing()
    while not shutdown.is_set():
        try:
            img_source, img_destination = queue.get(block=True, timeout=1)
        except Empty:
            continue
        # Open
        img = Image.open(img_source)
        # Resize
        resized_img = img.resize(
            size=_settings.image_size, resample=Image.LANCZOS
        ).convert(mode="L")
        # Normalize
        image_array = np.asarray(resized_img)
        normalized = (image_array - np.mean(image_array)) / np.std(image_array)
        scaled = (normalized - np.min(normalized)) / (
            np.max(normalized) - np.min(normalized)
        )
        scaled = scaled * 255
        norm_img = Image.fromarray(scaled.astype(np.int8), mode="L")
        # Save
        norm_img.save(img_destination, icc_profile=False)


def image_iterator(
    folder: Path, avoid_corrupted: bool = False, fraction: float = 1.0
) -> Generator[Path]:
    """Image iterator
    Print progress

    Parameters
    ----------
    folder: Path
        Folder path to search images
    avoid_corrupted: bool
        If set yields only images that matching folder name
    fraction: float
        Return only fraction of total files. If greater than 1 return oversampled list

    Returns
    -------
    Generator[Path]
        Scan all files in folder

    Yields
    ------
    Path
        Image path
    """
    logger = logging.getLogger("raido.pre_process")
    start = monotonic()
    step = start

    file_list = [
        file.absolute()
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix in [".png", ".jpg", ".jpeg"]
        and (file.name.startswith(folder.name) or (not avoid_corrupted))
    ]
    total_images = len(file_list)
    logger.info("Found %s images in folder %s", total_images, folder)
    status_string = f"Done: 0/{total_images} - 0.00%"
    print(status_string, end="")
    rng = np.random.default_rng()
    iterators = []
    while fraction > 0:
        iterators.append(
            rng.choice(
                file_list,
                size=int(min(fraction, 1.0) * total_images),
                replace=False,
            )
        )
        fraction -= 1

    for im_num, im_path in enumerate(chain(*iterators)):
        yield im_path
        if monotonic() - step > 1:
            print("\b" * len(status_string), end="")
            status_string = (
                f"Done: {im_num}/{total_images} - {im_num / (total_images / 100):.2f}%"
            )
            print(status_string, end="")
            step = monotonic()
    print("\b" * len(status_string), end="")
    logger.info(
        "Completed. Total processing time %.2f seconds [%.2f milli-seconds per image]",
        monotonic() - start,
        (monotonic() - start) / total_images * 1000,
    )


def class_interator(folder: Path):
    """Scan and iterate dataset classes

    Parameters
    ----------
    folder: Path
        Folder to scan. Assumes that inherent folders are class folders

    Yields
    ------
    Path
        Class folder path
    """
    logger = logging.getLogger("raido.pre_process")
    classes = [folder for folder in folder.iterdir() if folder.is_dir()]
    classes.sort()
    logger.info(
        "Found %s classes. Labels: %s",
        len(classes),
        ",".join([_class.name for _class in classes]),
    )
    for _folder in classes:
        logger.info("Processing class: %s", _folder.name)
        yield _folder


def prepare_dataset(export_path: Path) -> None:
    """Prepare dataset
    Must be called first

    Parameters
    ----------
    export_path: Path
        Where to store sample images
    """
    logger = logging.getLogger("raido.pre_process")
    _settings = PreProcessing()
    source_folder = Path("data", "raw")
    logger.info("Removing all existing data in processed and test folders")
    processed_folder = Path("data", "processed").absolute()
    test_folder = Path("data", "test").absolute()
    for folder in [processed_folder, test_folder]:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    logger.info("Starting Image processing workers")
    work_queue = Queue(cpu_count() * 2)
    exit_event = Event()
    workers = [
        Process(
            target=image_processor,
            kwargs={"queue": work_queue, "shutdown": exit_event},
        )
        for _ in range(cpu_count())
    ]
    for worker in workers:
        worker.start()
    logger.info("Processing all images")
    for class_path in class_interator(source_folder):
        class_name = class_path.name.lower()
        (processed_folder / Path(class_name)).mkdir()
        total_imgs = 0
        for image_num, source_path in enumerate(image_iterator(class_path)):
            destination_path = processed_folder / Path(
                class_name, f"{class_name}_{image_num:05d}.png"
            )
            work_queue.put((source_path, destination_path))
            total_imgs += 1
        logger.info("Total images for class: %s - %s", class_name, total_imgs)
    logger.info("Waiting to processing ends")
    exit_event.set()
    for worker in workers:
        worker.join()
    logger.info("Image processing complete")
    logger.info(
        "Removing all data in test folder and moving images from train folder to test"
    )
    for class_path in class_interator(processed_folder):
        class_name = class_path.name.lower()
        (test_folder / Path(class_name)).mkdir()
        images_path = list(image for image in image_iterator(class_path))
        images_path.sort(key=lambda x: x.name)
        test_images = [
            images_path[i * len(images_path) // _settings.test_images_per_class]
            for i in range(_settings.test_images_per_class)
        ]
        for image_num, img_path in enumerate(test_images):
            destination_path = test_folder / Path(
                class_name, f"{class_name}_{image_num:05d}.png"
            )
            shutil.move(src=img_path, dst=destination_path)
    save_dataset_sample(
        export_path=export_path / "train_dataset.png", search_path=processed_folder
    )
    save_dataset_sample(
        export_path=export_path / "test_dataset.png", search_path=test_folder
    )


def save_dataset_sample(export_path: Path, search_path: Path):
    """Create sample images from data source

    Parameters
    ----------
    export_path: Path
        Where to save sample image
    search_path: Path
        Datasource path
    """
    logger = logging.getLogger("raido.pre_process")
    logger.info("Saving test dataset samples from %s", search_path)
    use("Agg")
    _plot_settings = PlotSettings()
    fig = plt.figure(figsize=_plot_settings.plot_size, dpi=_plot_settings.plot_dpi)
    fig.tight_layout()
    sub_figs = fig.subfigures(nrows=len(list(class_interator(search_path))), ncols=1)
    for class_path, sub_fig in zip(class_interator(search_path), sub_figs):
        class_name = class_path.name.capitalize()
        images = sample(list(image_iterator(class_path)), 5)
        sub_fig.suptitle(class_name)
        subplot = sub_fig.subplots(nrows=1, ncols=5)
        for image_path, _plot in zip(images, subplot):
            image = Image.open(image_path)
            _plot.imshow(image, cmap="gray", vmin=0, vmax=255)
            _plot.axis("off")

    plt.savefig(export_path)


def cook_dataset(
    reduction: float,
    balance: bool,
    corruption: Dict[Tuple[str, str], float],
) -> Tuple[float, ...]:
    """Create dataset source

    Parameters
    ----------
    reduction: float
        Dataset reduction fraction [0-1]
    balance: bool
        If set the dataset will be balanced using over-sample.
    corruption: Dict[Tuple[str, str], float]
        Labels to corrupt as keys (existing class, new class) and corruption faction as value
    export_path: Optional[Path]
        Folder to save dataset info

    Returns
    -------
    Tuple[float, ...]
        Class balance
    """
    logger = logging.getLogger("raido.pre_process")
    logger.info("Removing all existing data in cooked folder")
    processed_folder = Path("data", "processed").absolute()
    cooked_folder = Path("data", "cooked").absolute()
    shutil.rmtree(cooked_folder, ignore_errors=True)
    cooked_folder.mkdir(parents=True)
    logger.info("Copy images to cooked folder with reduction %s", reduction)
    for class_folder in class_interator(processed_folder):
        logger.info("Processing class %s", class_folder.name)
        target_folder = cooked_folder / class_folder.name
        target_folder.mkdir()
        for source_image in image_iterator(
            folder=class_folder, fraction=(1 - reduction)
        ):
            shutil.copy(source_image, target_folder)
    logger.info("Corrupting labels")
    for (source_class, target_class), fraction in corruption.items():
        for source_image in image_iterator(
            folder=cooked_folder / source_class, avoid_corrupted=True, fraction=fraction
        ):
            shutil.move(source_image, cooked_folder / target_class, shutil.copy)
    logger.info("Calculating class sizes")
    images_count = [
        len(list(image_iterator(folder=_folder)))
        for _folder in class_interator(cooked_folder)
    ]
    if balance:
        logger.info("Balancing dataset")
        for class_folder, count in zip(class_interator(cooked_folder), images_count):
            for source_image in image_iterator(
                folder=class_folder,
                avoid_corrupted=False,
                fraction=(max(images_count) / count) - 1.0,
            ):
                suffix = 0
                target_image = source_image.with_name(
                    source_image.stem + f"_{suffix}" + source_image.suffix
                )
                while target_image.exists():
                    suffix += 1
                    target_image = source_image.with_name(
                        source_image.stem + f"_{suffix}" + source_image.suffix
                    )
                shutil.copy(source_image, target_image)
    logger.info("Calculating weights")
    images_count = [
        len(list(image_iterator(folder=_folder)))
        for _folder in class_interator(cooked_folder)
    ]
    total = sum(images_count)
    weights = [(1.0 / count) * (total / len(images_count)) for count in images_count]
    print(weights)
    return weights


def save_dataset_statistics(export_path: Path) -> None:
    """Save dataset statistics

    Parameters
    ----------
    export_path: Path
        Folder path to save statistics
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Exporting statistics")
    cooked_folder = Path("data", "cooked").absolute()
    processed_folder = Path("data", "processed").absolute()
    _plot_settings = PlotSettings()
    status = {}
    counts = []
    names = []
    source_count = [
        len(list(image_iterator(class_path)))
        for class_path in class_interator(processed_folder)
    ]
    for index, class_path in enumerate(class_interator(cooked_folder)):
        status[class_path.name] = {}
        names.append(class_path.name.capitalize())
        counts.append([0, 0])
        for image_path in image_iterator(class_path):
            status[class_path.name]["total"] = (
                status[class_path.name].get("total", 0) + 1
            )
            # Total class images
            if class_path.stem[-2] == "_":
                status[class_path.name]["oversampled"] = (
                    status[class_path.name].get("oversampled", 0) + 1
                )
            if (
                image_path.name.startswith(class_path.name)
                and image_path.stem[-2] != "_"
            ):
                status[class_path.name]["original"] = (
                    status[class_path.name].get("original", 0) + 1
                )
            if image_path.name.startswith(class_path.name):
                status[class_path.name]["correct"] = (
                    status[class_path.name].get("correct", 0) + 1
                )
                counts[index][0] += 1
            else:
                status[class_path.name]["corrupted"] = (
                    status[class_path.name].get("corrupted", 0) + 1
                )
                counts[index][1] += 1
        if status[class_path.name].get("corrupted", 0) > 0:
            status[class_path.name]["corruption"] = status[class_path.name][
                "corrupted"
            ] / (status[class_path.name]["total"] / 100.0)
        else:
            status[class_path.name]["corruption"] = 0

    use("Agg")
    fig = plt.figure(figsize=_plot_settings.plot_size, dpi=_plot_settings.plot_dpi)
    fig.tight_layout()
    fig, ax = plt.subplots()
    size = 0.3
    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap([1, 2, 5, 6, 9, 10])
    all_classes = np.array(counts).sum(axis=1)
    wedges, texts = ax.pie(
        all_classes,
        radius=1,
        colors=outer_colors,
        wedgeprops={"width": size, "edgecolor": "w"},
    )
    ax.pie(
        np.array(counts).flatten(),
        radius=1 - size,
        colors=inner_colors,
        wedgeprops={"width": size, "edgecolor": "w"},
        labeldistance=0.7,
        autopct=lambda x: f"{x:.1f}%\n({int(np.round(x/100.*np.sum(np.array(counts).flatten()))):d})",
    )
    ax.set(aspect="equal", title="Dataset balance and corruption")
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(
            f"{np.array(counts[i]).sum() / (np.array(counts).flatten().sum() / 100):.1f}%",
            xy=(x, y),
            xytext=(1.35 * np.sign(x), 1.4 * y),
            horizontalalignment=horizontalalignment,
            **kw,
        )
    plt.legend(
        loc=(0.8, -0.1),
        labels=[
            f"{name}({sub_name})"
            for sub_name in ["corrected", "corrupted"]
            for name in names
        ],
    )
    plt.savefig(export_path / "dataset_balance.png")
    plt.close()

    width = 0.5
    fig = plt.figure(figsize=_plot_settings.plot_size, dpi=_plot_settings.plot_dpi)
    fig.tight_layout()
    fig, ax = plt.subplots()
    width = 0.25  # the width of the bars
    multiplier = 0

    metrics = {}
    metrics["Source"] = source_count
    metrics["Unique"] = [count["original"] for count in status.values()]
    metrics["Total used"] = [count["total"] for count in status.values()]

    x = np.arange(len(names))  # the label locations

    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Number of images")
    ax.set_title("Dataset")
    ax.set_xticks(x + width, names)
    ax.legend(loc="upper left", ncols=3)

    plt.savefig(export_path / "dataset_usage.png")
    plt.close()

    with open(
        export_path / "dataset_statistics.json", encoding="utf-8", mode="w"
    ) as json_fh:
        json.dump(status, json_fh, indent=4)


def get_dataset(training: bool, batch: Optional[int] = None) -> tf.data.Dataset:
    """Load dataset
    Prefetch and batch dataset is batch provided

    Parameters
    ----------
    training: bool
        If set return training dataset, otherwise validation dataset returned
    batch: Optional[int] = None
        If provided batch dataset

    Returns
    ------
    tf.data.Dataset
        Loaded dataset
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Loading dataset: training %s, batch %s", training, batch)
    _settings = PreProcessing()
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=f'{Path("data", "cooked").absolute()}',
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch,
        image_size=_settings.image_size,
        shuffle=True,
        seed=2820285082,
        validation_split=0.2,
        subset="training" if training else "validation",
    )
    if batch is not None:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_test_dataset(batch: Optional[int] = None) -> tf.data.Dataset:
    """Return test dataset.
    The dataset not prefetched

    Returns
    -------
    tf.data.Dataset
    ---------------
        Test dataset
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Loading dataset: batch %s", batch)
    _settings = PreProcessing()
    return tf.keras.utils.image_dataset_from_directory(
        directory=f'{Path("data", "test").absolute()}',
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch,
        image_size=_settings.image_size,
        shuffle=True,
        seed=2820285082,
    )
