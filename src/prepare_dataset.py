"""Prepare dataset
Combine existing datasets split from raw folder into single set and create separated test set
"""
from __future__ import annotations

import logging
import shutil
from multiprocessing import Event, Process, Queue, cpu_count
from pathlib import Path
from queue import Empty
from random import sample
from time import monotonic
from typing import Generator, NoReturn

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use
from PIL import Image

from config import PlotSettings, PreProcessing
from logger import init_logger


def image_processor(queue: Queue, shutdown: Event) -> NoReturn:
    """Consumer process
    Load, process and save image

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


def image_iterator(folder: Path) -> Generator:
    """Image iterator
    Print progress

    Parameters
    ----------
    folder: Path
        Folder path to search images

    Returns
    -------
    Generator
        Scan all files in folder

    Yields
    ------
    Path
        Image path
    """
    logger = logging.getLogger("raido.pre_process")
    file_list = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix in [".png", ".jpg", ".jpeg"]
    ]
    file_list.sort()
    start = monotonic()
    step = start
    total_images = len(file_list)
    logger.info("Found %s images in folder %s", total_images, folder)
    status_string = f"Done: 0/{total_images} - 0.00%"
    print(status_string, end="")
    for im_num, im_path in enumerate(file_list):
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


def cook_dataset() -> None:
    """Prepare dataset
    Must be called first
    """
    logger = logging.getLogger("raido.pre_process")
    logger.info("Removing all existing data in cooked")
    train_folder = Path("data", "cooked").absolute()
    source_folder = Path("data", "raw").absolute()
    shutil.rmtree(train_folder, ignore_errors=True)
    train_folder.mkdir(parents=True)
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
        (train_folder / Path(class_name)).mkdir()
        total_imgs = 0
        for image_num, source_path in enumerate(image_iterator(class_path)):
            destination_path = train_folder / Path(
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


def create_test_set() -> None:
    """Create test set
    Since label corruption in hard to reverse this function must be called after each `create_train_set`
    """
    _settings = PreProcessing()
    logger = logging.getLogger("raido.pre_process")
    logger.info("Removing all data in test folder")
    test_folder = Path("data", "test").absolute()
    train_folder = Path("data", "cooked").absolute()
    shutil.rmtree(test_folder, ignore_errors=True)
    test_folder.mkdir(parents=True)
    logger.info("Moving images from train folder to test")
    for class_path in class_interator(train_folder):
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


def main():
    """Prepare dataset"""
    export_path = Path("logs", "pre_processing")
    test_folder = Path("data", "test")
    plot_path = Path("reports", "dataset_sample.png")
    plot_path.unlink(missing_ok=True)
    shutil.rmtree(export_path, ignore_errors=True)
    export_path.mkdir(parents=True)
    init_logger(export_path)
    logger = logging.getLogger("raido.pre_process")
    cook_dataset()
    create_test_set()

    logger.info("Saving test dataset samples")
    use("Agg")
    fig = plt.figure(figsize=PlotSettings.plot_size, dpi=PlotSettings.plot_dpi)
    fig.tight_layout()
    sub_figs = fig.subfigures(nrows=len(list(class_interator(test_folder))), ncols=1)
    for class_path, sub_fig in zip(class_interator(test_folder), sub_figs):
        class_name = class_path.name.capitalize()
        images = sample(list(image_iterator(class_path)), 5)
        sub_fig.suptitle(class_name)
        subplot = sub_fig.subplots(nrows=1, ncols=5)
        for image_path, _plot in zip(images, subplot):
            image = Image.open(image_path)
            _plot.imshow(image, cmap="gray", vmin=0, vmax=255)
            _plot.axis("off")

    plt.savefig(plot_path)


if __name__ == "__main__":
    main()
