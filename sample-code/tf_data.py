# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/load_data/images

import tensorflow as tf
import numpy as np
import os
import pathlib

IMG_WIDTH = 224
IMG_HEIGHT = 224


def get_dataset_root():
    data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                       fname='flower_photos',
                                       untar=True)
    data_dir = pathlib.Path(data_dir)
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    return data_dir


def process_path(file_path):
    CLASS_NAMES = np.array([item.name for item in dataset_root.glob('*') if item.name != "LICENSE.txt"])

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset_root = get_dataset_root()
    print(dataset_root)

    files_ds = tf.data.Dataset.list_files(str(dataset_root / '*/*'))
    xy_ds = files_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(xy_ds)
    batch_images, batch_labels = next(iter(train_ds))
    print(batch_images.shape, batch_labels.shape)

    # model.fit(train_ds)
