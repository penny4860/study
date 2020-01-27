# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/load_data/images

import tensorflow as tf
import numpy as np
import os
import pathlib

IMG_WIDTH = 224
IMG_HEIGHT = 224


def process_path(file_path, class_names, img_width, img_height):
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == class_names

    def decode_img(img):
        # img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [img_width, img_height])

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
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == '__main__':
    DATASET_ROOT = "C://Users//penny//git//dataset//cifar100//train"

    # 1. Define class names
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(DATASET_ROOT).glob('*')])

    # 2. Build train_dataset
    files_ds = tf.data.Dataset.list_files(DATASET_ROOT + '/*/*')
    xy_ds = files_ds.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT),
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(xy_ds)

    xs, ys = next(iter(train_ds))
    print(xs.shape, ys.shape)
