# -*- coding: utf-8 -*-

import tensorflow as tf
from src.model import create_model
from src.tf_data import process_path, prepare_for_training

DATASET_ROOT = "C://Users//penny//git//dataset//cifar100//train"

if __name__ == '__main__':
    # 1. Define class names
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(DATASET_ROOT).glob('*')])

    # 2. Build train_dataset
    files_ds = tf.data.Dataset.list_files(DATASET_ROOT + '/*/*')
    xy_ds = files_ds.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT),
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = prepare_for_training(xy_ds)

    model = create_model(n_classes=100, base_model_trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(train_ds,
              steps_per_epoch=100)
