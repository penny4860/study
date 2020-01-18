# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)


def create_model(n_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    # layer의 output tensor
    x = base_model.layers[-1].output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model


if __name__ == '__main__':
    model = create_model(n_classes=10)
    model.summary()

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #               metrics=['accuracy'])


