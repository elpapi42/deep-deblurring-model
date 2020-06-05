#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os
import pathlib

import tensorflow as tf
from dotenv import load_dotenv

from deblurrer.model.wrapper import ImageByteWrapper
from deblurrer.scripts.train import run as train

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

from tensorflow import keras
import numpy as np


def wrap(model):
    """
    Extracts the generator rom the supplied GAN.
    Wraps it with bytes encoder/decoder and b64 encoder.

    Args:
        model (tf.keras.Model): GAN Model

    Returns:
        Wrapped and initialized model generator
    """
    wrapped_generator = ImageByteWrapper(model)

    # Run a forward pass for init the inputs of the wrapper
    test_input = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=255, dtype=tf.int32)
    test_input = tf.cast(test_input, dtype=tf.uint8)
    test_input = tf.io.encode_jpeg(test_input)
    test_input = tf.reshape(test_input, [1, -1])

    with tf.device('/cpu:0'):
        b64_output = wrapped_generator(test_input)
    
    return wrapped_generator

def convert(model):
    """
    Converts the supplied model into tflite format.

    Applies quantization.

    Args:
        model (tf.keras.Model): Model to be converted yo tflite model

    Returns:
        quantized tflite model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    return tflite_model

def save(path, model):
    """
    This function writes the final tflite model to disk.

    Args:
        path (str): Where to store the flat buffer file
    """
    tflite_models_dir = pathlib.Path(path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/'mnist_model.tflite'
    tflite_model_file.write_bytes(model)



def run(gan, path):
    generator = wrap(gan)

    tflite_gen = convert(gan)

    save(path, tflite_gen)

    return tflite_gen

if (__name__ == '__main__'):
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__),
            ),
        ),
    )
    tflite_path = os.path.join(path, 'tflite')

    # Load .env vars
    dotenv_path = os.path.join(path, '.env')
    load_dotenv(dotenv_path)

    tfrec_path = os.path.join(
        path,
        os.path.join('datasets', 'tfrecords'),
    )

    output_path = os.path.join(path, 'output')

    logs_path = os.path.join(path, 'logs')

    model, _, _, _ = train(
        tfrec_path,
        output_folder=output_path,
        logs_folder=logs_path,
    )

    """
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels)
    )
    """

    generator = run(model.generator.fpn.backbone.backbone, tflite_path)
