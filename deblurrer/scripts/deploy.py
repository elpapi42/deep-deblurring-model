#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os

import tensorflow as tf
from dotenv import load_dotenv

from deblurrer.model.wrapper import ImageByteWrapper
from deblurrer.scripts.train import run as train


def wrap(gan):
    """
    Extracts the generator rom the supplied GAN.
    Wraps it with bytes encoder/decoder and b64 encoder.

    Args:
        gan (tf.keras.Model): GAN Model

    Returns:
        Wrapped and initialized model generator
    """
    wrapped_generator = ImageByteWrapper(model.generator)

    # Run a forward pass for init the inputs of the wrapper
    test_input = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=255, dtype=tf.int32)
    test_input = tf.cast(test_input, dtype=tf.uint8)
    test_input = tf.io.encode_jpeg(test_input)
    test_input = tf.reshape(test_input, [1, -1])

    with tf.device('/cpu:0'):
        b64_output = wrapped_generator(test_input)
    
    return wrapped_generator



def run(gan):
    generator = wrap(gan)

if (__name__ == '__main__'):
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__),
            ),
        ),
    )

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

    generator = wrap(model)

    generator.summary()
