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


def wrap(model):
    """
    Extracts the generator rom the supplied GAN.
    Wraps it with bytes encoder/decoder and b64 encoder.

    Args:
        model (tf.keras.Model): Generator Model

    Returns:
        Wrapped and initialized model generator
    """
    wrapped_generator = ImageByteWrapper(model)

    # Run a forward pass for init the inputs of the wrapper
    test_input = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=255, dtype=tf.int32)
    test_input = tf.cast(test_input, dtype=tf.uint8)
    test_input = tf.io.encode_jpeg(test_input)
    test_input = tf.stack([test_input, test_input])
    test_input = tf.reshape(test_input, [-1, 1])

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
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
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


def run(model, path):
    """
    Converts the supplied keras model to tflite.
    and write the flat buffer to the specified path.
    """
    wrapped = wrap(model)

    tflite = convert(wrapped)

    save(path, tflite)


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

    generator = run(model.generator, tflite_path)
