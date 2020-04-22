#!/usr/bin/python
# coding=utf-8

"""Defines a model wrapper for image bytes Inputs and outputs."""

import tensorflow as tf
from tensorflow.keras import Model


class ImageByteWrapper(Model):
    """Wrapes the supplied model into a image bytes decoder/encoder."""

    def __init__(self, model):
        """
        Init the wrapper.

        Args:
            model (int): model to wrappe with image decoding/encoding
        """
        super().__init__()

        self.model = model

    def call(self, inputs):
        """
        Decode and Forward propagates the supplied image file.

        Args:
            inputs (tf.Tensor): string tensor shape [batch, 1]

        Returns:
            b64 Encoded output of the model
        """
        images = tf.unstack(inputs)
        images_list = []
        for image in images:
            image = tf.squeeze(image, axis=[0])
            image = tf.io.decode_image(image, dtype=tf.float32)
            image = (image - 127.0) / 128.0
            images_list.append(image)

        images = tf.stack(images_list)

        outputs = self.model(images)

        outputs = tf.unstack(outputs)
        outputs_list = []
        for output in outputs:
            output = (output * 128.0) + 127.0
            output = tf.cast(output, dtype=tf.uint8)
            output = tf.io.encode_jpeg(output)
            outputs_list.append(output)

        return outputs_list

    def string_wrapped(self):
        """Wrape the model under string inputs/outputs types."""
        inputs = tf.keras.layers.Inputs(dtype=tf.string)
        outputs = self(inputs)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
