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

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.string)])
    def call(self, inputs):
        """
        Decode and Forward propagates the supplied image file.

        Args:
            inputs (tf.Tensor): string tensor shape [batch, 1]

        Returns:
            b64 Encoded output of the model
        """
        def pre_input(image):
            image = tf.squeeze(image, axis=[0],)
            image = tf.io.decode_image(image, dtype=tf.float32)
            image = (image - 127.0) / 128.0
            return image

        images = tf.map_fn(pre_input, inputs, dtype=tf.float32)

        outputs = self.model(images)

        def post_output(output):
            output = (output * 128.0) + 127.0
            output = tf.cast(output, dtype=tf.uint8)
            output = tf.io.encode_jpeg(output)
            output = tf.io.encode_base64(output, pad=True)
            return output

        outputs = tf.map_fn(post_output, outputs, dtype=tf.string)

        return outputs

