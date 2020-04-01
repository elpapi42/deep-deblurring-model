#!/usr/bin/python
# coding=utf-8

"""
Double Scale Discriminator.

Analize the full image for predict if its generated or not
parallel, check patch of the imput image and average its probs
of being real or generated
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.discriminator import LocalDiscriminator, GlobalDiscriminator


class DoubleScaleDiscriminator(Model):
    """
    Double Scale Discriminator.

    Analize the full image for predict if its generated or not
    parallel, check patch of the imput image and average its probs
    of being real or generated

    The input must be tensor with shape [batch, 2, height, width, chnls]
    """

    def __init__(self):
        """Init the layers of the model."""
        super().__init__()

        self.local = LocalDiscriminator()
        self.dglobal = GlobalDiscriminator()

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (list): two tensors shape [batch, height, width, chnls]

        Returns:
            Dict with local/global keys w/ tensors shape [batch, 1]
        """
        # Concat inputs in channels-wise
        inputs = tf.concat([inputs[0], inputs[1]], axis=-1)

        local = self.local(inputs)
        dglobal = self.dglobal(inputs)

        return {
            'local': local,
            'global': dglobal,
        }
