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
    """

    def __init__(self, filters=12, conv_count=4):
        """
        Init the layers of the model.

        Args:
            filters (int): Base number of filters, scaled by conv_count
            conv_count (int): Number of leaky conv layers to use
        """
        super().__init__()

        self.local = LocalDiscriminator()
        self.dglobal = GlobalDiscriminator(filters, conv_count)

    def call(self, inputs):
        """
        Forward call of the Model.

        Input[0] will stand for the sharp image of reference

        Args:
            inputs (list): two tensors shape [batch, height, width, chnls]

        Returns:
            Dict with local/global keys w/ tensors shape [batch, 1]
        """
        # Concat inputs in channels-wise
        inputs = tf.concat([inputs[1], inputs[0]], axis=-1)

        local = self.local(inputs)
        dglobal = self.dglobal(inputs)

        return {
            'local': local,
            'global': dglobal,
        }
