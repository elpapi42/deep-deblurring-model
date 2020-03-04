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

    The input must be python dictionary with two keys, sharp/blur
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
            inputs (tf.Tensor): Dict of sharp/blur img w Shape [btch, h, w, ch]

        Returns:
            Probabilities Tensor of shape [batch, 2]
        """
        local = self.local(inputs)
        dglobal = self.dglobal(inputs)

        # Stack and puts batch dim first
        outputs = tf.stack([dglobal, local])
        outputs = tf.squeeze(outputs, axis=[-1])
        outputs = tf.transpose(outputs, [1, 0])

        return outputs
