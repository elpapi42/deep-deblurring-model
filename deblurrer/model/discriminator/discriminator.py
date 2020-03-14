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
            inputs (tf.Tensor): Dict of sharp/generated img w Shape [btch, h, w, ch]

        Returns:
            Probabilities Tensor of shape [batch, 1]
        """
        local = self.local(inputs)
        dglobal = self.dglobal(inputs)

        # Returns mean of the losses between both networks
        outputs = tf.stack([local, dglobal], axis=2)
        outputs = tf.reduce_mean(outputs, [2])

        return outputs
