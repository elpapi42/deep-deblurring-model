#!/usr/bin/python
# coding=utf-8

"""
Global Discriminator side of the Discriminator.

Analize the full image for prdict if its generated or not
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.discriminator import LeakyConvBlock


class GlobalDiscriminator(Model):
    """
    Global Discriminator side of the Discriminator.

    Analize the full image for prdict if its generated or not
    """

    def __init__(self, alpha=0.2):
        """
        Init the layers of the model.

        Args:
            alpha (float): LeakyReLU alpha param
        """
        super().__init__()

        self.conv_a = LeakyConvBlock(32, 5, 2, alpha, True)
        self.conv_b = LeakyConvBlock(64, 5, 4, alpha)
        self.conv_c = LeakyConvBlock(128, 3, 2, alpha)
        self.conv_d = LeakyConvBlock(256, 3, 2, alpha)
        self.conv_e = LeakyConvBlock(512, 2, 2, alpha)
        self.conv_f = layers.Conv2D(64, 2, 2, padding='same')

        self.global_pool = layers.GlobalMaxPool2D()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (tf.Tensor): Dict of sharp/blur img w Shape [btch, h, w, ch]

        Returns:
            Probabilities Tensor of shape [batch, 1]
        """
        # Concat inputs in channels-wise
        outputs = tf.concat([inputs['sharp'], inputs['blur']], axis=-1)

        # Conv feedforward
        outputs = self.conv_a(outputs)
        outputs = self.conv_b(outputs)
        outputs = self.conv_c(outputs)
        outputs = self.conv_d(outputs)
        outputs = self.conv_e(outputs)
        outputs = self.conv_f(outputs)

        # Sigmoid activation function
        outputs = self.global_pool(outputs)
        outputs = self.dense(outputs)

        return outputs
