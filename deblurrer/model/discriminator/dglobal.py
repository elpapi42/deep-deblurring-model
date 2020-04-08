#!/usr/bin/python
# coding=utf-8

"""
Global Discriminator side of the Discriminator.

Analize the full image for prdict if its generated or not
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.discriminator import LeakyConvBlock
from deblurrer.model.generator import ConvBlock


class GlobalDiscriminator(Model):
    """
    Global Discriminator side of the Discriminator.

    Analize the full image for prdict if its generated or not
    """

    def __init__(self, filters=12, conv_count=4, alpha=0.2):
        """
        Init the layers of the model.

        Args:
            filters (int): Base number of filters, scaled by conv_count
            conv_count (int): Number of leaky conv layers to use
            alpha (float): LeakyReLU alpha param
        """
        super().__init__()

        self.conv_in = ConvBlock(filters, 5, 2, override_activation=True)

        # Interpolate kernel size from 5 to 2 over all the layers
        kernels = np.interp(
            x=[index / (conv_count - 1) for index in range(conv_count)],
            xp=[0.0, 1.0],
            fp=[5.0, 2.0],
        )
        kernels = np.round(kernels).astype(int)

        # Stack the con layers into conv_layers
        self.conv_layers = []
        for index, kernel in zip(range(2, conv_count + 2), kernels):
            self.conv_layers.append(
                LeakyConvBlock(
                    filters=index * filters,
                    kernel_size=(kernel, kernel),
                    strides=2,
                    alpha=alpha,
                ),
            )

        self.conv_out = layers.Conv2D(64, 2, 2, padding='same')

        self.global_pool = layers.GlobalMaxPool2D()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (tf.Tensor): with Shape [btch, h, w, 6]

        Returns:
            Probabilities Tensor of shape [batch, 1]
        """
        # Conv feedforward
        outputs = self.conv_in(inputs)

        for conv in self.conv_layers:
            outputs = conv(outputs)

        outputs = self.conv_out(outputs)

        # Sigmoid activation function
        outputs = self.global_pool(outputs)
        outputs = self.dense(outputs)

        return outputs
