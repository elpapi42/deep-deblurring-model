#!/usr/bin/python
# coding=utf-8

"""
Local Discriminator side of the Discriminator.

Implementation of PatchGAN Discriminator
outputs an NxN matrix of probs
each prob maps to a 70x70 patch of the input image
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.generator import ConvBlock


class LocalDiscriminator(Model):
    """
    Local Discriminator side of the Discriminator.

    Implementation of PatchGAN Discriminator
    outputs an NxN matrix of probs
    each prob maps to a 70x70 patch of the input image
    """

    def __init__(self, kernel_size=4, alpha=0.2):
        """
        Init the layers of the model.

        Args:
            kernel_size (int): Scalar or tuple, size of the kernel windows
            alpha (float): LeakyReLU alpha param
        """
        super().__init__()

        self.conv_a = LeakyConvBlock(64, kernel_size, alpha)
        self.conv_b = LeakyConvBlock(128, kernel_size, alpha)
        self.conv_c = LeakyConvBlock(256, kernel_size, alpha)
        self.conv_d = LeakyConvBlock(512, kernel_size, alpha)
        self.conv_e = LeakyConvBlock(512, kernel_size, alpha)
        self.conv_f = layers.Conv2D(1, kernel_size, padding='same')
        self.activation = layers.Activation('sigmoid')

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (tf.Tensor): Shape [batch, heigh, width, channels]

        Returns:
            Probabilities Tensor of shape [batch, heigh, width, 1]
        """
        outputs = self.conv_a(inputs)
        outputs = self.conv_b(outputs)
        outputs = self.conv_c(outputs)
        outputs = self.conv_d(outputs)
        outputs = self.conv_e(outputs)
        outputs = self.conv_f(outputs)

        outputs = self.activation(outputs)

        return inputs


class LeakyConvBlock(ConvBlock):
    """Conv Block with Leaky Relu."""

    def __init__(
        self,
        filters,
        kernel_size,
        alpha,
        override_activation=False,
    ):
        """
        Replace ReLU with LeakyReLU.

        Args:
            filters (int): Number of filters of the Conv Layer
            kernel_size (int): Scalar or tuple, size of the kernel windows
            alpha (float): LeakyReLU alpha param
            override_activation (bool): if calculate or not the relu activation
        """
        super().__init__(filters, kernel_size, override_activation)

        self.relu = layers.LeakyReLU(alpha=alpha)
