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

    def __init__(self, kernel_size=4, strides=2, alpha=0.2):
        """
        Init the layers of the model.

        Args:
            kernel_size (int): Scalar or tuple, size of the kernel windows
            strides (int): Strides of the convulution
            alpha (float): LeakyReLU alpha param
        """
        super().__init__()

        self.conv_a = LeakyConvBlock(64, kernel_size, strides, alpha, True)
        self.conv_b = LeakyConvBlock(128, kernel_size, strides, alpha)
        self.conv_c = LeakyConvBlock(256, kernel_size, strides, alpha)
        self.conv_d = LeakyConvBlock(512, kernel_size, strides, alpha)
        self.conv_e = LeakyConvBlock(512, kernel_size, 1, alpha)
        self.conv_f = layers.Conv2D(1, kernel_size, 1, padding='same')

        self.activation = layers.Activation('sigmoid')

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (tf.Tensor): Dict of sharp/blur img w Shape [btch, h, w, ch]

        Returns:
            Probabilities Tensor of shape [batch, heigh, width, 1]
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
        outputs = self.activation(outputs)

        return outputs


class LeakyConvBlock(ConvBlock):
    """Conv Block with Leaky Relu."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        alpha,
        override_activation=False,
    ):
        """
        Replace ReLU with LeakyReLU.

        Args:
            filters (int): Number of filters of the Conv Layer
            kernel_size (int): Scalar or tuple, size of the kernel windows
            strides (int): Strides of the convulution
            alpha (float): LeakyReLU alpha param
            override_activation (bool): if calculate or not the relu activation
        """
        super().__init__(filters, kernel_size, override_activation)

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding='same',
        )
        self.relu = layers.LeakyReLU(alpha=alpha)
