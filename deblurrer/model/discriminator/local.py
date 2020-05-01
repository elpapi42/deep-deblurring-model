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
from tensorflow.keras.initializers import RandomNormal

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
        self.conv_e = LeakyConvBlock(512, kernel_size, alpha=alpha)
        self.conv_f = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer=RandomNormal(stddev=0.2),
        )

        self.activation = layers.Activation('sigmoid')

    def call(self, inputs):
        """
        Forward call of the Model.

        Args:
            inputs (tf.Tensor): with Shape [btch, h, w, 6]

        Returns:
            Averaged probs of the generated image to be real, shape [batch, 1]
        """
        # Conv feedforward
        outputs = self.conv_a(inputs)
        outputs = self.conv_b(outputs)
        outputs = self.conv_c(outputs)
        outputs = self.conv_d(outputs)
        outputs = self.conv_e(outputs)
        outputs = self.conv_f(outputs)

        # Sigmoid activation function
        outputs = self.activation(outputs)

        # Averages the output patches
        outputs = tf.math.reduce_mean(outputs, axis=[1, 2, 3])
        outputs = tf.reshape(outputs, [-1, 1])

        return outputs


class LeakyConvBlock(ConvBlock):
    """Conv Block with Leaky Relu."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        alpha=0.2,
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
        super().__init__(filters, kernel_size, strides, override_activation)

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding='same',
            kernel_initializer=RandomNormal(stddev=0.2),
        )
        self.relu = layers.LeakyReLU(alpha=alpha)
