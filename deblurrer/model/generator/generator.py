#!/usr/bin/python
# coding=utf-8

"""
GAN generator architecture.

We will use MobileNetV2 as Backbone
Any other Sota Arch can be used
like Resnet or Inception
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.generator import FPN, FPNConvBlock, ConvBlock


class FPNGenerator(Model):
    """Define the FPN Generator Arch."""

    def __init__(self, channels=128):
        """
        Init the Generator instance.

        Args:
            channels (int): Number of std channels the FPN will manage
        """
        super().__init__()

        # Feature Pyramidal Network
        self.fpn = FPN(channels)

        # Scalers
        self.scale_a = layers.UpSampling2D([4, 4], interpolation='bilinear')
        self.scale_bc = layers.UpSampling2D([2, 2], interpolation='bilinear')

        # Image size restoration layers
        self.conv_res = ConvBlock(channels, 5)
        self.conv_up = FPNConvBlock(channels, 5)
        self.residual_up = layers.UpSampling2D([4, 4], interpolation='bilinear')
        self.residual_conv = ConvBlock(3, 5, override_activation=True)
        self.residual_act = layers.Activation('tanh')

    def call(self, inputs):
        """
        Forward pass of the Model.

        Args:
            inputs (tf.Tensor): Image, shape [batch, heigh, width, channels]

        Returns:
            4D tensor with shape [batch, heigh, width, channels]
        """
        # Call the FPN
        fpn_out = self.fpn(inputs)

        # Concat 0 to 3 outputs channels wise, and feed them to convblock
        residual = self.conv_res(
            tf.concat(
                [
                    self.scale_a(fpn_out[0]),
                    self.scale_bc(fpn_out[1]),
                    self.scale_bc(fpn_out[2]),
                    tf.cast(fpn_out[3], dtype=tf.float32),
                ],
                axis=-1,
            ),
        )

        # Size restoration layers
        residual = self.conv_up([residual, fpn_out[4]])
        residual = self.residual_up(residual)
        residual = self.residual_conv(residual)
        residual = self.residual_act(residual)

        return tf.add(inputs, residual)
