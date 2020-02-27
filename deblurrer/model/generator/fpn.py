#!/usr/bin/python
# coding=utf-8

"""
Define FPN Head.

The backbone is attached to this module
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model.generator import MobileNetV2Backbone


class FPN(Model):
    """Define DeblurGANV2 FPN HEAD."""

    def __init__(self, channels=128):
        """
        Init the model instance.

        Args:
            channels (int): Number of std channels the FPN will manage
        """
        super().__init__()

        self.backbone = MobileNetV2Backbone(output_channels=channels)

        # Up-Down Path layers
        self.conv_a = ConvBlock(channels, 3)
        self.conv_b = FPNConvBlock(channels, 3)
        self.conv_c = FPNConvBlock(channels, 3, override_upsample=True)
        self.conv_d = FPNConvBlock(channels, 5)
        self.conv_e = FPNConvBlock(channels, 5)

    def call(self, inputs):
        """
        Forward pass of the FPN Model.

        Args:
            inputs (tf.Tensor): blur image w/ shape [bch, h, w, ch]

        Returns:
            List of tensors
        """
        outputs = self.backbone(inputs)

        # Top-Down path
        conv_a = self.conv_a(outputs[4])
        conv_b = self.conv_b([conv_a, outputs[3]])
        conv_c = self.conv_c([conv_b, outputs[2]])
        conv_d = self.conv_d([conv_c, outputs[1]])
        conv_e = self.conv_e([conv_d, outputs[0]])

        return conv_a, conv_b, conv_c, conv_d, conv_e


class ConvBlock(Model):
    """
    Convolutional Block.

    Conv + Batch Norm + Relu
    """

    def __init__(
        self,
        filters,
        kernel_size,
        override_activation=False,
    ):
        """
        Init block layers.

        Args:
            filters (int): Number of filters of the Conv Layer
            kernel_size (int): Scalar or tuple, size of the kernel windows
            override_activation (bool): if calculate or not the relu activation
        """
        super().__init__()

        self.override_activation = override_activation

        self.conv = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU(max_value=6.0)

    def call(self, inputs):
        """
        Forward pass of the block.

        Args:
            inputs (tf.Tensor): tensor with shape [bch, h, w, ch]

        Returns:
            Tensor of shape [bch, h, w, ch]
        """
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)

        if (not self.override_activation):
            outputs = self.relu(outputs)

        return outputs


class FPNConvBlock(Model):
    """
    Conv block of FPN Head.

    Upsample first input to the scale of second input
    Add both tensors and feed them to ConvBlock
    """

    def __init__(
        self,
        filters,
        kernel_size,
        override_upsample=False,
    ):
        """
        Init block layers.

        Args:
            filters (int): Number of filters of the Conv Layer
            kernel_size (int): Scalar or tuple, size of the kernel windows
            override_upsample (bool): If dont do the upsample step
        """
        super().__init__()

        self.override_upsample = override_upsample

        self.upsample = layers.UpSampling2D()
        self.conv = ConvBlock(filters, kernel_size)

    def call(self, inputs):
        """
        Forward pass of the block.

        Args:
            inputs (tf.Tensor): list len=2, each tensor shape [bch, h, w, ch]

        Returns:
            Tesnor of shape [bch, h, w, ch]
        """
        upsampled = inputs[0]

        if (not self.override_upsample):
            upsampled = self.upsample(upsampled)

        outputs = tf.add(upsampled, inputs[1])
        outputs = self.conv(outputs)

        return outputs
