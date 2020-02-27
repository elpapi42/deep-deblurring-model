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

    def __init__(self):
        """Init the model instance."""
        super().__init__()

        self.backbone = MobileNetV2Backbone()

        self.conv = ConvBlock(64, 5)

    def call(self, inputs):
        """
        Forward pass of the FPN Model.

        Args:
            inputs (tf.Tensor): blur image w/ shape [bch, h, w, ch]

        Returns:
            Tensor of shape = inputs.shape
        """
        outputs = self.backbone(inputs)
        print(outputs[-1])
        outputs = self.conv(outputs[-1])
        print(outputs)

        return outputs


class ConvBlock(Model):
    """
    Convolutional Block.

    Conv + Batch Norm + Relu
    """

    def __init__(
        self,
        filters,
        kernel_size,
    ):
        """
        Init block layers.

        Args:
            filters (int): Number of filters of the Conv Layer
            kernel_size (int): Scalar or tuple, size of the kernel windows
        """
        super().__init__()

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
        outputs = self.relu(outputs)

        return outputs

