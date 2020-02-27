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

from deblurrer.model.generator import FPN


class MobileNetV2Backbone(Model):
    """Define the MobileNetV2 Backbone."""

    def __init__(self, channels=128):
        """
        Init the Backbone instance.

        Args:
            channels (int): Number of std channels the FPN will manage
        """
        super().__init__()

        # MobileNet backbone
        self.fpn = FPN()

    def call(self, inputs):
        """
        Forward pass of the Model.

        Args:
            inputs (tf.Tensor): Input, shape [batch, heigh, width, channels]

        Returns:
            List of Tensors
        """
        return inputs
