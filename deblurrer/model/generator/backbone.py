#!/usr/bin/python
# coding=utf-8

"""
Conv Backbone of generator.

We will use MobileNetV2
Any other Sota Arch can be used
like Resnet or Inception
"""

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


class MobileNetV2Backbone(Model):
    """Define the MobileNetV2 Backbone."""

    def __init__(self):
        """Init the Backbone instance."""
        super().__init__()

        self.backbone = MobileNetV2(include_top=False, weights='imagenet')

    def call(self, inputs):
        """
        Forward pass of the Model.

        Args:
            inputs (tf.Tensor): Input, shape [batch, heigh, width, channels]

        Returns:
            List of Tensors
        """
        outputs = self.backbone(inputs)

        return outputs
