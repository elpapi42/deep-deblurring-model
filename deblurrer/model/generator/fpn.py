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
    """Define the MobileNetV2 Backbone."""

    def __init__(self):
        """Init the Backbone instance."""
        super().__init__()

    def call(self, inputs):
        """
        Forward pass of the FPN Model.

        Args:
            inputs (tf.Tensor): List of tensors, each w/ shape [bch, h, w, ch]

        Returns:
            Tensor of shape = inputs.shape
        """
        return inputs
