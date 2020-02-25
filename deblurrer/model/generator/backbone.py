#!/usr/bin/python
# coding=utf-8

"""
Conv Backbone of generator.

We will use MobileNetV2
Any other Sota Arch can be used
like Resnet or Inception
"""

from tensorflow.keras import layers, applications, Model


class MobileNetV2Backbone(Model):
    """Define the MobileNetV2 Backbone."""

    def __init__(self):
        """Init the Backbone instance."""
        super(MobileNetV2Backbone, self).__init__(self)