#!/usr/bin/python
# coding=utf-8

"""
Generator side of the DCGAN.

We will use MobileNetV2
Any other Sota Arch can be used
like Resnet or Inception
"""

from deblurrer.model.generator.backbone import MobileNetV2Backbone
from deblurrer.model.generator.fpn import FPN, FPNConvBlock, ConvBlock
from deblurrer.model.generator.generator import FPNGenerator
