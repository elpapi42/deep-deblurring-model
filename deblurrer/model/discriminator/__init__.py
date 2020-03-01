#!/usr/bin/python
# coding=utf-8

"""
Discriminator side of the DeblurGAN.

Features two parralel Bin Classifiers
One is a PathGAN Discriminator
Other a Global Image Discriminator
"""

from deblurrer.model.discriminator.local import LocalDiscriminator, LeakyConvBlock
from deblurrer.model.discriminator.dglobal import GlobalDiscriminator
