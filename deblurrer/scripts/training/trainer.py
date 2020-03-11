#!/usr/bin/python
# coding=utf-8

"""
Trainer class that defines the gradient descent of the GAN.

This module will eclusively contain training logic.
"""

import os
import time
from sys import stdout

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator
from deblurrer.model.losses import ragan_ls_loss, generator_loss


class Trainer(object):
    """Define training of the GAN."""

    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
    ):
        """
        Init the Trainer required Objects.

        Args:
            generator (tf.keras.Model): FPN Generator
            discriminator (tf.keras.Model): DS Discriminator
            gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
            disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
        """
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
