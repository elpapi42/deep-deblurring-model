#!/usr/bin/python
# coding=utf-8

"""
Tester class that implements the evaluation and testing behavior.

This module will eclusively contain test/evaluation logic.
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


class Tester(object):
    """Define testing and evaluation of the GAN."""

    def __init__(self, generator, discriminator):
        """
        Init the models required.

        Args:
            generator (tf.keras.Model): FPN Generator
            discriminator (tf.keras.Model): DS Discriminator
        """
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def test(self, dataset):
        """
        Test the geneator and discriminator against the supplied dataset.

        Args:
            dataset (tf.data.Dataset): dataset to test the model

        Returns:
            loss and metrics
        """
        return 0.0
