#!/usr/bin/python
# coding=utf-8

"""
Deblur GAN architecture.

We will use MobileNetV2 as Backbone
Any other Sota Arch can be used
like Resnet or Inception
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from deblurrer.model import FPNGenerator, DoubleScaleDiscriminator


class DeblurGAN(Model):
    """Define the FPN Generator Arch."""

    def __init__(self, channels=128):
        """
        Init the GAN instance.

        Args:
            channels (int): Number of std channels the FPN will manage
        """
        super().__init__()

        self.generator = FPNGenerator(channels)
        self.discriminator = DoubleScaleDiscriminator()

    def call(self, inputs):
        """
        Forward propagates the supplied batch of images.

        Args:
            inputs (dict): Sharp/Blur image batches of 4d tensors

        Returns:
            Output of the GAN, including generated images
        """
        # Forward pass generator with blurred images
        generated_images = self.generator(inputs['blur'])

        # Repeat sharp images for get real_output
        sharp_images = {
            'sharp': inputs['sharp'],
            'generated': inputs['sharp'],
        }

        # Stack gen images and sharp images for get fake_output
        generated_images = {
            'sharp': inputs['sharp'],
            'generated': generated_images,
        }

        # Forward pass discriminator with generated and real images
        return (
            self.discriminator(sharp_images),
            self.discriminator(generated_images),
            generated_images['generated'],
        )
