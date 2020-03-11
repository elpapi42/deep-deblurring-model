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

    def test(self, dataset, print_metrics=False):
        """
        Test the geneator and discriminator against the supplied dataset.

        Args:
            dataset (tf.data.Dataset): dataset to test the model
            print_metrics (bool): If output to std out the metrics results

        Returns:
            loss and metrics
        """
        # Metrics
        gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

        # Loop over full dataset batches
        for image_batch in dataset:
            # Exec test step
            gen_loss, disc_loss = self.test_step(image_batch)

            gen_loss_metric(gen_loss)
            disc_loss_metric(disc_loss)

            if (print_metrics):
                self.print_metrics([gen_loss_metric, disc_loss_metric])

        # Return mean loss across all the batches
        return gen_loss_metric.result(), disc_loss_metric.result()

    @tf.function
    def test_step(self, images):
        """
        Forward pass images trought model and calculate loss and metrics.

        Args:
            images (dict): Of Tensors with shape [batch, height, width, chnls]

        Returns:
            Loss and metrics for this step
        """
        # Forward pass generator with blurred images
        generated_images = self.generator(images['blur'], training=False)

        # Repeat sharp images for get real_output
        sharp_images = {
            'sharp': images['sharp'],
            'generated': images['sharp'],
        }

        # Stack gen images and sharp images for get fake_output
        generated_images = {
            'sharp': images['sharp'],
            'generated': generated_images,
        }

        # Forward pass discriminator with generated and real images
        real_output = self.discriminator(sharp_images, training=False)
        fake_output = self.discriminator(generated_images, training=False)

        # Calculate and return losses
        return (
            generator_loss(fake_output),
            ragan_ls_loss(real_output, fake_output),
        )

    def print_metrics(self, metrics, preffix=''):
        """
        Print to std output the supplied Metrics

        Args:
            metrics (tuple/list): list of tf.keras.metrics to be printed
            preffix (str): String to be concat at the start of the print
        """
        # Stores the full metrics strings to be printed
        output = preffix

        # Iter over supplied metrics
        for metric in metrics:
            # Get and format metric data
            metric_string = '[{name}: {value:.6f}]'.format(
                name=metric.name,
                value=metric.result(),
            )

            # Concat the result to the output string
            output = '{output} {metric_string}'.format(
                output=output,
                metric_string=metric_string,
            )

        stdout.write('\r{out}'.format(out=output))
        stdout.flush()
