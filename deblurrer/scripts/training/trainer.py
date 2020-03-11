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

from deblurrer.scripts.training import Tester
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator
from deblurrer.model.losses import ragan_ls_loss, generator_loss


class Trainer(Tester):
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
        super().__init__(generator, discriminator)

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def train(self, dataset, epochs, valid_dataset=None, verbose=False):
        """
        Trining cycle.

        Args:
            dataset (tf.data.Dataset): dataset with sharp/blur entries
            epochs (int): Hot many cycles run trought the full dataset
            valid_dataset (tf.data.Dataset): dataset for cross validation
            verbose (bool): If output to std out the metrics results
        """
        # Metrics
        gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
        disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')

        for epoch in range(epochs):
            # Jump to next line if verbose, for pretty formatting
            if (verbose):
                stdout.write('\n')

            metrics = [gen_train_loss, disc_train_loss]

            # Loop over full dataset batches
            for image_batch in dataset:
                # Exec train step
                gen_loss, disc_loss = self.train_step(image_batch)

                gen_train_loss(gen_loss)
                disc_train_loss(disc_loss)

                # Real time traning metrics
                if (verbose):
                    self.print_metrics(
                        metrics,
                        preffix='Epoch {epoch}:'.format(epoch=epoch),
                    )

            # If thereis validation dataset, do cross validation
            if (valid_dataset):
                gen_valid_loss, disc_valid_loss = self.test(valid_dataset)
                gen_valid_loss.name = 'gen_valid_loss'
                disc_valid_loss.name = 'disc_valid_loss'
                metrics += [gen_valid_loss, disc_valid_loss]

            # Print final metrics of the epoch
            if (verbose):
                self.print_metrics(
                    metrics,
                    preffix='Epoch {epoch}:'.format(epoch=epoch),
                )

            # Reset metrics state
            gen_train_loss.reset_states()
            disc_train_loss.reset_states()

    @tf.function
    def train_step(self, images):
        """
        Run a single trining step that update params for both models.

        Args:
            images (dict): Batch of sharp/blur image pairs

        Returns:
            Loss and metrics for this step
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            real_output, fake_output = self.gan_forward_pass(
                images,
                training=True,
            )

            # Calc and scale losses(avoid mixed presicion float16 underflow)
            gen_loss = generator_loss(fake_output)
            #scaled_gen_loss = gen_optimizer.get_scaled_loss(loss=gen_loss)

            disc_loss = ragan_ls_loss(real_output, fake_output)
            #scaled_disc_loss = disc_optimizer.get_scaled_loss(disc_loss)

            # Calculate gradients and downscale them
            gen_grads = gen_tape.gradient(
                gen_loss,
                self.generator.trainable_variables,
            )
            #gen_grads = gen_optimizer.get_unscaled_gradients(gen_grads)

            disc_grads = disc_tape.gradient(
                disc_loss,
                self.discriminator.trainable_variables,
            )
            #disc_grads = disc_optimizer.get_unscaled_gradients(disc_grads)

        # Apply gradient updates to both models
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables),
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables),
        )

        return gen_loss, disc_loss
