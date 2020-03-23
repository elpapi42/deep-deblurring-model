#!/usr/bin/python
# coding=utf-8

"""
Trainer class that defines the gradient descent of the GAN.

This module will eclusively contain training logic.
"""

from sys import stdout
import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.mixed_precision.experimental import global_policy

from deblurrer.scripts.training import Tester
from deblurrer.model.losses import discriminator_loss, generator_loss


class Trainer(Tester):
    """Define training of the GAN."""

    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        strategy,
    ):
        """
        Init the Trainer required Objects.

        Args:
            generator (tf.keras.Model): FPN Generator
            discriminator (tf.keras.Model): DS Discriminator
            gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
            disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
            strategy (tf.distribute.Strategy): Distribution strategy
        """
        super().__init__(generator, discriminator, strategy)

        # Retrieves if current global policy is mixed presicion
        gp_name = global_policy().name
        self.mixed_presicion = gp_name in {'mixed_float16', 'mixed_bfloat16'}

        # Enable fp16 mixed safe underflow mechanics
        if (self.mixed_presicion):
            # Wrap optimizers with loss scaling optimizer
            self.gen_optimizer = mixed_precision.LossScaleOptimizer(
                gen_optimizer,
                loss_scale='dynamic',
            )

            self.disc_optimizer = mixed_precision.LossScaleOptimizer(
                disc_optimizer,
                loss_scale='dynamic',
            )
        else:
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
                gen_loss, disc_loss = self.distributed_step(
                    image_batch,
                    training=True,
                )

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
    def step_fn(self, images, training=False):
        """
        Run a single train step that update params.

        This is an overriding of Tester implementation
        that updates the weights

        Args:
            images (dict): Batch of sharp/blur image pairs
            training (bool): If th forward pass is part of a training step

        Returns:
            Loss and metrics for this step
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward propagates the supplied batch of images and get losses.
            gen_loss, disc_loss = super().step_fn(images, training)

            # Calculate gradients and downscale them
            self.update_weights(gen_loss, disc_loss, gen_tape, disc_tape)

        return gen_loss, disc_loss

    def update_weights(self, gen_loss, disc_loss, gen_tape, disc_tape):
        """
        Compute gradients and apply them to the models.

        Args:
            gen_loss (float): generator loss
            disc_loss (float): discriminator loss
            gen_tape (tf.GradientTape): tf computations data for gen
            disc_tape (tf.GradientTape): tf computations data for disc
        """
        # Scale losses if mixed presicion enabled, avoid float16 underflow
        if (self.mixed_presicion):
            gen_loss = self.gen_optimizer.get_scaled_loss(loss=gen_loss)
            disc_loss = self.disc_optimizer.get_scaled_loss(loss=disc_loss)

        # Compute Grads for both models
        gen_grads = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )

        disc_grads = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables,
        )

        # Scale down the grads if mixed presicion is enabled
        if (self.mixed_presicion):
            gen_grads = self.gen_optimizer.get_unscaled_gradients(gen_grads)
            disc_grads = self.disc_optimizer.get_unscaled_gradients(disc_grads)

        # Apply gradient updates to both models
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables),
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables),
        )
