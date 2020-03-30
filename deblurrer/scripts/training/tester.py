#!/usr/bin/python
# coding=utf-8

"""
Tester class that implements the evaluation and testing behavior.

This module will eclusively contain test/evaluation logic.
"""

from sys import stdout

import tensorflow as tf

from deblurrer.model.losses import discriminator_loss, generator_loss


class Tester(object):
    """Define testing and evaluation of the GAN."""

    def __init__(self, generator, discriminator, strategy):
        """
        Init the models required.

        Args:
            generator (tf.keras.Model): FPN Generator
            discriminator (tf.keras.Model): DS Discriminator
            strategy (tf.distribute.Strategy): Distribution strategy
        """
        super().__init__()

        self.strategy = strategy

        self.generator = generator
        self.discriminator = discriminator

        self.loss_network = self.get_loss_network()

    def test(self, dataset, verbose=False):
        """
        Test the geneator and discriminator against the supplied dataset.

        Args:
            dataset (tf.data.Dataset): dataset to test the model
            verbose (bool): If output to std out the metrics results

        Returns:
            loss and metrics
        """
        # Jump to next line if verbose, for pretty formatting
        if (verbose):
            stdout.write('\n')

        # Metrics
        gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

        # Loop over full dataset batches
        for image_batch in dataset:
            # Exec test step
            gen_loss, disc_loss = self.distributed_step(image_batch)

            gen_loss_metric(gen_loss)
            disc_loss_metric(disc_loss)

            if (verbose):
                self.print_metrics(
                    [gen_loss_metric, disc_loss_metric],
                    preffix='Test:',
                )

        # Return mean loss across all the batches
        return gen_loss_metric, disc_loss_metric

    @tf.function
    def distributed_step(self, images, training=False):
        """
        Forward pass images trought model and calculate loss and metrics.

        This is done in a distributed way
        using the supplied distribute strategy

        Args:
            images (dict): Of Tensors with shape [batch, height, width, chnls]
            training (bool): If th forward pass is part of a training step

        Returns:
            Loss and metrics for this step
        """
        # Execute a distributed step on each replica
        per_replica_losses = self.strategy.run(
            self.step_fn,
            args=(images, training),
        )

        # Agregates all the replicas results
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_losses,
            axis=None,
        )

    @tf.function
    def step_fn(self, images, training=False):
        """
        Run a single step that calculates and return metrics.

        Args:
            images (dict): Batch of sharp/blur image pairs
            training (bool): If th forward pass is part of a training step

        Returns:
            Loss and metrics for this step
        """
        return self.get_loss_over_batch(images, training)

    def print_metrics(self, metrics, preffix=''):
        """
        Print to std output the supplied Metrics.

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

    def gan_forward_pass(self, images, training=False):
        """
        Forward propagates the supplied batch of images.

        Args:
            images (dict): Sharp/Blur image batches of 4d tensors
            training (bool): If th forward pass is part of a training step

        Returns:
            Output of the GAN, including generated images
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
        return (
            self.discriminator(sharp_images, training=False),
            self.discriminator(generated_images, training=False),
            generated_images['generated'],
        )

    def get_loss_over_batch(self, images, training=False):
        """
        Compute losses of the GAN over a batch of images.

        Args:
            images (dict): Sharp/Blur image batches of 4d tensors
            training (bool): If the forward pass is part of a training step
        
        Returns:
            Losses of the generator and discriminator networks
        """
        # Forward propagates the supplied batch of images.
        real_output, fake_output, gen_images = self.gan_forward_pass(
            images,
            training=training,
        )

        # Calculate and return losses
        return (
            generator_loss(
                gen_images,
                images['sharp'],
                fake_output,
                self.loss_network,
            ),
            discriminator_loss(real_output, fake_output),
        )

    def get_loss_network(self):
        """
        Build model based on VGG19.

        The model will output conv3_3 layer output
        the remaining architecture will be discarded

        Returns:
            Loss network based on VGG19
        """
        vgg19 = tf.keras.applications.VGG19(include_top=False)

        return tf.keras.Model(
            inputs=vgg19.inputs,
            outputs=vgg19.get_layer(name='block3_conv3').output,
        )
