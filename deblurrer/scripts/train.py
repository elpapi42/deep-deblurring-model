#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
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


@tf.function
def train_step(
    images,
    generator,
    discriminator,
    gen_optimizer,
    disc_optimizer,
):
    """
    Run a single trining step that update params for both models.

    Args:
        images (tf.Tensor): Batch of sharp/blur image pairs
        generator (tf.keras.Model): FPN Generator
        discriminator (tf.keras.Model): DS Discriminator
        gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
        disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images['blur'], training=True)

        # Repeat sharp images for get real disc output
        sharp_images = {
            'sharp': images['sharp'],
            'blur': images['sharp'],
        }

        # Stack gen images and sharp images for get fake_output
        generated_images = {
            'sharp': images['sharp'],
            'blur': generated_images,
        }

        real_output = discriminator(sharp_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate and scale losses(avoid mixed presicion float16 underflow)
        gen_loss = generator_loss(fake_output)
        #scaled_gen_loss = gen_optimizer.get_scaled_loss(loss=gen_loss)

        disc_loss = ragan_ls_loss(real_output, fake_output)
        #scaled_disc_loss = disc_optimizer.get_scaled_loss(disc_loss)

        # Calculate gradients and downscale them
        gen_grads = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables,
        )
        #gen_grads = gen_optimizer.get_unscaled_gradients(gen_grads)

        disc_grads = disc_tape.gradient(
            disc_loss,
            discriminator.trainable_variables,
        )
        #disc_grads = disc_optimizer.get_unscaled_gradients(disc_grads)

    # Apply gradient updates to both models
    gen_optimizer.apply_gradients(
        zip(gen_grads, generator.trainable_variables),
    )
    disc_optimizer.apply_gradients(
        zip(disc_grads, discriminator.trainable_variables),
    )

    return gen_loss, disc_loss


def train(
    dataset,
    epochs,
    generator,
    discriminator,
    gen_optimizer,
    disc_optimizer,
):
    """
    Trining cycle.

    Args:
        dataset (tf.data.Dataset): Tensorflow dataset with sharp/blur entries
        epochs (int): Hot many cycles run trought the full dataset
        generator (tf.keras.Model): FPN Generator
        discriminator (tf.keras.Model): DS Discriminator
        gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
        disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
    """
    # Metrics
    gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
    disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')

    for epoch in range(epochs):
        # Loop over full dataset batches
        for image_batch in dataset:
            # Exec train step
            gen_loss, disc_loss = train_step(
                image_batch,
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
            )

            gen_train_loss(gen_loss)
            disc_train_loss(disc_loss)

            # Show epoch results
            # Collect generator metrics
            gen_metrics = 'gen_train_loss: {gtl}.'.format(
                gtl=gen_train_loss.result(),
            )

            # Collect discrimiantor metrics
            disc_metrics = 'disc_train_loss: {dtl}.'.format(
                dtl=disc_train_loss.result(),
            )

            stdout.write(
                '\rEpoch {e}: {gen_metrics} {disc_metrics}'.format(
                    e=epoch,
                    gen_metrics=gen_metrics,
                    disc_metrics=disc_metrics,
                ),
            )
            stdout.flush()

        # Go to next line
        stdout.write('\n')

        # Reset metrics state
        gen_train_loss.reset_states()


def run(path):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
    """
    # Setup float16 mixed precision
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)

    # Create train dataset
    train_dataset = get_dataset(
        path,
        name='train',
        batch_size=int(os.environ.get('BATCH_SIZE')),
    )

    # Create validation dataset
    valid_dataset = get_dataset(
        path,
        name='valid',
        batch_size=int(os.environ.get('BATCH_SIZE')),
    )

    # If the machine executing the code has TPUs, use them
    if (True):
        strategy = tf.distribute.MirroredStrategy()
    else:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'),
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    #with strategy.scope():
    # Instantiates the models for training
    generator = FPNGenerator(int(os.environ.get('FPN_CHANNELS')))
    discriminator = DoubleScaleDiscriminator()

    # Instantiate optimizers with loss scaling
    gen_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('GEN_LR')))
    #gen_optimizer = mixed_precision.LossScaleOptimizer(
    #    gen_optimizer,
    #    loss_scale='dynamic',
    #)

    disc_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('DISC_LR')))
    #disc_optimizer = mixed_precision.LossScaleOptimizer(
    #    disc_optimizer,
    #    loss_scale='dynamic',
    #)

    # Run training
    train(
        train_dataset,
        2,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
    )


if (__name__ == '__main__'):
    # Get the path to the tfrcords folder
    path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__),
                ),
            ),
        ),
        os.path.join('datasets', 'tfrecords'),
    )

    run(path)
