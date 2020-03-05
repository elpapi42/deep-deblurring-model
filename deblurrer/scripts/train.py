#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os
import time

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
        #gen_loss = gen_optimizer.get_scaled_loss(loss=gen_loss)

        disc_loss = ragan_ls_loss(real_output, fake_output)
        #disc_loss = disc_optimizer.get_scaled_loss(disc_loss)

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
    for epoch in range(epochs):

        for image_batch in dataset:
            print(image_batch['blur'].shape)
            train_step(
                image_batch,
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
            )


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
    gen_optimizer = tf.keras.optimizers.Adam(0.001)
    #gen_optimizer = mixed_precision.LossScaleOptimizer(
    #    gen_optimizer,
    #    loss_scale='dynamic',
    #)

    disc_optimizer = tf.keras.optimizers.Adam(0.001)
    #disc_optimizer = mixed_precision.LossScaleOptimizer(
    #    disc_optimizer,
    #    loss_scale='dynamic',
    #)

    # Run training
    train(
        train_dataset,
        1,
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
