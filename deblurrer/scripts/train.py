#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os

import tensorflow as tf
import numpy as np

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.model.generator import FPNGenerator
from deblurrer.model.discriminator import DoubleScaleDiscriminator
from deblurrer.model.losses import ragan_ls_loss, generator_loss


@tf.function
def train_step(
    images,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):
    """
    Run a single trining step that update params for both models.

    Args:
        images (tf.Tensor): Batch of sharp/blur image pairs
        generator (tf.keras.Model): FPN Generator
        discriminator (tf.keras.Model): DS Discriminator
        generator_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
        discriminator_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
    """
    with tf.GradientTape() as (gen_tape, disc_tape):
        generated_images = generator(images['blur'], training=True)

        real_output = discriminator(images['sharp'], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = ragan_ls_loss(real_output, fake_output)

        # Calculate gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables,
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            discriminator.trainable_variables,
        )

    # Apply gradient updates to both models
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables),
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables),
    )


def run(path):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
    """
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

    # Instantiates the model for training
    with strategy.scope():
        model = FPNGenerator()#DoubleScaleDiscriminator()

    #print(model.fpn.backbone.backbone.summary())
    # Instantiate model and run training
    # Mock training
    for example in train_dataset.take(1):
        print(np.shape(example['blur']))
        print(model(example['blur']))

    

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
