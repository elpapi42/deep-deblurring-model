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
from deblurrer.model.losses import discriminator_loss, generator_loss


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

        self.loss_network = self.get_loss_network()

    def call(self, inputs):
        """
        Forward propagates the supplied batch of images.

        Args:
            inputs (tf.Tensor): shape [batch, 2, h, w, chnls]

        Returns:
            Output of the GAN, including generated images
        """
        # Unstack the two images batches
        sharp, blur = tf.unstack(inputs, axis=1)

        # Forward pass generator with blurred images
        gen_images = self.generator(blur)

        # Forward pass discriminator with generated and real images
        return (
            self.discriminator([sharp, sharp]),
            self.discriminator([sharp, gen_images]),
            gen_images,
        )

    def train_step(self, datas):
        """
        The logic for one training step.

        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.

        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_output, fake_output, gen_images = self(datas)

            sharp, _ = tf.unstack(datas, axis=1)

            # Calculate losses
            gen_loss = generator_loss(
                gen_images,
                sharp,
                fake_output,
                self.loss_network,
            )

            disc_loss = discriminator_loss(real_output, fake_output)

        return {'loss': 0.0}

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
