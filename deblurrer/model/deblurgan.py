#!/usr/bin/python
# coding=utf-8

"""
Deblur GAN architecture.

We will use MobileNetV2 as Backbone
Any other Sota Arch can be used
like Resnet or Inception
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.python.distribute import parameter_server_strategy

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

        # Update generator params
        self._minimize(
            self.distribute_strategy,
            gen_tape,
            self.optimizer[0],
            gen_loss,
            self.generator.trainable_variables,
        )

        # Update discriminator params
        self._minimize(
            self.distribute_strategy,
            disc_tape,
            self.optimizer[1],
            disc_loss,
            self.discriminator.trainable_variables,
        )

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}

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

    def _minimize(self, strategy, tape, optimizer, loss, trainable_variables):
        """
        This code was taken from Tensorflow source code.
        this is not exposed trought the public API

        Minimizes loss for one step by updating `trainable_variables`.
        This is roughly equivalent to
        ```python
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        ```
        However, this function also applies gradient clipping and loss scaling if the
        optimizer is a LossScaleOptimizer.
        Args:
            strategy: `tf.distribute.Strategy`.
            tape: A gradient tape. The loss must have been computed under this tape.
            optimizer: The optimizer used to minimize the loss.
            loss: The loss tensor.
            trainable_variables: The variables that will be updated in order to minimize
            the loss.
        """

        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, trainable_variables)

        # Whether to aggregate gradients outside of optimizer. This requires support
        # of the optimizer and doesn't work with ParameterServerStrategy and
        # CentralStroageStrategy.
        aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and not isinstance(
                strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended,
            ),
        )

        if aggregate_grads_outside_optimizer:
            # We aggregate gradients before unscaling them, in case a subclass of
            # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
            # done on scaled gradients, not unscaled gradients, for numeric stability.
            gradients = optimizer._aggregate_gradients(zip(gradients, trainable_variables))

        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        gradients = optimizer._clip_gradients(gradients)

        if trainable_variables:
            if aggregate_grads_outside_optimizer:
                optimizer.apply_gradients(
                    zip(gradients, trainable_variables),
                    experimental_aggregate_gradients=False,
                )
            else:
                optimizer.apply_gradients(zip(gradients, trainable_variables))
