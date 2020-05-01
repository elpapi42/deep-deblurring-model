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

    def __init__(self, channels=128, filters=12, conv_count=4):
        """
        Init the GAN instance.

        Args:
            channels (int): Number of std channels the FPN will manage
            filters (int): Base number of filters, scaled by conv_count
            conv_count (int): Number of leaky conv layers to use
        """
        super().__init__()

        self.generator = FPNGenerator(channels)
        self.discriminator = DoubleScaleDiscriminator(filters, conv_count)

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

    def train_step(self, images):
        """
        Logic for one training step.

        Arguments:
            images: A nested structure of Tensors.

        Returns:
            Dict of Metrics of the GAN
        """
        # Unstack the two images batches
        sharp, blur = tf.unstack(images, axis=1)

        # Update generator parameters and return metrics
        generator_metrics, generated = self.generator_train_step(
            sharp,
            blur,
        )

        # Discriminator train step for generated images
        generated_metrics = self.discriminator_train_step(
            sharp,
            generated,
            real_preds=False,
        )

        # Discriminator train step for sharp images
        sharp_metrics = self.discriminator_train_step(
            sharp,
            sharp,
            real_preds=True,
        )

        return {
            'gen_loss': generator_metrics['loss'],
            'disc_sharp_loss': sharp_metrics['loss'],
            'disc_generated_loss': generated_metrics['loss'],
        }

    def test_step(self, images):
        """
        The logic for one testing step.

        Args:
            images: A nested structure of Tensors

        Returns:
            Dict of Metrics of the GAN
        """
        # Unstack the two images batches
        sharp, blur = tf.unstack(images, axis=1)

        # Calculate metrics
        generator_metrics, generated = self.generator_metrics_over_batch(sharp, blur)
        generated_metrics = self.discriminator_metrics_over_batch(sharp, generated, real_preds=False)
        sharp_metrics = self.discriminator_metrics_over_batch(sharp, sharp, real_preds=True)

        return {
            'gen_loss': generator_metrics['loss'],
            'disc_sharp_loss': sharp_metrics['loss'],
            'disc_generated_loss': generated_metrics['loss'],
        }

    def generator_metrics_over_batch(self, sharp, blur, return_generated_images=True):
        """
        Compute metrics of the generator over a batch of images.

        Args:
            sharp (tensor): shape [batch, h, w, chnls]
            blur (tensor): shape [batch, h, w, chnls]
            return_generated_images (bool): Self explanatory

        Returns:
            Dict of Metrics of the generator 
            or list len=2 with [metrics, generated_images]
        """
        # Generate a batch of sinthetized images
        generated = self.generator(blur)

        # Forward pass generated images over discriminator
        preds = self.discriminator([sharp, generated])

        # Calculate loss
        loss = generator_loss(
            generated,
            sharp,
            preds,
            self.loss_network,
        )

        # Register metrics on dictionary
        metrics = {
            'loss': loss,
        }

        if (return_generated_images):
            return [metrics, generated]
        else:
            return metrics
        
    def generator_train_step(self, sharp, blur, return_generated_images=True):
        """
        Update generator params over a batch of images.

        Args:
            sharp (tensor): shape [batch, h, w, chnls]
            blur (tensor): shape [batch, h, w, chnls]
            return_generated_images (bool): Self explanatory

        Returns:
            Dict of Metrics of the generator 
            or list len=2 with [metrics, generated_images]
        """
        # Record operations including metric calculations
        with tf.GradientTape() as tape:
            output = self.generator_metrics_over_batch(
                sharp,
                blur,
                return_generated_images,
            )
        
        # Update **Generator** parameters
        self._minimize(
            self.distribute_strategy,
            tape,
            self.optimizer[0],
            output[0]['loss'] if return_generated_images else output['loss'],
            self.generator.trainable_variables,
        )

        return output
        
    def discriminator_metrics_over_batch(self, sharp, image, real_preds):
        """
        Compute metrics of the discriminator over a batch of images.

        Args:
            sharp (tensor): required by the DSCaleDisc. [batch, h, w, chnls]
            image (tensor): sharp or generated image. [batch, h, w, chnls]
            real_preds (bool): if supplied preds comes from real images

        Returns:
            Dict of Metrics of the discriminator 
        """
        # Forwardpass the discriminator
        preds = self.discriminator([sharp, image])

        # Calculate metrics
        loss = discriminator_loss(preds, real_preds)

        return {
            'loss': loss,
        }

    def discriminator_train_step(self, sharp, image, real_preds):
        """
        Update discriminator params over a batch of images.

        Args:
            sharp (tensor): required by the DSCaleDisc. [batch, h, w, chnls]
            image (tensor): sharp or generated image. [batch, h, w, chnls]
            real_preds (bool): if supplied preds comes from real images

        Returns:
            Dict of Metrics of the discriminator
        """
        with tf.GradientTape() as tape:
            metrics = self.discriminator_metrics_over_batch(
                sharp,
                image,
                real_preds,
            )

        # Update discriminator params for generated images
        self._minimize(
            self.distribute_strategy,
            tape,
            self.optimizer[1],
            metrics['loss'],
            self.discriminator.trainable_variables,
        )

        return metrics

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
            name='loss_network',
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
