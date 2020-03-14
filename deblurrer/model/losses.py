#!/usr/bin/python
# coding=utf-8

"""Defines the custom loss functions used at train time."""

import tensorflow as tf


def ragan_ls_loss(pred, real_preds):
    """
    Compute the RaGAN-LS loss of supplied prediction.

    It is a two-term binary crossentropy loss
    Author call it RaGAN-LS loss

    Args:
        pred (tf.Tensor): discriminator over real or generated batch of images
        real_preds (bool): if supplied preds comes from real images

    Returns:
        Loss of predictions compared to expectation
    """
    real_preds = tf.constant(real_preds, dtype=tf.float32)

    first_side = tf.multiply(real_preds, tf.square(pred - 1))
    second_side = tf.multiply(1 - real_preds, tf.square(pred + 1))

    return tf.reduce_mean(first_side + second_side)


def discriminator_loss(real_pred, fake_pred):
    """
    Compute the **TOTAL** RaGAN-LS loss of DScaleDiscrim.

    Args:
        real_pred (tf.Tensor): discriminator over real images
        fake_pred (tf.Tensor): discrim over generated images

    Returns:
        Total loss over real and fake images
    """
    real_loss = ragan_ls_loss(real_pred, real_preds=True)
    fake_loss = ragan_ls_loss(fake_pred, real_preds=False)

    return real_loss + fake_loss


def generator_loss(fake_pred):
    """
    Define the cutom loss function for generator.

    It is a three-term loss:

    Lg = 0.5 * Lp + 0.006 * Lx + 0.01 * Ladv

    Lp = MSE Loss
    Lx = Perceptual Loss
    Ladv = Discriminator Loss

    Args:
        fake_pred (tf.Tensor): Scalar, output of the discriminator

    Returns:
        Generator three-term loss function output
    """
    bin_cross = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return bin_cross(tf.ones_like(fake_pred), fake_pred)
