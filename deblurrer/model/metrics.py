#!/usr/bin/python
# coding=utf-8

"""Defines the custom loss functions used at train time."""

import tensorflow as tf


def discriminator_accuracy(preds, real_preds):
    """
    Compute the accuracy of the discriminator output

    Args:
        preds (tf.Tensor): discriminator over real or generated batch of images
        real_preds (bool): if supplied preds comes from real images

    Returns:
        Acc of predictions compared to expectation
    """
    labels = tf.ones_like(preds['local']) if real_preds else tf.zeros_like(preds['local'])

    local_acc = tf.keras.metrics.binary_accuracy(labels, preds['local'])
    global_acc = tf.keras.metrics.binary_accuracy(labels, preds['global'])

    return (tf.reduce_mean(local_acc) + tf.reduce_mean(global_acc)) / 2.0