#!/usr/bin/python
# coding=utf-8

"""
Start the training of the Model Architecture.

This module will eclusively contains training logic.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from deblurrer.scripts.datasets.generate_dataset import get_dataset
from deblurrer.model.callbacks import SaveImageToDisk
from deblurrer.model import DeblurGAN


def run(
    path,
    model=None,
    gen_optimizer=None,
    disc_optimizer=None,
    strategy=None,
    output_folder='',
    logs_folder='',
):
    """
    Run the training script.

    Args:
        path (str): path from where to load tfrecords
        model (tf.keras.Model): DeblurGAN model
        gen_optimizer (tf.keras.optimizers.Optimizer): Gen Optimizer
        disc_optimizer (tf.keras.optimizers.Optimizer): Disc optimizer
        strategy (tf.distributed.Strategy): Distribution strategy
        output_folder (str): Where to store images for performance test
        logs_folder (str): Where to store training logs
        warm_epochs (int): Number of epoch for freeze the FPN Backend

    Returns:
        model, generator optimizer, discriminator optimizer and strategy
        in that order
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

    # Setup float16 mixed precision
    if (int(os.environ.get('USE_MIXED_PRECISION'))):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    # If in colba tpu instance, use tpus, gpus otherwise
    colab_tpu = os.environ.get('COLAB_TPU_ADDR') is not None
    if (colab_tpu and (strategy is None)):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'),
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        # Convert dataset to distribute datasets
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    elif (strategy is None):
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

    with strategy.scope():
        # Instantiate models and optimizers
        if (model is None):
            model = DeblurGAN(
                channels=int(os.environ.get('FPN_CHANNELS')),
                filters=int(os.environ.get('DISC_FILTERS')),
                conv_count=int(os.environ.get('DISC_CONV_COUNT')),
            )
        if (gen_optimizer is None):
            gen_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('GEN_LR')))
        if (disc_optimizer is None):
            disc_optimizer = tf.keras.optimizers.Adam(float(os.environ.get('DISC_LR')))

        # This is for init modelweights and pickup a test sample
        for batch in train_dataset.skip(10).take(1):
            model(batch)
            # This will be used for visual performance test gen
            test_image = batch[0]

        model.summary()

        # Train GAN freezing the generator backbone
        if (int(os.environ['WARM_EPOCHS']) > 0):
            print('Starting model warm up...')

            model.generator.fpn.backbone.backbone.trainable = False

            model.compile(
                optimizer=[
                    gen_optimizer,
                    disc_optimizer,
                ],
            )

            model.fit(
                train_dataset,
                epochs=warm_epochs,
                validation_data=valid_dataset,
                callbacks=[
                    SaveImageToDisk(output_folder, test_image),
                ],
            )

        print('Starting model training...')

        model.generator.fpn.backbone.backbone.trainable = True

        model.compile(
            optimizer=[
                gen_optimizer,
                disc_optimizer,
            ],
        )

        model.fit(
            train_dataset,
            epochs=int(os.environ['EPOCHS']),
            validation_data=valid_dataset,
            callbacks=[
                SaveImageToDisk(output_folder, test_image),
                tf.keras.callbacks.TensorBoard(
                    log_dir = os.path.join(logs_folder, datetime.now().strftime("%Y%m%d-%H%M%S")),
                    histogram_freq = 1,
                    profile_batch = os.environ['PROFILE_BATCH'],
                ),
            ],
        )

    return model, gen_optimizer, disc_optimizer, strategy


if (__name__ == '__main__'):
    # Get the path to the tfrcords folder
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__),
            ),
        ),
    )

    # Load .env vars
    dotenv_path = os.path.join(path, '.env')
    load_dotenv(dotenv_path)

    tfrec_path = os.path.join(
        path,
        os.path.join('datasets', 'tfrecords'),
    )

    output_path = os.path.join(path, 'output')

    logs_path = os.path.join(path, 'logs')

    run(
        tfrec_path,
        output_folder=output_path,
        logs_folder=logs_path,
    )
