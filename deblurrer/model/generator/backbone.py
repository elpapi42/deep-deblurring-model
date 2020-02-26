#!/usr/bin/python
# coding=utf-8

"""
Conv Backbone of generator.

We will use MobileNetV2
Any other Sota Arch can be used
like Resnet or Inception
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


class MobileNetV2Backbone(Model):
    """Define the MobileNetV2 Backbone."""

    def __init__(
        self,
        output_index=(4, 39, 84, 119, 153),
        output_filters=(16, 96, 192, 288, 640),
    ):
        """
        Init the Backbone instance.

        Args:
            output_index (list): Index of layers that will be backbone output
            output_filters (list): Number of 1x1 filters for each output conv
        """
        super().__init__()

        # MobileNet backbone
        self.backbone = self.get_backbone(output_index)

        # List of 1x1 Convs for each output
        self.output_convs = []
        for filters in output_filters:
            self.output_convs.append(
                layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                ),
            )

    def call(self, inputs):
        """
        Forward pass of the Model.

        Args:
            inputs (tf.Tensor): Input, shape [batch, heigh, width, channels]

        Returns:
            List of Tensors
        """
        outputs = self.backbone(inputs)

        for index - 1, (out, conv) in enumerate(zip(outputs, self.output_convs)):
            x = conv(out)

            if (index == 0):
                x_past = self.output_convs[index](outputs[index])
                x = tf.concat([x, x_past])

        outputs = [tf.concat([conv(out), ]) for out, conv in ]
        print(outputs)
        return outputs

    def get_backbone(self, output_index):
        """
        Build a submodel that return the backbone required layers.

        Args:
            output_index (list): Index of layers that will be backbone output

        Returns:
            MobileNet Model with multiple outputs
        """
        # Get MobileNet
        backbone = MobileNetV2(include_top=False, weights='imagenet')

        # Find the outputs from backbone middle layers
        outputs = []
        for backbone_index in output_index:
            outputs.append(backbone.layers[backbone_index].output)

        # Build Model
        return Model(
            inputs=backbone.inputs,
            outputs=outputs,
        )
