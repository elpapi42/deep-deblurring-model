#!/usr/bin/python
# coding=utf-8

"""
Conv Backbone of generator.

We will use MobileNetV2
Any other Sota Arch can be used
like Resnet or Inception
"""

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2


class MobileNetV2Backbone(Model):
    """Define the MobileNetV2 Backbone."""

    def __init__(
        self,
        output_index=(27, 45, 72, 107, 151),
        output_channels=128,
    ):
        """
        Init the Backbone instance.

        Args:
            output_index (list): Index of layers that will be backbone output
            output_channels (int): Number of out channels for each lateral
        """
        super().__init__()

        # MobileNet backbone
        self.backbone = self.get_backbone(output_index)

        # List of 1x1 Convs for each output
        self.output_convs = []
        for _ in output_index:
            self.output_convs.append(
                layers.Conv2D(
                    filters=output_channels,
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
        # Collect the raw backbone outputs
        intakes = self.backbone(inputs)

        # Will store the transformed backbone outputs
        outputs = []

        # Loop over every intake tensor and transform it
        for intake, conv in zip(intakes, self.output_convs):
            output = conv(intake)
            outputs.append(output)

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
