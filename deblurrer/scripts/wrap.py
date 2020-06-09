import tensorflow as tf


class ImageByteWrapper(tf.keras.Model):

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.string)])
    def call(self, inputs):
        def pre_input(image):
            image = tf.io.decode_image(image)
            image = tf.cast(image, dtype=tf.float32)
            image = (image - 127.0) / 128.0
            return image

        images = tf.map_fn(pre_input, tf.squeeze(inputs, axis=1), dtype=tf.float32)

        return images


def convert(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.target_spec.supported_ops = [
        #tf.lite.OpsSet.TFLITE_BUILTINS,
        #tf.lite.OpsSet.SELECT_TF_OPS,
    #]
    tflite_model = converter.convert()

    return tflite_model

model = ImageByteWrapper()

test_input = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=255, dtype=tf.int32)
test_input = tf.cast(test_input, dtype=tf.uint8)
test_input = tf.io.encode_jpeg(test_input)
test_input = tf.stack([test_input, test_input])
test_input = tf.reshape(test_input, [-1, 1])

with tf.device('/cpu:0'):
    test_output = model(test_input)

tflite = convert(model)

'''
class ImageByteWrapper(Model):
    """Wrapes the supplied model into a image bytes decoder/encoder."""

    def __init__(self, model):
        """
        Init the wrapper.

        Args:
            model (int): model to wrappe with image decoding/encoding
        """
        super().__init__()

        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.string)])
    def call(self, inputs):
        """
        Decode and Forward propagates the supplied image file.

        Args:
            inputs (tf.Tensor): string tensor shape [batch, 1]

        Returns:
            b64 Encoded output of the model
        """
        def pre_input(image):
            image = tf.io.decode_image(image, channels=3)
            image = tf.cast(image, dtype=tf.float32)
            image = (image - 127.0) / 128.0
            return image

        images = tf.map_fn(pre_input, tf.squeeze(inputs, axis=1), dtype=tf.float32)

        outputs = self.model(images)

        def post_output(output):
            output = (output * 128.0) + 127.0
            output = tf.cast(output, dtype=tf.uint8)
            output = tf.io.encode_jpeg(output)
            output = tf.io.encode_base64(output, pad=True)
            return output

        outputs = tf.map_fn(post_output, outputs, dtype=tf.string)

        return outputs
'''