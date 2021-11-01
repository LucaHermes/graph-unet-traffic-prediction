import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose,
    MaxPooling2D, Dropout, Input, concatenate, Cropping2D, SpatialDropout2D
)

from models.base_model import ImageBaseModel


class VanillaUNet(ImageBaseModel):

    def __init__(self, units, out_units=8, depth=5, activation=None, use_bias=True, 
        output_activation=None, **kwargs):
        super(VanillaUNet, self).__init__()

        self.depth = depth
        self.downsampling_layers = []
        self.upsampling_layers = []
        layer_units = units

        for i in range(self.depth):
            print(max(layer_units, 128))
            conv1 = Conv2D(max(layer_units, 128), 3, 1, padding='SAME', activation=activation)
            conv2 = Conv2D(max(layer_units, 128), 3, 1, padding='SAME', activation=activation)
            pool  = MaxPooling2D(2, 2)
            layer_units = units * 2**(i+1)
            self.downsampling_layers.append((conv1, conv2, pool))

        self.bottleneck_conv1 = Conv2D(layer_units, 3, 1, padding='SAME', activation=activation)
        self.bottleneck_conv2 = Conv2D(layer_units, 3, 1, padding='SAME', activation=activation)
        self.bottleneck_conv3 = Conv2D(layer_units, 3, 1, padding='SAME', activation=activation)

        for i in range(self.depth):
            layer_units = units * 2**(self.depth-i-1)
            print(max(layer_units, 128))
            conv1 = Conv2D(max(layer_units, 128), 3, 1, padding='SAME', activation=activation)
            conv2 = Conv2D(max(layer_units, 128), 3, 1, padding='SAME', activation=activation)
            self.upsampling_layers.append((conv1, conv2))

        self.out_model = Conv2D(out_units, 1, strides=1, use_bias=False, activation=output_activation)

    @tf.function
    def call(self, x, mask, training=False):
        skip_out = []

        for conv1, conv2, pool in self.downsampling_layers:
            skip_out.append(x)
            x = conv1(x)
            x = conv2(x)
            x = pool(x)

        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_conv3(x)

        for d, (conv1, conv2) in enumerate(self.upsampling_layers):
            skip = skip_out[-(d+1)]
            x = tf.image.resize(x, skip.shape[1:3], method='nearest')
            x = tf.concat((x, skip), axis=-1)
            x = conv1(x)
            x = conv2(x)

        x = self.out_model(x)

        return x