# -*- coding: utf-8 -*-
import tensorflow as tf


class UNetFactory:
    """
    Implementation of U-Net.

    Usage: Initialise the UNet Factory with the properties of your choice and
    retrieve the tensorflow model by calling get_model().
    U-Net: Convolutional Networks for Biomedical Image Segmentation,
    Ronneberger et al.
    https://arxiv.org/abs/1505.04597
    """
    def __init__(self, input_shape, n_blocks=4, n_classes=1, batch_norm=True,
                 drop_out=False, logits=False):
        """
        :param input_shape: tuple, shape of input images. Has to be of form
        [b, 2**n, 2**n, c] where n >= n_blocks. Otherwise skip connection
        dimensions dont match.
        :param n_blocks: int, number of convolutional blocks to use
        :param n_classes: int, number of output classes, i.e. number of output
        channels for segmentation mask
        :param batch_norm: bool, whether to use batch_norm after convolutions
        :param drop_out: bool, whether to use drop_out before bottleneck
        """
        self.input_shape = input_shape
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.batch_norm = batch_norm
        self.drop_out = drop_out
        self.n_filters = 64
        self.logits = logits

    def get_model(self):
        """
        Construct the keras model for U-Net with keras functional API

        :return: tf.keras.model instance
        """
        inputs = tf.keras.Input(shape=self.input_shape, name='img')

        # down convolution
        skip_input = []
        y = inputs
        for level in range(self.n_blocks):
            y, skip = self._down_block(y, n_filters=(2**level * self.n_filters))
            # remember convolution output before pooling for skip connections
            skip_input.append(skip)

        if self.drop_out:
            y = tf.keras.layers.Dropout(0.3)(y)

        # bottleneck
        y = self._conv_block(y, n_filters=(2**self.n_blocks * self.n_filters))

        # up 'convolution'
        for level in range(self.n_blocks-1, -1, -1):
            y = self._up_block(y,
                               skip_input[level],
                               n_filters=(2**level * self.n_filters))

        if self.logits:
            activation = 'linear'
        else:
            activation = 'softmax'
        out = tf.keras.layers.Conv2D(filters=self.n_classes,
                                     kernel_size=1,
                                     activation=activation)(y)
        # construct keras model
        model = tf.keras.Model(inputs=inputs, outputs=out)

        return model

    def _down_block(self, inputs, n_filters):
        """
        Block for encoding ("contracting") part of U-Net.

        :param inputs: tf.tensor, input tensor
        :param n_filters: int, number of filters to be used by convolution block
        :return: tuple, output tensors before and after MaxPooling2D operation.
        """
        conv = self._conv_block(inputs, n_filters)
        pool = tf.keras.layers.MaxPool2D(pool_size=2)(conv)
        return pool, conv

    def _up_block(self, inputs, skip_input, n_filters):
        """
        Block for decoding ("expanding") part of U-Net.

        :param inputs: tf.tensor, input tensor
        :param skip_input: tf.tensor, skip_connection input from encoding part
        :param n_filters: int, number of filters to be used by convolution block
        :return: tf.tensor, output tensor
        """
        up = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                             kernel_size=2,
                                             strides=2,
                                             padding='same',
                                             activation='relu')(inputs)
        merge = tf.keras.layers.concatenate([up, skip_input], axis=-1)
        conv = self._conv_block(merge, n_filters)

        return conv

    def _conv_block(self, inputs, n_filters):
        """
        Convolution block for U-Net.

        Consists of two convolutional layers with kernel size 3 and n_filters
        output channels. If self.batch_norm is True, BatchNormalization is
        applied after each convolution operation.
        :param inputs: tf.tensor, input tensor
        :param n_filters: int, number of filters to be used by each convolution
        :return: tf.tensor, output tensor
        """
        conv1 = tf.keras.layers.Conv2D(n_filters,
                                       kernel_size=3,
                                       activation='relu',
                                       padding='same')(inputs)
        if self.batch_norm:
            conv1 = tf.keras.layers.BatchNormalization()(conv1)

        conv2 = tf.keras.layers.Conv2D(n_filters,
                                       kernel_size=3,
                                       activation='relu',
                                       padding='same')(conv1)
        if self.batch_norm:
            conv2 = tf.keras.layers.BatchNormalization()(conv2)

        return conv2
