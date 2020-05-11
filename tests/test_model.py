# -*- coding: utf-8 -*-
import tensorflow as tf
import unittest

from model import UNetFactory


class TestUNetFactory(unittest.TestCase):

    def setUp(self) -> None:
        self.tensor_shape = [1, 32, 32, 3]
        self.input_shape = self.tensor_shape[1:]
        self.tensor_sample = tf.random.uniform(shape=self.tensor_shape)

    def test_conv_block(self):
        factory = UNetFactory(input_shape=self.input_shape)

        # check output shapes for different number of filters
        for i in range(1, 4, 1):
            with self.subTest():
                conv = factory._conv_block(self.tensor_sample, n_filters=i)

                expected_shape = self.tensor_shape
                expected_shape[-1] = i

                self.assertTupleEqual(tuple(expected_shape), tuple(conv.shape))

    def test_down_block(self):
        factory = UNetFactory(input_shape=self.input_shape)

        # check output shapes for different number of filters
        for i in range(1, 8, 2):
            with self.subTest():
                down, skip = factory._down_block(self.tensor_sample,
                                                 n_filters=i)

                exp_skip_shape = self.tensor_shape
                exp_skip_shape[-1] = i

                # due to pooling image dimensions should be halved for down
                exp_down_shape = exp_skip_shape.copy()
                exp_down_shape[1] = int(exp_down_shape[1] / 2)
                exp_down_shape[2] = int(exp_down_shape[2] / 2)

                self.assertTupleEqual(tuple(exp_down_shape), tuple(down.shape))
                self.assertTupleEqual(tuple(exp_skip_shape), tuple(skip.shape))

    def test_up_block(self):
        factory = UNetFactory(input_shape=self.input_shape)

        skip_shape = self.tensor_shape.copy()
        skip_shape[1] = int(skip_shape[1] * 2)
        skip_shape[2] = int(skip_shape[2] * 2)
        skip = tf.random.uniform(shape=skip_shape)

        # check output shapes for different number of filters
        for i in range(1, 8, 2):
            with self.subTest():
                up = factory._up_block(self.tensor_sample,
                                       skip,
                                       n_filters=i)

                exp_up_shape = skip_shape.copy()
                exp_up_shape[-1] = i

                self.assertTupleEqual(tuple(exp_up_shape), tuple(up.shape))

    def test_get_model(self):
        # check output shapes for different number of classes
        for i in range(1, 4, 1):
            with self.subTest():
                factory = UNetFactory(input_shape=self.input_shape, n_classes=i)
                model = factory.get_model()

                # check that a model is actually the result
                self.assertTrue(isinstance(model, tf.keras.Model))

                # check inference shape
                pred = model(self.tensor_sample)

                exp_shape = self.tensor_shape.copy()
                exp_shape[-1] = factory.n_classes

                self.assertTupleEqual(tuple(exp_shape), tuple(pred.shape))
