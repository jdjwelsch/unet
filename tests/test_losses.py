# -*- coding: utf-8 -*-
import tensorflow as tf
import unittest

from losses import dice_loss, cce_loss, get_mixed_loss_function


class TestLosses(unittest.TestCase):

    def setUp(self) -> None:
        self.tensor_shape = [1, 20, 20]

    def test_dice_loss(self):
        for n_classes in range(2, 5, 1):
            with self.subTest():
                y_true, y_pred = self._get_sample_tensors(n_classes)
                loss_val = dice_loss(y_true, y_pred, n_out_channels=n_classes)

                # check that there is scalar output and value is between 0 and 1
                self.assertTrue(tf.rank(loss_val) == 0)
                self.assertGreaterEqual(loss_val, 0)
                self.assertLessEqual(loss_val, 1)

    def test_cce_loss(self):
        for n_classes in range(2, 5, 1):
            with self.subTest():
                y_true, y_pred = self._get_sample_tensors(n_classes)
                loss_val = cce_loss(y_true, y_pred, n_out_channels=n_classes)

                # check that there is scalar output and value is between 0 and 1
                self.assertTrue(tf.rank(loss_val) == 0)
                self.assertGreaterEqual(loss_val, 0)
                self.assertLessEqual(loss_val, 1)

    def test_get_mixed_loss(self):
        mixed_loss = get_mixed_loss_function(dice_weight=0.2,
                                             n_out_channels=2)
        y_true, y_pred = self._get_sample_tensors(2)
        loss_val = mixed_loss(y_true, y_pred)

        # check that there is scalar output and value is between 0 and 1
        self.assertTrue(tf.rank(loss_val) == 0)
        self.assertGreaterEqual(loss_val, 0)
        self.assertLessEqual(loss_val, 1)

    def _get_sample_tensors(self, n_classes):
        y_true = tf.random.uniform(shape=self.tensor_shape,
                                   minval=0,
                                   maxval=n_classes,
                                   dtype=tf.int32)

        # prediction has a channel for each class
        pred_shape = list(self.tensor_shape).copy()
        pred_shape.append(n_classes)
        y_pred = tf.random.uniform(shape=pred_shape,
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32)
        return y_true, y_pred
