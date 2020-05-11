# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import numpy as np
import unittest
import pandas as pd
from PIL import Image

from data import Dataset, DataAugmenter


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        # set up mock data directories
        self.image_dir = 'tmp_img'
        self.label_dir = 'tmp_label'

        for directory in [self.image_dir, self.label_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # create and save mock images and labels
        self.n_datapoints = 10
        for i in range(self.n_datapoints):
            img_np = np.random.randint(low=0, high=256, size=(20, 20, 3))
            label_np = np.random.randint(low=0, high=256, size=(20, 20))

            img = Image.fromarray(img_np, mode='RGB')
            img_path = os.path.join(self.image_dir, f'datapoint_{i}.png')
            img.save(img_path, 'PNG')

            label = Image.fromarray(label_np, mode='L')
            label_path = os.path.join(self.label_dir, f'datapoint_{i}.png')
            label.save(label_path, 'PNG')

    def tearDown(self) -> None:
        # clean up mock data
        for directory in [self.image_dir, self.label_dir]:
            shutil.rmtree(directory)

    def test_calculate_data_split_train_val(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir,
                       val_ratio=0.2)

        self.assertEqual(self.n_datapoints, len(dset.split_df))

        # check that validation data makes up proportion set in init
        n_val = dset.split_df.purpose.value_counts()['val']
        self.assertEqual(dset.val_ratio, n_val / self.n_datapoints)

    def test_calculate_data_split_crossvalidation(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir,
                       cross_validation=True,
                       n_folds=5)

        self.assertEqual(self.n_datapoints, len(dset.split_df))
        # check that there as many different folds as required
        n_folds = len(np.unique(dset.split_df.fold))
        self.assertEqual(dset.n_folds, n_folds)

    def test_filter_df(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir)

        # mock dataframe
        n_dp = 10
        name_list = [f'dp_{i}.png' for i in range(n_dp)]
        # split train val 50:50
        purpose_list = ['train' if i % 2 else 'val' for i in range(n_dp)]
        # take to samples for each fold
        fold_list = [i % 5 for i in range(n_dp)]
        dset.split_df = pd.DataFrame({'img_name': name_list,
                                      'purpose': purpose_list,
                                      'fold': fold_list})

        # check that filtered dataframes have the expected number of entries
        val_filtered = dset._filter_df(purpose='val')
        self.assertEqual(n_dp / 2, len(val_filtered))

        fold_filtered = dset._filter_df(cross_validation=True, fold=1)
        self.assertEqual(n_dp / 5, len(fold_filtered))

    def test_load_images(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir,
                       val_ratio=0.5)

        img_stack = dset._load_images(purpose='train')
        self.assertTupleEqual((self.n_datapoints / 2, 20, 20, 3),
                              img_stack.shape)

    def test_load_labels(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir,
                       val_ratio=0.5)

        label_stack = dset._load_labels(purpose='train')
        self.assertTupleEqual((self.n_datapoints / 2, 20, 20, 1),
                              label_stack.shape)

        label_vals = np.unique(label_stack)
        # the way label loading is currently implemented, there should not be
        # more than 3 different classes in the loaded labels
        self.assertTrue(len(label_vals) <= 3)

    def test_create_tf_dataset(self):
        dset = Dataset(image_dir=self.image_dir,
                       label_dir=self.label_dir,
                       val_ratio=0.2)

        tf_dset, class_ratio = dset.create_tf_dataset(purpose='val',
                                                      return_stats=True)
        self.assertTrue(np.ndarray == type(class_ratio))


class TestDataAugmenter(unittest.TestCase):

    def setUp(self) -> None:
        # create mock tensorflow dataset containing one image and label
        self.np_image = np.random.random(size=(1, 20, 20, 3))
        tf_image = tf.convert_to_tensor(self.np_image, dtype=tf.float32)

        self.np_label = np.random.randint(low=0, high=3, size=(1, 20, 20, 1))
        tf_label = tf.convert_to_tensor(self.np_label, dtype=tf.float32)

        self.tf_dset = tf.data.Dataset.from_tensor_slices({'image': tf_image,
                                                           'label': tf_label})

    def test_augment_train_images(self):
        aug = DataAugmenter()
        aug_set = self.tf_dset.map(aug.augment_train_images)
        aug_img, aug_label = next(aug_set.as_numpy_iterator())

        self.assertTupleEqual((20, 20, 3), aug_img.shape)
        self.assertTupleEqual((20, 20, 1), aug_label.shape)

        # assert that label values have not changed
        self.assertTrue(np.array_equal(np.unique(self.np_label),
                                       np.unique(aug_label)))

    def test_load_validation_images(self):
        aug = DataAugmenter()
        aug_set = self.tf_dset.map(aug.load_validation_images)
        aug_img, aug_label = next(aug_set.as_numpy_iterator())

        # assert that values have not changed
        np.testing.assert_allclose(self.np_image.squeeze(), aug_img)
        self.assertTrue(np.array_equal(self.np_label[0], aug_label))
