# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image


class Dataset:
    """
    Class handling all operations concerning data loading, splitting,
    and augmenting.
    """
    def __init__(self, image_dir, label_dir, val_ratio=0.2,
                 cross_validation=False, n_folds=None, normalise=False,
                 ignore=None, flip_left_right=0.5, flip_up_down=0.5,
                 brightness=True, gaussian=0.2, gaussian_std=0.03,
                 rotate_prob=0.2, rotate_range=90):
        """
        :param image_dir: string, directory where images can be found
        :param label_dir: string, directory where labels can be found -
        labels are expected to have the same name as corresponding image.
        :param val_ratio: float, ratio of samples with existing groundtruth to
        be used as validation set
        :param cross_validation: bool, wether to calculate crossvalidation folds
        :param n_folds: int, number of folds for crossvalidation
        :param normalise: wether to normalise images when loading
        :param ignore: list, image names to be ignored (the corresponding labels
        will automatically be ignored)
        :param flip_left_right: float, probability for doing a left-right flip
        for data augmentation
        :param flip_up_down: float, probability for doing a up_down flip for
        data augmentation
        :param brightness: bool, wether to do brightness augmentation on image
        :param gaussian: float, probability for adding gaussian noise to image
        as data augmentation
        :param gaussian_std: float, standard deviation for gaussian noise,
        only considered if gaussian > 0
        :param rotate_prob: float, probability for rotating data as augmentation
        :param rotate_range: float, if rotate_prob > 0, data will be rotated by
        a random angle in [-rotate_range, rotate_range] in degree.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.val_ratio = val_ratio
        self.cross_validation = cross_validation
        self.n_folds = n_folds

        # data augmentation
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down
        self.brightness = brightness
        self.gaussian = gaussian
        self.gaussian_std = gaussian_std
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range

        self.img_shape = None
        self.ignore = ignore or []
        self.normalise = normalise
        self.split_df = self._calculate_data_split(self.val_ratio,
                                                   self.cross_validation,
                                                   self.n_folds)

    def _calculate_data_split(self, val_ratio=0.2, cross_validation=False,
                              n_folds=None):
        """
        Split dataset into purpose (train/val/test) or crossvalidation folds
        and test.

        :param val_ratio:  ratio of samples with existing groundtruth to be used
        as validation set, ignored if cross_validation=True
        :param cross_validation: wether to calculate crossvalidation folds
        :param n_folds: number of folds for crossvalidation
        :return: dataframe with entries for each sample and its corresponding
        purpose / fold.
        """
        if not cross_validation:
            if not (0 <= val_ratio <= 1):
                raise ValueError('Please choose a val_ratio between 0 and 1.')
        else:
            if type(n_folds) is not int:
                raise TypeError('Only integers are allowed for n_folds.')

        # fix seed to ensure reproducible results
        np.random.seed(1234)

        img_list = list(os.walk(self.image_dir))[0][2]
        label_list = list(os.walk(self.label_dir))[0][2]

        # remove samples to be ignored
        for name in self.ignore:
            img_list.remove(name)
            label_list.remove(name)

        # test images are missing corresponding labels with same name as image
        is_test = np.array([img not in label_list for img in img_list])

        split_df = pd.DataFrame(data={'img_name': img_list})
        test_idx = np.where(is_test)[0]
        split_df.loc[test_idx, 'purpose'] = 'test'

        if not cross_validation:
            self.n_val = int(val_ratio * len(label_list))
            self.n_train = len(label_list) - self.n_val
            # indices available for training and validation
            train_val_idx = np.where(~is_test)[0]
            # choose training indices randomly
            train_idx = np.random.choice(train_val_idx, self.n_train,
                                         replace=False)
            # take remaining indices for validation
            val_idx = np.setdiff1d(train_val_idx, train_idx)
            split_df.loc[train_idx, 'purpose'] = 'train'
            split_df.loc[val_idx, 'purpose'] = 'val'
        else:
            # choose indices for folds of crossvalidation
            n_train_samples = len(split_df) - len(test_idx)
            self.n_sample_per_fold = int(n_train_samples / n_folds)
            split_df.loc[:, 'fold'] = None
            for i in range(n_folds-1):
                unassigned_idx = np.where(split_df.fold.isnull())[0]
                available_idx = np.setdiff1d(unassigned_idx, test_idx)
                # choose fold indices randomly from remaining indices
                fold_idx = np.random.choice(available_idx,
                                            self.n_sample_per_fold,
                                            replace=False)
                split_df.loc[fold_idx, 'fold'] = i

            # take remaining samples for last fold
            unassigned_idx = np.where(split_df.fold.isnull())[0]
            remaining_idx = np.setdiff1d(unassigned_idx, test_idx)
            split_df.loc[remaining_idx, 'fold'] = i + 1

        return split_df

    def create_tf_dataset(self, purpose=None, cross_validation=False, fold=None,
                          return_stats=False):
        """
        Create a tensorflow dataset object, containing the images specified in
        purpose/fold.

        :param purpose: string, 'train'/'val'/'test'
        :param cross_validation: bool, wether to get crossvalidation folds
        :param fold: int, fold for crossvalidation
        :param return_stats: bool, wether to return class_ratios as list
        :return: tensorflow dataset (, class_ratio)
        """
        np_images = self._load_images(purpose, cross_validation, fold) / 255
        np_labels, class_ratio = self._load_labels(purpose,
                                                   cross_validation,
                                                   fold,
                                                   return_stats=True)

        tf_images = tf.convert_to_tensor(np_images, dtype=tf.float32)
        tf_labels = tf.convert_to_tensor(np_labels, dtype=tf.float32)

        tf_set = tf.data.Dataset.from_tensor_slices({'image': tf_images,
                                                     'label': tf_labels
                                                     })

        if return_stats:
            return tf_set, class_ratio
        else:
            return tf_set

    @tf.function
    def augment_train_images(self, datapoint):
        """
        Augment a datapoint consisting of an image and corresponding label.

        Augmentation functions are specified for class.
        :param datapoint: dictionary for one datapoint. Needs keys 'image' and
        'label'. Images and labels can be numpy arrays or tf.Tensor objects.
        :return: tuple augmented image, augmented label
        """
        # take arguments as class functions
        img = datapoint['image']
        label = datapoint['label']

        if self.flip_left_right:
            do_flip = tf.random.uniform([]) < self.flip_left_right
            img = tf.cond(do_flip,
                          lambda: tf.image.flip_left_right(img),
                          lambda: img)
            label = tf.cond(do_flip,
                            lambda: tf.image.flip_left_right(label),
                            lambda: label)

        if self.flip_up_down:
            do_flip = tf.random.uniform([]) < self.flip_up_down
            img = tf.cond(do_flip,
                          lambda: tf.image.flip_up_down(img),
                          lambda: img)
            label = tf.cond(do_flip,
                            lambda: tf.image.flip_up_down(label),
                            lambda: label)

        if self.brightness:
            do_bright = tf.random.uniform([]) < self.brightness
            img = tf.cond(do_bright,
                          lambda: tf.image.random_brightness(img,
                                                             max_delta=0.1),
                          lambda: img)

        if self.gaussian:
            do_noise = tf.random.uniform([]) < self.gaussian
            std = tf.random.normal(shape=[1], stddev=self.gaussian_std)
            noise = tf.random.normal(shape=tf.shape(img), stddev=std)
            img = tf.cond(do_noise, lambda: tf.add(img, noise), lambda: img)
            img = tf.clip_by_value(img, 0.0, 1.0)

        if self.rotate_prob > 0:
            angle = tf.random.uniform([],
                                      minval=-self.rotate_range,
                                      maxval=self.rotate_range)
            do_rotate = tf.random.uniform([]) < self.rotate_prob
            img = tf.cond(do_rotate,
                          lambda: tfa.image.rotate(img, angle,
                                                   interpolation='BILINEAR'),
                          lambda: img)
            label = tf.cond(do_rotate,
                            lambda: tfa.image.rotate(label, angle,
                                                     interpolation='NEAREST'),
                            lambda: label)

        return img, label

    @tf.function
    def load_validation_images(self, datapoint):
        """
        Load a datapoint without any augmentations.

        :param datapoint: dictionary for one datapoint. Needs keys 'image' and
        'label'. Images and labels can be numpy arrays or tf.Tensor objects.
        :return: tuple image, label
        """
        return datapoint['image'], datapoint['label']

    def _load_images(self, purpose=None, cross_validation=False, fold=None):
        """
        Load all images with specified properties as one numpy array.

        For documentation on properties see self._filter_df.
        :param purpose: string, 'train'/'val'/'test'
        :param cross_validation: bool, wether to get crossvalidation folds
        :param fold: int, fold for crossvalidation
        :return: numpy array, shape (n_sample, x, y, n_channel)
        """
        filtered_df = self._filter_df(purpose, cross_validation, fold)
        images = []
        for path in filtered_df.img_name:
            img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            if self.normalise:
                img = tf.keras.utils.normalize(img)
            images.append(img)

        stack = np.stack(images)
        self.img_shape = stack.shape[1:]

        return stack

    def _load_labels(self, purpose=None, cross_validation=False, fold=None,
                     return_stats=False):
        """
        Load all labels with specified properties as one numpy array.

        For documentation on properties see self._filter_df.
        :param purpose: string,  'train'/'val'/'test'
        :param cross_validation: bool, wether to get crossvalidation folds
        :param fold: int, fold for crossvalidation
        :param return_stats: bool, wether to return ratio of different labels
        over all loaded data
        :return: numpy array, shape (n_samples, x, y, 1)
        or tuple of numpy array, class_ratios if return_stats is True
        """
        if purpose == 'test':
            raise AttributeError('Test images have no corresponding label.')

        filtered_df = self._filter_df(purpose, cross_validation, fold)
        labels = []
        for path in filtered_df.img_name:
            img_path = os.path.join(self.label_dir, path)
            label_img = Image.open(img_path).convert(mode='L')
            label = np.array(label_img)
            label_mask = np.zeros_like(label)
            label_mask[(0 < label) & (label < 255)] = 1
            label_mask[label == 255] = 2
            label_mask = np.expand_dims(label_mask.astype('int32'), axis=-1)
            labels.append(label_mask)

        label_stack = np.stack(labels)

        if not return_stats:
            return label_stack

        class_count = np.bincount(label_stack.astype('int').flatten())
        class_ratio = class_count / np.size(label_stack)
        return label_stack, class_ratio

    def _filter_df(self, purpose=None, cross_validation=False, fold=None):
        """
        Filter dataframe with all samples for specified properties.

        Purpose is ignored, when cross_validation=True
        :param purpose: one of 'train'/'val'/'test'
        :param cross_validation: wether to use cross validation folds
        :param fold: crossvalidation fold to filter for
        :return: dataframe containing only filtered entries
        """
        if cross_validation:
            assert type(fold) is int
            filtered_df = self.split_df[self.split_df.fold == fold]
        else:
            assert type(purpose) is str
            filtered_df = self.split_df[self.split_df.purpose == purpose]
        return filtered_df