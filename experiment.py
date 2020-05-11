# -*- coding: utf-8 -*-
import tensorflow as tf
from datetime import datetime
import functools

from data import Dataset, DataAugmenter
from model import UNetFactory
from losses import dice_loss, cce_loss, get_mixed_loss_function

"""
Example script for starting a training experiment for an image segmentation 
task with tensorflow and keras. 
"""
n_out_channels = 3  # number of output channels for neural network
batch_size = 4  # batch size for data sets

# configure where to find the data and what split should be used
dset = Dataset(image_dir='images',
               label_dir='labels',
               val_ratio=0.2,
               cross_validation=False,
               normalise=False,
               ignore=['278.png'])

# configure which data augmentations to use and how often
augmenter = DataAugmenter(flip_left_right=0.5,
                          flip_up_down=0.5,
                          brightness=True,
                          gaussian=0.3,
                          gaussian_std=0.03,
                          rotate_prob=0.2,
                          rotate_range=20)

# create training set
train_set, class_ratio = dset.create_tf_dataset('train', return_stats=True)
train_set = train_set.map(augmenter.augment_train_images)
train_set = train_set.shuffle(buffer_size=dset.n_train).batch(batch_size)

# create validation set
val_set = dset.create_tf_dataset('val')
val_set = val_set.map(augmenter.load_validation_images)
val_set = val_set.batch(batch_size)


# define model
model_factory = UNetFactory(input_shape=dset.img_shape,
                            n_blocks=4,
                            n_classes=n_out_channels,
                            batch_norm=True,
                            drop_out=True,
                            logits=True)
model = model_factory.get_model()

# mixed loss takes additional arguments, but passing arguments to loss
# function is not allowed by keras API, so use function_wrapper:
mixed_loss = get_mixed_loss_function(n_out_channels, class_ratio)


# define optimizer and additional metrics
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.1),
              loss=mixed_loss,
              metrics=[])

# define learning rate scheduler, to reduce learning rate once the validation
# loss stagnates.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 min_delta=0.001,
                                                 patience=10,
                                                 min_lr=0.0001,
                                                 cool_down=10,
                                                 verbose=1)

# create callbacks for tensorboard logs
time_string = datetime.now().strftime('%Y%m%d-%H%M%S')
logdir = 'logs/scalars/' + time_string
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# callback to save model with best validation loss
best_save_path = 'best_model' + time_string
model_saving = tf.keras.callbacks.ModelCheckpoint(best_save_path,
                                                  monitor='val_loss',
                                                  save_best_only=True)

# gather callbacks for training
callbacks = [tensorboard_callback, reduce_lr, model_saving]

# start training
model_history = model.fit(x=train_set,
                          epochs=10,
                          verbose=2,
                          validation_data=val_set,
                          workers=4,
                          use_multiprocessing=False,
                          callbacks=callbacks)

# save model again after training
save_name = 'test_model'
model.save_weights(save_name)
model.save(save_name)
