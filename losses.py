import tensorflow as tf

"""
Different loss functions, e.g. suitable for segmentation tasks.
"""


@tf.function
def dice_loss(y_true, y_pred, n_out_channels, class_ratio=None):
    """
    Compute the dice loss of two tensors.

    :param y_true: tf.tensor, tensor with true mask of type tf.int, containing
    integer labels for classes, dimension: (batch, x, y)
    :param y_pred: tf.tensor, tensor with predicted mask as logits. One slice
    for each class, dimension: (batch, x, y, class)
    :param n_out_channels: int, number of output channels for the prediction of
    the network
    :param class_ratio: list, class frequency ratios. If provided, the loss of
    class i will be weighted by 1-class_ratio[i]
    :return: tf.float32, tensor with dice loss
    """
    loss = tf.convert_to_tensor(0, dtype=tf.float32)

    # channel 0 is background, so start at 1
    for i in range(1, n_out_channels):
        # ground truth for this channel
        y_true_mask = tf.cast(tf.equal(y_true, i), dtype=tf.float32)
        y_true_f = tf.keras.backend.batch_flatten(y_true_mask)
        # prediction for this channel
        y_pred_sig = tf.keras.activations.softmax(y_pred[:, :, :, i])
        y_pred_f = tf.keras.backend.batch_flatten(y_pred_sig)

        epsilon = tf.constant(1e-5)
        # numerator
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f,
                                            axis=1,
                                            keepdims=True)
        num = 2. * intersection + epsilon
        # denominator
        denom = (tf.keras.backend.sum(y_true_f, axis=1, keepdims=True)
                 + tf.keras.backend.sum(y_pred_f, axis=1, keepdims=True)
                 + epsilon)
        dice_coef = tf.keras.backend.mean(num / denom)
        layer_loss = 1 - dice_coef
        loss = tf.add(tf.reduce_mean(layer_loss), loss)

    loss = tf.math.truediv(loss, tf.cast(n_out_channels - 1, dtype=tf.float32))

    return loss


@tf.function
def cce_loss(y_true, y_pred, n_out_channels, class_ratio=None):
    """"
    Compute the categorical cross entropy between y_true and y_pred.

    :param y_true: tf.tensor, true mask of type tf.int, containing integer
    labels for classes, dimension: (batch, x, y)
    :param y_pred: tf.tensor, predicted mask as logits. One slice for each
    class, dimension: (batch, x, y, class)
    :param n_out_channels: int, Number of output channels in prediction
    :param class_ratio: list, class frequency ratios. If provided, the loss of
    class i will be weighted by 1-class_ratio[i]
    :return: tf.tensor, dice loss (tf.float32 scalar)
    """

    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    if class_ratio is None:
        class_ratio = [0] * n_out_channels

    # class weights
    neg_weights = tf.multiply(-1., tf.cast(class_ratio, dtype=tf.float32))
    w = tf.add([1.] * n_out_channels, neg_weights)

    for i in range(n_out_channels):
        logit = tf.keras.backend.flatten(y_pred[:, :, :, i])
        label = tf.keras.backend.flatten(tf.cast(tf.equal(y_true, i),
                                                 dtype=tf.float32))
        layer_loss = tf.nn.weighted_cross_entropy_with_logits(labels=label,
                                                              logits=logit,
                                                              pos_weight=w[i])
        loss = tf.add(tf.reduce_mean(layer_loss), loss)

    loss = tf.math.truediv(loss, tf.cast(n_out_channels, dtype=tf.float32))
    return loss


def get_mixed_loss_function(n_out_channels, class_ratio=None):
    """
    Return mixed loss function with additional arguments already set.

    This wrapper is necessary, as the keras training API does not allow for
    additional arguments in loss functions.

    :param n_out_channels: int, Number of output channels in prediction
    :param class_ratio: list, class frequency ratios. If provided,
    the cce_loss of class i will be weighted by 1-class_ratio[i]
    :return: function, mixed_loss function
    """

    @tf.function
    def mixed_loss(y_true, y_pred, sample_weight=None):
        """
        Calculate weighted loss from weighted categorical cross entropy and dice
        loss.

        :param y_true: tf.tensor, tensor with true mask of type tf.int,
        containing integer labels for classes, dimension: (batch, x, y)
        :param y_pred: tf.tensor, tensor with predicted mask as logits. One
        slice for each class, dimension: (batch, x, y, class)
        :param sample_weight: This parameter is not used and only exists,
        because some versions of the keras training API expect loss functions
        to have it.
        :return: tf.float32, tensor with mixed loss
        """
        cross = cce_loss(y_true, y_pred, n_out_channels,
                         class_ratio=class_ratio)
        dice = dice_loss(y_true, y_pred, n_out_channels)
        loss = 1.0 * cross + 0.15 * dice
        return loss

    return mixed_loss
