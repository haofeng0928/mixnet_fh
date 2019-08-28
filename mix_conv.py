import tensorflow as tf
from keras.layers import DepthwiseConv2D


# def split_channels(total_filters, num_groups):
#     split = [total_filters // num_groups for _ in range(num_groups)]
#     split[0] += total_filters - sum(split)
#     return split
#
#
# def mixconv(inputs, kernel_sizes, strides, padding, depth_multiplier):
#     convs = []
#     for kernel_size in kernel_sizes:
#         convs.append(DepthwiseConv2D(kernel_size,
#                                      strides=strides,
#                                      padding=padding,
#                                      depth_multiplier=depth_multiplier))
#
#     if len(convs) == 1:
#         return convs[0](inputs)
#     filters = inputs.shape[-1].value
#     splits = split_channels(filters, len(convs))
#     x_splits = tf.split(inputs, splits, -1)
#     x_outputs = [c(x) for x, c in zip(x_splits, convs)]
#     x = tf.concat(x_outputs, -1)
#     return x


from keras import backend as K
from keras.layers import concatenate
from keras_mixnets.custom_objects import MixNetConvInitializer


def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def mixconv(inputs, kernel_sizes=[3, 5], strides=[1, 1], padding='same', depth_multiplier=1):
    kernel_sizes = [3, 5, 7]
    layers = [DepthwiseConv2D(kernel_sizes[i],
                              strides=[1, 1],
                              padding='same',
                              dilation_rate=(1, 1),
                              use_bias=False,
                              kernel_initializer=MixNetConvInitializer())
              for i in range(len(kernel_sizes))]

    if len(layers) == 1:
        return layers[0](inputs)

    filters = K.int_shape(inputs)[-1]
    splits = _split_channels(filters, len(kernel_sizes))
    x_splits = tf.split(inputs, splits, -1)
    x_outputs = [c(x) for x, c in zip(x_splits, layers)]
    x = concatenate(x_outputs, axis=-1)
    return x

