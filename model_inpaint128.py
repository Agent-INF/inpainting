import tensorflow as tf
from ops import fully_connect
from ops import conv2d
from ops import transconv2d
from ops import batch_norm
from ops import lrelu


def generator(masked_image, batch_size, image_dim, is_train=True, no_reuse=False):
  with tf.variable_scope('generator128') as scope:
    if not (is_train or no_reuse):
      scope.reuse_variables()

    # input 128x128ximage_dim
    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(masked_image, 64, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 64x64x64

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 64, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 32x32x64

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 128, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 16x16x128

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 256, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 8x8x256

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 512, (4, 4), (2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 4x4x512

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = tf.reshape(hidden, [batch_size, 4 * 4 * 512])
      hidden = fully_connect(hidden, 4000, trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 4000

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = fully_connect(hidden, 4 * 4 * 512, trainable=is_train)
      hidden = tf.reshape(hidden, [batch_size, 4, 4, 512])
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 4x4x512

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=256,
                           kernel=(4, 4), stride=(2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 8x8x256

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=128,
                           kernel=(4, 4), stride=(2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 16x16x128

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=64,
                           kernel=(4, 4), stride=(2, 2), trainable=is_train)
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train))
    # output 32x32x64

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = transconv2d(hidden, output_channel=image_dim,
                           kernel=(4, 4), stride=(2, 2), trainable=is_train)
      hidden = tf.nn.tanh(hidden)
    # output 64x64ximage_dim

    return hidden


def discriminator(inpaint, batch_size, reuse=False, is_train=True):
  with tf.variable_scope('discriminator128') as scope:
    if reuse:
      scope.reuse_variables()

    # inpaint 64x64ximage_dim
    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(inpaint, 64, (4, 4), (2, 2), trainable=is_train)
      hidden = lrelu(batch_norm(hidden, train=is_train))
    # output 32x32x64

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 128, (4, 4), (2, 2), trainable=is_train)
      hidden = lrelu(batch_norm(hidden, train=is_train))
    # output 16x16x128

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 256, (4, 4), (2, 2), trainable=is_train)
      hidden = lrelu(batch_norm(hidden, train=is_train))
    # output 8x8x256

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 512, (4, 4), (2, 2), trainable=is_train)
      hidden = lrelu(batch_norm(hidden, train=is_train))
    # output 4x4x512

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = tf.reshape(hidden, [batch_size, 4 * 4 * 512])
      hidden = fully_connect(hidden, 1, trainable=is_train)
    # output 1

    return hidden[:, 0]