import tensorflow as tf


def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)


def batch_norm(inputs, train=True, epsilon=1e-5, momentum=0.9, name="batch_norm"):
  return tf.contrib.layers.batch_norm(inputs, decay=momentum,
                                      updates_collections=None, epsilon=epsilon,
                                      scale=True, is_training=train, scope=name)


def conv_cond_concat(input_x, input_y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = input_x.get_shape()
  y_shapes = input_y.get_shape()
  return concat(
      [input_x, input_y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(inputs, output_channel=None, kernel=(4, 4), stride=(2, 2), stddev=0.02,
           padding="SAME", name="conv2d", trainable=True):
  """Convolution layer"""
  assert len(
      kernel) == 2, 'Wrong kernel shape! Expect 2 but found %d' % len(kernel)
  assert len(
      stride) == 2, 'Wrong stride shape! Expect 2 but found %d' % len(stride)

  with tf.variable_scope(name):
    weights = tf.get_variable(
        'weights', [kernel[0], kernel[1],
                    inputs.get_shape()[-1], output_channel],
        initializer=tf.truncated_normal_initializer(stddev=stddev),
        trainable=trainable)
    conv = tf.nn.conv2d(inputs, weights,
                        strides=[1, stride[0], stride[1], 1], padding=padding)
    biases = tf.get_variable(
        'biases', [output_channel],
        initializer=tf.constant_initializer(0.0), trainable=trainable)
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def transconv2d(inputs, output_channel=None, output_shape=None, kernel=(4, 4), stride=(2, 2),
                stddev=0.02, padding="SAME", name="deconv2d", trainable=True):
  """
  Deconvolution or Transpose-convolution layer, if output_channel not None and output_shape
  is None, than the output shape (height and width) is inputs shape multiply stride;
  otherwise use the output_shape as output_shape in tf.nn.conv2d_transpose()
  """
  assert len(
      kernel) == 2, 'Wrong kernel shape! Expect 2 but found %d' % len(kernel)
  assert len(
      stride) == 2, 'Wrong stride shape! Expect 2 but found %d' % len(stride)
  if output_shape is None:
    input_shape = inputs.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1] * stride[0],
                    input_shape[2] * stride[1], output_channel]

  with tf.variable_scope(name):
    weights = tf.get_variable(
        'weights', [kernel[0], kernel[1],
                    output_shape[-1], inputs.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev),
        trainable=trainable)
    deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape,
                                    [1, stride[0], stride[1], 1], padding=padding)
    biases = tf.get_variable(
        'biases', [output_shape[-1]],
        initializer=tf.constant_initializer(0.0), trainable=trainable)
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv


def lrelu(inputs, leak=0.2):
  """Leaky ReLU layer"""
  return tf.maximum(inputs, leak * inputs)


def prelu(inputs, name='prelu'):
  """Parametric ReLU layer"""
  with tf.variable_scope(name):
    alphas = tf.get_variable('alpha', inputs.get_shape()[-1],
                             dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    pos = tf.nn.relu(inputs)
    neg = tf.multiply(alphas, (inputs - tf.abs(inputs))) * 0.5
  return pos + neg


def pooling(inputs, kernel, stride, padding='SAME', name='pooling', pooltype='max'):
  """Pooling layer"""
  assert len(
      kernel) == 2, 'Wrong kernel shape! Expect 2 but found %d' % len(kernel)
  assert len(
      stride) == 2, 'Wrong stride shape! Expect 2 but found %d' % len(stride)
  assert pooltype == 'max' or pooltype == 'avg', 'Pooling type error!'

  if pooltype == 'max':
    return tf.nn.max_pool(
        inputs, [1, kernel[0], kernel[1], 1], [1, stride[0], stride[1], 1], padding, name=name)
  else:
    return tf.nn.avg_pool(
        inputs, [1, kernel[0], kernel[1], 1], [1, stride[0], stride[1], 1], padding, name=name)


def fully_connect(inputs, output_size, stddev=0.02,
                  bias_init=0.0, name="fully_connect", trainable=True):
  """Fully-connected layer"""
  shape = inputs.get_shape().as_list()

  with tf.variable_scope(name):
    weights = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer(stddev=stddev),
                              trainable=trainable)
    bias = tf.get_variable("biases", [output_size],
                           initializer=tf.constant_initializer(bias_init),
                           trainable=trainable)
    return tf.matmul(inputs, weights) + bias


def channel_wise_fc(inputs, stddev=0.02, bias_start=0.0, name='channel_wise_fc'):
  """Channel wise fully connected layer"""
  _, width, height, channel = inputs.get_shape().as_list()
  input_reshape = tf.reshape(inputs, [-1, width * height, channel])
  input_transpose = tf.transpose(input_reshape, [2, 0, 1])

  with tf.variable_scope(name):
    weights = tf.get_variable("weights", shape=[channel, width * height, width * height],
                              initializer=tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("biases", [channel, 1, width * height],
                           initializer=tf.constant_initializer(bias_start))
    output = tf.matmul(input_transpose, weights) + bias

  output_transpose = tf.transpose(output, [1, 2, 0])
  output_reshape = tf.reshape(output_transpose, [-1, height, width, channel])
  return output_reshape
