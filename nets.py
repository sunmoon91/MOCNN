import tensorflow as tf


def weight_variable_(shape, stddev=0.1, scope='weight'):
    with tf.variable_scope(scope):
        initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable_(shape, scope='bias'):
    with tf.variable_scope(scope):
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm3d_(x, is_training, eps=1e-8, decay=0.9, scope='BatchNorm3d'):
    with tf.variable_scope(scope):
        bn = tf.contrib.layers.batch_norm(x, decay=decay, epsilon=eps, updates_collections=None,
                                          is_training=is_training)
    return bn


def conv3d_(net,
            out_plane,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            is_training=False,
            is_batch_norm=False,
            scope=None):
    """Building the convolutional layers"""
    with tf.variable_scope(scope):
        in_plane = int(net.shape[4])
        w = weight_variable_([kernel_size, kernel_size, kernel_size, in_plane, out_plane])
        b = bias_variable_([out_plane])
        c = tf.nn.conv3d(net, w, strides=[1, stride, stride, stride, 1], padding=padding) + b
        if is_batch_norm:
            c = batch_norm3d_(c, is_training=is_training)
        if activation_fn:
            c = activation_fn(c)
    return c


def de_conv3d_(net,
               out,
               kernel_size=2,
               stride=2,
               padding='SAME',
               activation_fn=tf.nn.relu,
               scope=None):
    with tf.variable_scope(scope):
        in_plane = int(net.shape[4])
        w = weight_variable_([kernel_size, kernel_size, kernel_size, int(out.shape[4]), in_plane])
        b = bias_variable_([int(out.shape[4])])
        d = tf.nn.conv3d_transpose(net, w, tf.shape(out), [1, stride, stride, stride, 1], padding=padding) + b
        d = activation_fn(d)
    return d


def high_order_conv3d_a_(net,
                         out_plane,
                         kernel_size=3,
                         stride=1,
                         padding='SAME',
                         activation_fn=tf.nn.relu,
                         is_training=False,
                         is_batch_norm=False,
                         scope=None
                         ):
    with tf.variable_scope(scope):
        net = conv3d_(net, out_plane=out_plane, kernel_size=kernel_size, stride=stride, padding=padding,
                      activation_fn=activation_fn, is_training=is_training, is_batch_norm=is_batch_norm, scope='h')
        net = 0.5 * tf.square(net)
        net = conv3d_(net, out_plane=out_plane, kernel_size=1, stride=1, padding=padding,
                      activation_fn=activation_fn, is_training=is_training, is_batch_norm=is_batch_norm, scope='c')

    return net


def max_pool3d_(net,
                kernel_size=2,
                stride=2,
                padding='SAME',
                scope=None):
    with tf.variable_scope(scope):
        p = tf.nn.max_pool3d(net, ksize=[1, kernel_size, kernel_size, kernel_size, 1],
                             strides=[1, stride, stride, stride, 1], padding=padding)
    return p


def c_block_(net,
             out_plane,
             kernel_size=3,
             stride=1,
             padding='SAME',
             activation_fn=tf.nn.relu,
             is_training=False,
             is_batch_norm=False,
             is_pooling=False,
             scope=None):
    with tf.variable_scope(scope):
        if is_pooling:
            net = max_pool3d_(net, scope='m')

        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='h')
        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='c')
    return net


def a_block_(net,
             out_plane,
             kernel_size=3,
             stride=1,
             group=1,
             padding='SAME',
             activation_fn=tf.nn.relu,
             is_training=False,
             is_batch_norm=False,
             is_pooling=False,
             scope=None):
    with tf.variable_scope(scope):
        if is_pooling:
            net = max_pool3d_(net, scope='m')

        net = high_order_conv3d_a_(net,
                                   out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                                   padding=padding, activation_fn=activation_fn,
                                   is_training=is_training, is_batch_norm=is_batch_norm, scope='h')
        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='c')
    return net


def d_block_(input1,
             input2,
             out_plane,
             kernel_size=3,
             d_kernel_size=3,
             stride=1,
             d_stride=2,
             padding='SAME',
             activation_fn=tf.nn.relu,
             is_training=False,
             is_batch_norm=False,
             scope=None):
    with tf.variable_scope(scope):
        d = de_conv3d_(input1, out=input2, kernel_size=d_kernel_size, stride=d_stride, padding=padding,
                       activation_fn=activation_fn, scope='d')
        net = tf.concat([d, input2], 4, name='concat')
        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='h')
        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='c')
    return net


def a_d_block_(input1,
               input2,
               out_plane,
               kernel_size=3,
               d_kernel_size=3,
               stride=1,
               d_stride=2,
               group=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               is_training=False,
               is_batch_norm=False,
               scope=None):
    with tf.variable_scope(scope):
        d = de_conv3d_(input1, out=input2, kernel_size=d_kernel_size, stride=d_stride, padding=padding,
                       activation_fn=activation_fn, scope='d')
        net = tf.concat([d, input2], 4, name='concat')
        net = high_order_conv3d_a_(net,
                                   out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                                   padding=padding, activation_fn=activation_fn,
                                   is_training=is_training, is_batch_norm=is_batch_norm, scope='h')
        net = conv3d_(net,
                      out_plane=out_plane, kernel_size=kernel_size, stride=stride,
                      padding=padding, activation_fn=activation_fn,
                      is_training=is_training, is_batch_norm=is_batch_norm, scope='c')
    return net


def se_block_(input1,
              input2,
              ratio=4,
              scope=None):
    with tf.variable_scope(scope):
        net = tf.concat([input1, input2], 4, name='concat')
        rc = net.shape[4] // ratio
        squeeze_2 = tf.reduce_mean(net, [1, 2, 3], keepdims=True)
        excitation_2_1 = tf.nn.relu(tf.layers.dense(inputs=squeeze_2, use_bias=True, units=rc))
        excitation_2_2 = tf.nn.sigmoid(tf.layers.dense(inputs=excitation_2_1, use_bias=True, units=net.shape[4]))
        scale_2 = tf.reshape(excitation_2_2, [-1, 1, 1, 1, net.shape[4]])
        se_2 = net * scale_2
    return se_2


def se_block_4_(input1,
                input2,
                input3,
                input4,
                ratio=4,
                scope=None):
    with tf.variable_scope(scope):
        net = tf.concat([input1, input2, input3, input4], 4, name='concat')
        rc = net.shape[4] // ratio
        squeeze_2 = tf.reduce_mean(net, [1, 2, 3], keepdims=True)
        excitation_2_1 = tf.nn.relu(tf.layers.dense(inputs=squeeze_2, use_bias=True, units=rc))
        excitation_2_2 = tf.nn.sigmoid(tf.layers.dense(inputs=excitation_2_1, use_bias=True, units=net.shape[4]))
        scale_2 = tf.reshape(excitation_2_2, [-1, 1, 1, 1, net.shape[4]])
        se_2 = net * scale_2
    return se_2



def MO_AGUNet(inputs1,
              inputs2,
              num_class=1,
              is_training=True,
              is_batch_norm=True,
              scope='MO-AGUNet'):
    """
    :param inputs1:
    :param inputs2:
    :param num_class:
    :param is_training:
    :param is_batch_norm:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, 'MO-AGUNet'):
        b1 = c_block_(inputs1, out_plane=32, is_training=is_training, is_batch_norm=is_batch_norm, scope='b1')
        h1 = a_block_(inputs1, out_plane=32, is_training=is_training,
                      is_batch_norm=is_batch_norm, scope='h1')
        a1 = c_block_(inputs2, out_plane=32, is_training=is_training, is_batch_norm=is_batch_norm, scope='a1')
        ah1 = a_block_(inputs2, out_plane=32, group=1, is_training=is_training,
                       is_batch_norm=is_batch_norm, scope='ah1')
        s1 = se_block_4_(b1, h1, a1, ah1, scope='s1')

        b2 = c_block_(s1, out_plane=128, is_training=is_training, is_batch_norm=is_batch_norm, is_pooling=True,
                      scope='b2')
        h2 = a_block_(s1, out_plane=128, group=1, is_training=is_training, is_batch_norm=is_batch_norm,
                      is_pooling=True,
                      scope='h2')
        s2 = se_block_(b2, h2, scope='s2')

        b3 = c_block_(s2, out_plane=256, is_training=is_training, is_batch_norm=is_batch_norm, is_pooling=True,
                      scope='b3')
        h3 = a_block_(s2, out_plane=256, group=1, is_training=is_training, is_batch_norm=is_batch_norm,
                      is_pooling=True,
                      scope='h3')
        s3 = se_block_(b3, h3, scope='s3')

        b4 = d_block_(s3, s2, 128, is_training=is_training, is_batch_norm=is_batch_norm, scope='b4')
        h4 = a_d_block_(s3, s2, 128, group=1, is_training=is_training, is_batch_norm=is_batch_norm,
                        scope='h4')
        s4 = se_block_(b4, h4, scope='s4')

        b5 = d_block_(s4, s1, 64, is_training=is_training, is_batch_norm=is_batch_norm, scope='b5')
        h5 = a_d_block_(s4, s1, 64, group=1, is_training=is_training, is_batch_norm=is_batch_norm,
                        scope='h5')
        s5 = se_block_(b5, h5, scope='s5')

        out = conv3d_(s5, out_plane=num_class, kernel_size=1, activation_fn=None, scope='out')

        return out

