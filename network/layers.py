import tensorflow as tf
import network.functions as func


def conv(input_tensor, kernel_shape, stride, name,
         padding="SAME",
         bn=True,
         leakyrelu=True,
         bias=False,
         trainable=True):
    # kernel_shape = [kernel_h, keinel_w, channel_out]
    # stride = [1, stride_h, stride_w, 1]
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs


    kernel_h, keinel_w, channel_out = kernel_shape
    _, stride_h, stride_w, _ = stride
    if stride_h > 1:
        input_tensor = _fixed_padding(input_tensor, kernel_h)
    convnormal = lambda in_tensor, k, pad: tf.nn.conv2d(in_tensor, k, [1, stride_h, stride_w, 1], padding=pad)
    channel_in = input_tensor.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        init_weights = func.init_truncated_normal(0.0, stddev=0.01)
        kernel = tf.get_variable("weight", [kernel_h, keinel_w, channel_in, channel_out],
                                 initializer=init_weights, trainable=trainable)
        conv_out = convnormal(input_tensor, kernel, ('SAME' if stride_h == 1 else 'VALID'))
        if bias:
            bias_para = tf.get_variable("bias", [channel_out],
                                        initializer=tf.zeros_initializer(),
                                        trainable=trainable)
            conv_out = tf.nn.bias_add(conv_out, bias_para)
        if bn:
            conv_out = batch_normalization(conv_out, trainable)
        if leakyrelu:
            conv_out = tf.nn.leaky_relu(conv_out, alpha=0.1)
        return conv_out


def conv_share(input, kernel_shape, stride, name,
         padding="SAME",
         leakyrelu=True,
         trainable=True,
         share_channel = "out"):
    # kernel_shape = [kernel_h, keinel_w, channel_out]
    # stride = [1, stride_h, stride_w, 1]
    # share_channel: "in" or "out"
    assert share_channel in ["in", "out"]
    kernel_h, keinel_w, channel_out = kernel_shape
    _, stride_h, stride_w, _ = stride
    convnormal = lambda in_tensor, k: tf.nn.conv2d(in_tensor, k, [1, stride_h, stride_w, 1], padding=padding)
    channel_in = input.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        init_weights = func.init_truncated_normal(0.0, stddev=0.01)
        if share_channel == "out":
            kernel = tf.get_variable("weight", [kernel_h, keinel_w, channel_in, 1],
                                     initializer=init_weights, trainable=trainable)
            kernel = tf.tile(kernel, [1,1,1,channel_out])
        else:
            kernel = tf.get_variable("weight", [kernel_h, keinel_w, 1, channel_out],
                                     initializer=init_weights, trainable=trainable)
            kernel = tf.tile(kernel, [1, 1, channel_in, 1])

        conv_out = convnormal(input, kernel)

        if leakyrelu:
            conv_out = tf.nn.leaky_relu(conv_out, alpha=0.2)
        return conv_out

def batch_normalization_cannot_rebuild(input_tensor, is_training, name='BatchNorm', moving_decay=0.999, eps=1e-5):
    shape = input_tensor.get_shape()
    # only for conv or fc
    assert len(shape) in [2, 4]
    param_shape = shape[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        beta = tf.get_variable('beta', param_shape,initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma',param_shape,initializer=tf.ones_initializer())

        # mean and var
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(input_tensor,axes,name='moments')
        # moving average
        ema = tf.train.ExponentialMovingAverage(moving_decay)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                            lambda:(ema.average(batch_mean),ema.average(batch_var)))

        return tf.nn.batch_normalization(input_tensor,mean,var,beta,gamma,eps)


def batch_normalization(input_tensor, is_training, name="BN", moving_decay=0.999, eps=1e-5):
    input_shape = input_tensor.get_shape()
    params_shape = input_shape[-1]
    axis = list(range(len(input_shape) - 1))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        mean, var = tf.nn.moments(input_tensor, axis)
        gamma = tf.get_variable('gamma',
                                params_shape,
                                initializer=tf.ones_initializer,
                                dtype=tf.float32)
        beta = tf.get_variable('beta',
                               params_shape,
                               initializer=tf.zeros_initializer,
                               dtype=tf.float32)
        moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=tf.zeros_initializer,
                                      dtype=tf.float32,
                                      trainable=False
                                      )
        moving_var = tf.get_variable('moving_var',
                                     params_shape,
                                     initializer=tf.ones_initializer,
                                     dtype=tf.float32,
                                     trainable=False
                                     )

        if is_training:
            update_moving_mean = tf.assign(moving_mean, mean * (1 - moving_decay) + moving_mean * moving_decay)
            update_moving_var = tf.assign(moving_var, var * (1 - moving_decay) + moving_var * moving_decay)
            with tf.control_dependencies([update_moving_mean, update_moving_var]):
                return tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(input_tensor, moving_mean, moving_var, beta, gamma, eps)


def upsample(input_tensor, out_shape, type="nearest_neighbor"):
    if type == "nearest_neighbor":
        new_height, new_width = out_shape
        upsample_net = tf.image.resize_nearest_neighbor(input_tensor, (new_height, new_width), name='upsample')

        return upsample_net


def conv_transpose(input, kernel_shape, output_shape, stride, name,
                   padding="SAME",
                   bn=True,
                   leakyrelu=True,
                   trainable=True):
    # kernel_shape = [kernel_h, kernel_w, channel_out]
    # output_shape = H, W
    # stride = [1, stride_h, stride_w, 1]
    input_shape = input.get_shape().as_list()
    batch_size = input_shape[0]
    kernel_h, kernel_w, channel_out = kernel_shape

    _, stride_h, stride_w, _ = stride
    height, width = output_shape
    channel_in = input.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        init_weights = func.init_bilinear_weight([kernel_h, kernel_w, channel_out, channel_in])
        kernel = tf.get_variable("weight", initializer=init_weights, trainable=trainable)
        conv_out = tf.nn.conv2d_transpose(input, kernel, [batch_size, height, width, channel_out],
                                       [1, stride_h, stride_w, 1], padding=padding)
        if bn:
            conv_out = batch_normalization(conv_out, trainable)
        if leakyrelu:
            conv_out = tf.nn.leaky_relu(conv_out, alpha=0.2)
    return conv_out
