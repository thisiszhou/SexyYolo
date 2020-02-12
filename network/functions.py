import numpy as np
import tensorflow as tf


Seed = 32


def init_truncated_normal(avg, stddev=0.01, seed=Seed):
    return tf.truncated_normal_initializer(avg, stddev=stddev, seed=seed)


def init_bilinear_weight(shape, type="float32"):
    # shape is a kernel shape, such as [3,3,64,32]
    # only support shape[0] == shape[1]
    weights = np.zeros(shape)

    k_size = shape[0]
    factor = (k_size+1)//2
    if k_size % 2 == 0:
        center = factor - 0.5
    else:
        center = factor - 1

    row, col = np.ogrid[:k_size, :k_size]

    weights_2d = np.zeros([k_size, k_size, 1, 1])
    weights_2d[:,:,0,0] = (1 - abs(row - center)*1.0/factor) * (1 - abs(col - center)*1.0/factor)
    weights[...] = weights_2d
    if type == "float32":
        return np.array(weights, dtype=np.float32)
    else:
        raise ValueError("wrong data type!")


def cross_entropy_binary_loss(logits, labels, if_sigmoid=False, epsilon=1e-8, boolean=True):
    if if_sigmoid:
        logits = tf.sigmoid(logits)
    if boolean:
        ob_mask = labels >= 0
    def cross_entropy():
        if boolean:
            return -tf.reduce_sum(tf.boolean_mask(
                labels * tf.log(tf.clip_by_value(logits, epsilon, 1)) +
                (1 - labels) * tf.log(tf.clip_by_value(1 - logits, epsilon, 1)),
                ob_mask
                )
            )
        else:
            return -tf.reduce_sum(
                (
                        labels * tf.log(tf.clip_by_value(logits, epsilon, 1)) +
                        (1 - labels) * tf.log(tf.clip_by_value(1 - logits, epsilon, 1))
                )
            )
    tensor_type = logits.dtype
    tensor_available = tf.py_func(check_tensor, [logits], [tf.bool])
    tensor_available = tf.reshape(tensor_available, ())
    return tf.cond(
        tensor_available,
        lambda: cross_entropy(),
        lambda: tf.cast(0, tensor_type)
    )


def mse_loss(logits, labels, weights=None, if_sigmoid=False, if_exp=False, boolean=True):
    if if_sigmoid:
        logits = tf.sigmoid(logits)
    elif if_exp:
        logits = tf.exp(logits)
    if boolean:
        ob_mask = labels >= 0
    if weights is None:
        weights = 1

    def mse():
        if boolean:
            return tf.reduce_sum(tf.boolean_mask(tf.square(logits - labels) * weights, ob_mask))
        else:
            return tf.reduce_sum(tf.square(logits - labels) * weights)
    tensor_type = logits.dtype
    tensor_available = tf.py_func(check_tensor, [logits], [tf.bool])
    tensor_available = tf.reshape(tensor_available, ())
    return tf.cond(
        tensor_available,
        lambda: mse(),
        lambda: tf.cast(0, tensor_type)
    )


def check_tensor(input_tensor):
    shape = input_tensor.shape
    if shape[0] == 0:
        return False
    return True

def print_input_tensor(logits, labels):
    print("logits in func:", logits,"labels in func:", labels)
    return np.array([True], dtype=np.bool)