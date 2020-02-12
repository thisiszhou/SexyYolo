from network.layers import conv, upsample
import tensorflow as tf


def darknet53_body(inputs, trainable):

    def res_block(inputs, filters, name):
        # filters = 1/2 inputs' channel
        cut_across = inputs
        net = conv(inputs, [1, 1, filters], [1, 1, 1, 1], name+"/conv_1", trainable=trainable)
        net = conv(net, [3, 3, filters*2], [1, 1, 1, 1], name+"/conv_2", trainable=trainable)
        return net + cut_across


    def res_operator(inputs, filters, num_of_res, name):
        net = conv(inputs, [3, 3, 2*filters], [1, 2, 2, 1], name+"/stride2_conv", trainable=trainable)
        for i in range(num_of_res):
            net = res_block(net, filters, name+"/"+str(i+1))
        return net

    net = conv(inputs, [3, 3, 32], [1, 1, 1, 1], "1_conv", trainable=trainable)

    net = res_operator(net, 32, 1, "res_block1")

    net = res_operator(net, 64, 2, "res_block2")
    net1 = res_operator(net, 128, 8, "res_block3")
    net2 = res_operator(net1, 256, 8, "res_block4")
    net3 = res_operator(net2, 512, 4, "res_block5")

    return net1, net2, net3


def yolo_fpn_head(nets, trainable):
    # nets is a list
    # the area of nets must be the order of big to small
    # return: feature map and upsample map
    def yolo_block(inputs, filters, name):
        net = conv(inputs, [1, 1, filters], [1, 1, 1, 1], name + "/1_conv", trainable=trainable)
        net = conv(net, [3, 3, filters * 2], [1, 1, 1, 1], name + "/2_conv", trainable=trainable)
        net = conv(net, [1, 1, filters], [1, 1, 1, 1], name + "/3_conv", trainable=trainable)
        net = conv(net, [3, 3, filters * 2], [1, 1, 1, 1], name + "/4_conv", trainable=trainable)
        net = conv(net, [1, 1, filters], [1, 1, 1, 1], name + "/5_conv", trainable=trainable)
        return net


    num_of_nets = len(nets)
    last_net = None
    filters_iter = 512
    fpn_maps = []

    for i in range(num_of_nets - 1, -1, -1):
        current_net = nets[i]
        if last_net is not None:
            shape = current_net.get_shape().as_list()
            last_net = conv(last_net, [1, 1, filters_iter], [1, 1, 1, 1],
                            "before_yb_conv_" + str(num_of_nets - i - 1),
                            trainable=trainable)
            last_net = upsample(last_net, (shape[1], shape[2]))
            current_net = tf.concat([last_net, current_net], axis=3)
        current_net = yolo_block(current_net, filters_iter, "yolo_block" + str(num_of_nets - i))
        last_net = current_net
        net = conv(current_net, [3, 3, filters_iter * 2], [1, 1, 1, 1],
                   str(i) + "_fp_final_3conv", trainable=trainable)
        fpn_maps.append(net)
        filters_iter = int(filters_iter / 2)

    return fpn_maps


def yolo_regression(fpn_maps, class_num, trainable):
    def reshape_reg(input_reg):
        shape = input_reg.get_shape()
        final_dim = int(reg_dim / 3)
        return tf.reshape(input_reg, (-1, shape[1], shape[2], 3, final_dim))

    reg_dim = 3 * (1 + 4 + class_num)
    feature_maps = []
    for i, net in enumerate(fpn_maps):
        net = conv(net, [1, 1, reg_dim], [1, 1, 1, 1],
                   str(2 - i) + "_fp_final_1conv",
                   bn=False,
                   leakyrelu=False,
                   bias=True,
                   trainable=trainable)
        net = reshape_reg(net)
        feature_maps.append(net)

    return feature_maps


def get_shape(input_tensor):
    shape = input_tensor.shape()
    return shape