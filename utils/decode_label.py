from utils.visualizer import show_annotation
import numpy as np
import tensorflow as tf


def label_encode_for_yolo(boxes_batch_label, obj_nums_label,
                          regs_pre1, regs_pre2, regs_pre3,
                          decode_boxes1, decode_boxes2, decode_boxes3,
                          image_size, sampled, coco_classes, prior_boxes,
                          classify_threshold,
                          debug_model,
                          valid_iou):
    # boxes shape: [num_of_object, 5]; 5: x, y, w, h, categoty
    # obj_nums: [batch_size, ]: object_num_of_pic1, object_num_of_pic2, object_num_of_pic3 ...
    # regs_pre: regs.shape: (bz, 15, 15, 3, 86) (bz, 30, 30, 3, 86) (bz, 60, 60, 3, 86)
    # return: label need to be encoded

    regs_pre_list = [regs_pre1, regs_pre2, regs_pre3]
    decode_boxes = [decode_boxes1, decode_boxes2, decode_boxes3]
    batch_size = len(obj_nums_label)
    size1 = int(image_size / sampled)
    size2 = size1 * 2
    size3 = size2 * 2
    # sizes: 15,30,60

    # first: exist object and need to be trained
    # second: exist object need to be trained and object don't need to be trained
    oj1_bool = np.zeros([batch_size, size1, size1, 3, 2], dtype=bool)
    oj2_bool = np.zeros([batch_size, size2, size2, 3, 2], dtype=bool)
    oj3_bool = np.zeros([batch_size, size3, size3, 3, 2], dtype=bool)
    ojs_bool = [oj1_bool, oj2_bool, oj3_bool]

    regs_label_encode1 = []
    regs_label_encoed2 = []
    regs_label_encode3 = []
    regs_label_encode = [regs_label_encode1, regs_label_encoed2, regs_label_encode3]

    index = 0
    ratios = [sampled / (2 ** fp_index) for fp_index in range(3)]
    for batch_i, num in enumerate(obj_nums_label):
        # per image
        if num >= 0:
            boxes_label = boxes_batch_label[index: index + num]
            encode_one_image(boxes_label, regs_pre_list, decode_boxes, ratios, coco_classes, prior_boxes,
                             regs_label_encode, ojs_bool, batch_i, image_size, valid_iou, debug_model)
            index += num
        else:
            cate_label = boxes_batch_label[index]
            encode_one_image_classify(regs_pre_list, decode_boxes, regs_label_encode, cate_label, ojs_bool,  batch_i,
                                      coco_classes,
                                      classify_threshold,
                                      prior_boxes, ratios, valid_iou, debug_model)
            index += 1

    boxes_label_encode = []
    scale_weights = []
    for label_encode in regs_label_encode:
        boxes_label_encode.append(np.array([x[1] for x in label_encode], dtype=np.float32))
        scale_weights.append(np.array([x[2] for x in label_encode], dtype=np.float32))
    return (
        boxes_label_encode[0], boxes_label_encode[1], boxes_label_encode[2],
        ojs_bool[0], ojs_bool[1], ojs_bool[2],
        scale_weights[0], scale_weights[1], scale_weights[2]
    )


def encode_one_image_classify(regs_pre_list, decode_boxes, label_encode, cate_label, ojs_bool, batch_i,
                              coco_classes,
                              classify_threshold,
                              prior_boxes,
                              ratios,
                              valid_iou,
                              debug_model):

    for i in range(len(ojs_bool)):
        ojs_bool[i][batch_i, :, :, :, 1] = True

    if debug_model :
        print("batch_i:", batch_i)
        # check another two fp
        for i in range(0, 3):
            regs_pre = regs_pre_list[i]
            oc = sigmoid(regs_pre[batch_i, :, :, :, 4])
            cate = sigmoid(regs_pre[batch_i, :, :, :, 5])
            conf = oc * cate
            maxxy = np.where(conf == np.max(conf))
            print("feature map", str(i), ":")
            print("max oc", oc[maxxy])
            print("max cate", cate[maxxy])
            print("maxxy:", maxxy)
        cate = sigmoid(regs_pre_list[0][batch_i, :, :, :, 5])
    # train only one feature map
    train_fp = 0
    label_encode_batch_i = [[], [], []]
    first_cate, sec_cate = cate_label[:2]
    regs_pre = regs_pre_list[train_fp][batch_i]

    cate_pre = np.argmax(regs_pre[..., 5: 5 + coco_classes], axis=-1)
    valid_cate_arg = (cate_pre == int(first_cate) - 1)
    oc = sigmoid(regs_pre[:, :, :, 4])

    # first oc < 0.5 -----> negative object
    non_oc_arg = (oc < 0.5) & valid_cate_arg
    ojs_bool[train_fp][batch_i, non_oc_arg, 1] = False

    # second 0.5 =< oc < 0.7 ------> ignore

    # third oc >= 0.7 (classify_threshold) ------> positive object
    valid_index = (oc >= classify_threshold) & valid_cate_arg
    y_valid, x_valid, box_valid = np.where(valid_index)

    valid_oc = oc[y_valid, x_valid, box_valid]
    order_of_valid = np.argsort(valid_oc)[::-1]
    valid_boxs = None
    for j in order_of_valid:
        y, x, box_i = y_valid[j], x_valid[j], box_valid[j]
        one_ob_label = np.array(regs_pre[y, x, box_i])
        box_pre = decode_boxes[train_fp][batch_i, y, x, box_i]
        over_lap = False
        if valid_boxs is None:
            valid_boxs = np.array([box_pre], dtype=np.float32)
        else:
            biou1, biou2 = special_iou(box_pre, valid_boxs)
            if (np.max(biou1) > valid_iou) or (np.max(biou2) > valid_iou):
                over_lap = True
            else:
                valid_boxs = np.row_stack((valid_boxs, box_pre))

        if not over_lap:
            if debug_model:
                # visual
                print("sexy:")
                print("prior_boxes:", prior_boxes[train_fp*3 + box_i])
                print("box:", box_i, "radio:", ratios[train_fp], "y:", y, "x:", x)
                print("pre oc:", oc[y, x, box_i], "cate c:", cate[y, x, box_i])
                print("pre joint:", sigmoid(one_ob_label[-3:]))
                print("true label:", int(sec_cate))
                show_annotation("debug_info/test_sexy_ob" + str(train_fp) + ".jpg", None, box_pre)
                # visual end
            ojs_bool[train_fp][batch_i, y, x, box_i, 0] = True
            one_ob_label[:] = 0
            one_ob_label[: 4] = -1
            one_ob_label[4] = 1
            one_ob_label[5 + int(first_cate) - 1] = 1
            one_ob_label[5 + coco_classes + int(sec_cate) - 1] = 1
            label_encode_batch_i[train_fp].append((y * 10000 + x, one_ob_label, 1))

    for i, label_encode_one in enumerate(label_encode_batch_i):
        label_encode_one.sort(key=lambda x_v: x_v[0])
        label_encode[i].extend(label_encode_one)


def encode_one_image(boxes, regs_pre_list, decode_boxes, ratios, coco_classes, prior_boxes,
                     label_encode, ojs_bool, batch_i, image_size,
                     ignore_threshold, debug_model):
    # return: three reg feature map, mask feature map
    # prior_boxes: area from big to small, big for reg1, small for reg3
    # boxes shape: [num_of_object, 5]
    # masks shape: [num_of_object, image_size, image_size]

    center_x = boxes[:, 0] + boxes[:, 2] / 2.0
    center_y = boxes[:, 1] + boxes[:, 3] / 2.0
    label_encode_batch_i = [[], [], []]
    # ratio_dict = {0: 32., 1: 16., 2: 8.}

    for i, box in enumerate(boxes):
        box_label, cate_label = box[:4], box[4]
        # calculate iou
        #print("box_label:", box_label)
        #print("decode_boxes[0][batch_i]:", decode_boxes[0][batch_i], decode_boxes[0][batch_i].shape)
        biou0 = box_iou(box_label, decode_boxes[0][batch_i], "absolute")
        max_biou, max_arg = np.max(biou0), 0
        biou1 = box_iou(box_label, decode_boxes[1][batch_i], "absolute")
        if np.max(biou1) > max_biou:
            max_biou = np.max(biou1)
            max_arg = 1
        biou2 = box_iou(box_label, decode_boxes[2][batch_i], "absolute")
        if np.max(biou2) > max_biou:
            max_biou = np.max(biou2)
            max_arg = 2
        biou = [biou0, biou1, biou2]

        if max_biou < 1e-8:
            continue

        # ignore the grid which object confidence more than ignore_threshold
        for fp_index, oj_bool in enumerate(ojs_bool):
            ojs_bool[fp_index][batch_i, biou[fp_index] >= ignore_threshold, 1] = True

        # generate max iou grid label
        fp_index = max_arg
        biou = biou[fp_index]
        y, x, box_i = [index[0] for index in np.where(biou==max_biou)]
        ojs_bool[fp_index][batch_i, y, x, box_i] = True

        one_ob_label = np.array(regs_pre_list[fp_index][batch_i, y, x, box_i])

        # decode pre output
        if debug_model:
            # visual
            # encode and calculate label
            #biou = box_iou(box_label, box_pre, "absolute")
            print("normal:")
            print("pre cate:", np.argmax(one_ob_label[5: -3]) + 1)
            #print("pre joint:", sigmoid(one_ob_label[-3:]))
            print("conf:", sigmoid(one_ob_label[4]))
            print("max biou:", max_biou)
            print("index:", fp_index, y, x, box_i)
            box_pre = decode_boxes[fp_index][batch_i, y, x, box_i]

            print("box_label:", box_label)
            print("true label", int(cate_label))
            show_annotation("debug_info/test_ob.jpg", box_label, box_pre)

            # visual end

        # one pre-box only assign to one label
        decode_boxes[fp_index][batch_i, y, x, box_i] = 0
        # generate reg label
        pw, ph = prior_boxes[fp_index*3 + box_i]
        one_ob_label[0] = center_x[i] / ratios[fp_index] - x
        one_ob_label[1] = center_y[i] / ratios[fp_index] - y
        one_ob_label[2] = box_label[2] * 1.0 / pw
        one_ob_label[3] = box_label[3] * 1.0 / ph
        one_ob_label[4] = 1
        one_ob_label[5:] = 0
        one_ob_label[5 + int(cate_label) - 1] = 1
        one_ob_label[5 + coco_classes:] = -1

        scale_weight = 2 - (box_label[2]/image_size) * (box_label[3]/image_size)
        label_encode_batch_i[fp_index].append((y*10000 + x, one_ob_label, scale_weight))

    for i, label_encode_one in enumerate(label_encode_batch_i):
        label_encode_one.sort(key=lambda x_v: x_v[0])
        label_encode[i].extend(label_encode_one)

def get_decode_matrix(grid_y, grid_x, prior_box):
    box_matrix = np.zeros([grid_y, grid_x, 3, 2], dtype=np.float32)
    xy_offset = np.zeros([grid_y, grid_x, 3, 2], dtype=np.float32)

    for i in range(3):
        box_matrix[:, :, i] = prior_box[i]

    for y in range(grid_y):
        for x in range(grid_x):
            xy_offset[y, x, :, 0] = x
            xy_offset[y, x, :, 1] = y
    return tf.constant(box_matrix), tf.constant(xy_offset)


def nearest_box(input_box, prior_boxes, iou_type):
    # input_box: numpy shape [4,] [x, y, w, h]
    # prior_boxes: numpy shape of [9, 4]
    # iou_type: 'absolute' or 'relative'
    iou = box_iou(input_box, prior_boxes, iou_type)
    nearest_index = np.argmax(iou)
    return nearest_index


def special_iou(box_pre, valid_boxs):
    box1 = np.expand_dims(box_pre, 0)
    x11, y11, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x21, y21, w2, h2 = valid_boxs[:, 0], valid_boxs[:, 1], valid_boxs[:, 2], valid_boxs[:, 3]
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2
    center_x1, center_y1 = np.maximum(x11, x21), np.maximum(y11, y21)
    center_x2, center_y2 = np.minimum(x12, x22), np.minimum(y12, y22)
    intersection = np.maximum(0, center_x2 - center_x1) * np.maximum(0, center_y2 - center_y1)
    return intersection / (w2 * h2 + 1e-10), intersection / (w1 * h1 + 1e-10)


def box_iou(box1, box2, iou_type):
    # box1: numpy of shape [4,]
    # box2: numpy of shape [n, 4]
    box1 = np.array(box1)
    box2 = np.array(box2)
    if len(box1.shape) == 1:
        box1 = np.expand_dims(box1, 0)
    else:
        box1 = np.expand_dims(box1, -2)
    if len(box2.shape) == 1:
        box2 = np.expand_dims(box2, 0)
    if iou_type == "absolute":
        x11, y11, w1, h1 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
        x21, y21, w2, h2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
        x12, y12 = x11 + w1, y11 + h1
        x22, y22 = x21 + w2, y21 + h2
    elif iou_type == "relative":
        assert box2.shape[-1] == 2
        w1, h1 = box1[...,2], box1[...,3]
        w2, h2 = box2[...,0], box2[...,1]
        x12, y12 = w1, h1
        x22, y22 = w2, h2
        x11, y11, x21, y21 = 0, 0, 0, 0
    else:
        raise ValueError("wrong iou type!")
    center_x1, center_y1 = np.maximum(x11, x21), np.maximum(y11, y21)
    center_x2, center_y2 = np.minimum(x12, x22), np.minimum(y12, y22)

    intersection = np.maximum(0, center_x2 - center_x1) * np.maximum(0, center_y2 - center_y1)
    return intersection / (w1 * h1 + w2 * h2 - intersection + 1e-8)


def log(x, epsilon=1e-8):
    return np.log(x + epsilon)


def sigmoid(x):
    return 1.0 / (1 + clip_exp(-x))


def clip_exp(x, min=-100, max=50):
    return np.exp(np.clip(x, min, max))


def trans_sigmoid(x, epsilon=1e-8):
    return np.log((x + epsilon)/ (1 - x + epsilon))