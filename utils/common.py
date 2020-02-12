import cv2
import numpy as np
import tensorflow as tf


def polygon2mask2(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        polygons[i] = np.asarray(polygon, np.int32).reshape(-1,2)
    cv2.fillPoly(mask, polygons, color=1)
    return mask


def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def learning_rate_decay(lr, decay=0.999, min_lr=1e-7):
    lr = min_lr if lr <= min_lr else lr*decay
    return lr


def nms(boxes, scores, first_classes, second_classes, label_values1, label_values2,
        total_classes=range(80),
        max_num_boxes=50,
        activation_threshold=0.7,
        nms_threshold=0.5,
        activate_second_cate=False):
    # boxes shape: [-1, 4]; scores shape: [-1]; classes shape: [-1]
    # box: [x_min, y_min, x_max, _y_max]

    boxes_list, labels_list, scores_list, second_labels_list = [], [], [], []
    labels_value_out1, labels_value_out2 = [], []
    boxes = tf.reshape(boxes, [-1, 4])
    scores = tf.reshape(scores, [-1])
    first_classes = tf.reshape(first_classes, [-1])
    label_values1 = tf.reshape(label_values1, [-1])
    if activate_second_cate:
        second_classes = tf.reshape(second_classes, [-1])
        label_values2 = tf.reshape(label_values2, [-1])
    for cate in total_classes:
        class_mask = tf.equal(first_classes, cate)
        scores_mask = scores >= activation_threshold
        activate_mask = class_mask & scores_mask
        activate_boxes = tf.boolean_mask(boxes, activate_mask)
        activate_scores = tf.boolean_mask(scores, activate_mask)
        activate_fc = tf.boolean_mask(first_classes, activate_mask)
        activate_fcv = tf.boolean_mask(label_values1, activate_mask)
        if activate_second_cate:
            activate_sc = tf.boolean_mask(second_classes, activate_mask)
            activate_scv = tf.boolean_mask(label_values2, activate_mask)
        else:
            activate_sc = None
            activate_scv = None
        activate_indices = tf.image.non_max_suppression(boxes=activate_boxes,
                                                        scores=activate_scores,
                                                        max_output_size=max_num_boxes,
                                                        iou_threshold=nms_threshold)
        after_nms_boxes = tf.gather(activate_boxes, activate_indices)
        after_nms_scores = tf.gather(activate_scores, activate_indices)
        after_nms_fc = tf.gather(activate_fc, activate_indices)
        after_nms_fcv = tf.gather(activate_fcv, activate_indices)
        if activate_second_cate:
            after_nms_sc = tf.gather(activate_sc, activate_indices)
            after_nms_scv = tf.gather(activate_scv, activate_indices)
        else:
            after_nms_sc = None
            after_nms_scv = None
        boxes_list.append(after_nms_boxes)
        scores_list.append(after_nms_scores)
        labels_list.append(after_nms_fc)
        labels_value_out1.append(after_nms_fcv)
        if activate_second_cate:
            second_labels_list.append(after_nms_sc)
            labels_value_out2.append(after_nms_scv)
    boxes = tf.concat(boxes_list, axis=0)
    scores = tf.concat(scores_list, axis=0)
    labels = tf.concat(labels_list, axis=0)
    labels_value_out1 = tf.concat(labels_value_out1, axis=0)
    if activate_second_cate:
        second_labels = tf.concat(second_labels_list, axis=0)
        labels_value_out2 = tf.concat(labels_value_out2, axis=0)
    else:
        second_labels = []
        labels_value_out2 = []
    return boxes, scores, labels, second_labels, labels_value_out1, labels_value_out2


keep_5 = lambda x: round(x, 5)


def show_loss(name, loss, avg_loss, spend_time):
    print(name)
    print("loss_oc: ", keep_5(loss[0]), ",",
          "loss_xy: ", keep_5(loss[1]), ",",
          "loss_wh: ", keep_5(loss[2]), ",",
          "loss_cate: ", keep_5(loss[3]), ",",
          "loss_nooc: ", keep_5(loss[4]), ",",
          "total_loss: ", keep_5(loss[5]), ",",
          "avg_loss: ", keep_5(avg_loss), ","
          "spend time: ", spend_time, "s"
          )

class Bidict(object):
    def __init__(self, init_dict=None):
        if init_dict is None:
            self.forward_dict = dict()
            self.inverse_dict = dict()
        else:
            self.forward_dict = init_dict
            self.inverse_dict = dict()
            for key in self.forward_dict.keys():
                value = self.forward_dict[key]
                self.inverse_dict[value]=key
    def __setitem__(self,key,value):
        self.forward_dict[key] = value
        self.inverse_dict[value] = key
    def __getitem__(self, key):
        return self.forward_dict[key]
    def __len__(self):
        return len(self.forward_dict.keys())
    def value2key(self, value):
        return self.inverse_dict[value]
    def keys(self):
        return self.forward_dict.keys()
    def values(self):
        return self.inverse_dict.keys()