from network.blocks import darknet53_body, yolo_fpn_head, yolo_regression
from network.functions import mse_loss, cross_entropy_binary_loss
import tensorflow as tf
from utils.common import nms
import numpy as np
import cv2
import utils.visualizer as vis
from utils.datareader import DataSet
from utils.decode_label import label_encode_for_yolo, get_decode_matrix


BOXES = np.array([(116, 90), (156, 198), (373, 326),
                  (30, 61), (62, 45), (59, 119),
                  (10,13), (16,30), (33,23)], dtype=np.float32) # w, h


class YoloJoint(object):

    def __init__(self, trainable=None, category=None, debug_model=False):
        if trainable is None:
            raise ValueError("trainable must be setted")
        self.trainable = trainable
        self.sess = None
        self.debug_model = debug_model
        if not trainable:
            self.dataset = DataSet()
            self.dataset.load_categories_name(category)
    def init_sess(self, model_weights=None, pre_weights=None):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())

        if model_weights is not None:
            self.global_saver.restore(self.sess, model_weights)
            print("load model weights success!")
        elif pre_weights is not None:
            self.pre_train_saver.restore(self.sess, pre_weights)
            print("load pre-training model weights success!")

    def close_sess(self):
        self.sess.close()

    def save_model_weights(self, save_file, global_step=None):
        if self.sess is None:
            raise ValueError("sess has not been set")
        self.global_saver.save(self.sess, save_file, global_step=global_step)
        print("save model weights success!")

    def build(self,
              image_size,
              coco_classes,
              joint_classes,
              classes_weights=None,
              boxes_weights=None,
              oc_weights=None,
              nooc_weights=None,
              ignore_thresh=None,
              prior_boxes=BOXES,
              classify_threshold=None,
              sampled=32):

        self.coco_classes = coco_classes
        self.joint_classes = joint_classes
        self.image_size = image_size
        self.sampled = sampled
        self.prior_boxes = prior_boxes
        self.classify_threshold = classify_threshold
        # input: placeholder
        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        if self.trainable:
            self.boxes_label = tf.placeholder(tf.float32, [None, 5])
            self.objs_num = tf.placeholder(tf.int64, [None])
            # learning_rate
            self.lr = tf.placeholder(tf.float32, ())

            if self.debug_model:
                self.oc_threshold = tf.placeholder(tf.float32, ())
        else:
            self.oc_threshold = tf.placeholder(tf.float32, ())

            if self.debug_model:
                self.boxes_label = tf.placeholder(tf.float32, [None, 5])
                self.objs_num = tf.placeholder(tf.int64, [None])
                # learning_rate
                self.lr = tf.placeholder(tf.float32, ())
        # pre deal
        inputs = self.input_tensor / 255.0

        # backbone
        with tf.variable_scope('darknet53_body'):

            net1, net2, net3 = darknet53_body(inputs, trainable=self.trainable)

        # yolo fpn head
        with tf.variable_scope('yolov3_head'):
            fpn_maps = yolo_fpn_head([net1, net2, net3], trainable=self.trainable)

            self.pre_train_saver = tf.train.Saver()

            feature_maps = yolo_regression(fpn_maps, coco_classes + joint_classes, trainable=self.trainable)

        self.feature_maps = feature_maps
        self.global_saver = tf.train.Saver()

        if self.trainable or self.debug_model:
            self.box_matrix_list, self.offset_list = [], []
            for i, feature_map in enumerate(feature_maps):
                shape = feature_map.get_shape().as_list()
                box_matrix, xy_offset = get_decode_matrix(shape[1], shape[2], prior_boxes[3*i: 3*i + 3])
                self.box_matrix_list.append(box_matrix)
                self.offset_list.append(xy_offset)
            self.decode_boxes = self.decode_boxes_train()
            (
                boxes_label_encode1,
                boxes_label_encode2,
                boxes_label_encode3,
                ojs_bool1,
                ojs_bool2,
                ojs_bool3,
                scale_weight1,
                scale_weight2,
                scale_weight3
            ) = tf.py_func(label_encode_for_yolo,
                           [self.boxes_label,
                            self.objs_num,
                            self.feature_maps[0],
                            self.feature_maps[1],
                            self.feature_maps[2],
                            self.decode_boxes[0],
                            self.decode_boxes[1],
                            self.decode_boxes[2],
                            image_size,
                            sampled,
                            coco_classes,
                            BOXES,
                            self.classify_threshold,
                            self.debug_model,
                            ignore_thresh],
                           [tf.float32,
                            tf.float32,
                            tf.float32,
                            tf.bool,
                            tf.bool,
                            tf.bool,
                            tf.float32,
                            tf.float32,
                            tf.float32])

            self.boxes_label_encode = [boxes_label_encode1, boxes_label_encode2, boxes_label_encode3]
            self.ojs_bool = [ojs_bool1, ojs_bool2, ojs_bool3]
            self.scale_weights = [scale_weight1, scale_weight2, scale_weight3]
            # reshape
            size1 = int(image_size / sampled)
            size2 = size1 * 2
            size3 = size2 * 2
            fp_sizes = [size1, size2, size3]

            for i in range(3):
                self.boxes_label_encode[i] = tf.reshape(self.boxes_label_encode[i], [-1, coco_classes + joint_classes + 5])
                self.scale_weights[i] = tf.reshape(self.scale_weights[i], [-1, 1])
                self.ojs_bool[i] = tf.reshape(self.ojs_bool[i], [-1, fp_sizes[i], fp_sizes[i], 3, 2])

            self.get_loss(classes_weights, boxes_weights, oc_weights, nooc_weights)
        if self.joint_classes > 0:
            activate_second = True
        else:
            activate_second = False
        if not self.trainable or self.debug_model:
            boxes, scores, labels1, labels2, label_values1, label_values2, batch_indexs = self.decode_feature_map()
            boxes, scores, first_labels, second_labels, fcv, scv = nms(boxes, scores, labels1, labels2,
                                                                       label_values1, label_values2,
                                                                       activation_threshold=self.oc_threshold,
                                                                       activate_second_cate=activate_second)
            self.inference_info = [boxes, scores, first_labels, second_labels, fcv, scv]

    def decode_feature_map(self):

        prior_boxes = self.prior_boxes
        boxes_list, scores_list, obj_nums = [], [], []
        first_labels_list, second_labels_list, first_label_values, second_label_values = [], [], [], []
        for fp_index, feature_map in enumerate(self.feature_maps):
            prior_box = prior_boxes[3 * fp_index: 3 * (fp_index+1)]
            fp_shape = feature_map.get_shape().as_list()
            grid_size = fp_shape[1]
            radio = self.image_size / grid_size
            oc = tf.sigmoid(feature_map[..., 4])
            cate = tf.reduce_max(tf.sigmoid(feature_map[..., 5: 5 + self.coco_classes]))
            conf = oc * cate
            active_index = tf.where(conf >= self.oc_threshold)
            batch_i, y, x, box_i = active_index[:, 0], active_index[:, 1], active_index[:, 2], active_index[:, 3]
            active_fp = tf.gather_nd(feature_map, active_index)
            # active_fp shape: [-1, 88]
            pre_xy_center, pre_wh, pre_oc, pre_cate, pre_sec_cate = tf.split(
                active_fp,
                [2, 2, 1, self.coco_classes, self.joint_classes],
                axis=-1
            )
            pre_wh = tf.exp(pre_wh) * tf.gather(prior_box, box_i)

            xy_offset = tf.concat([
                tf.reshape(x, [-1, 1]),
                tf.reshape(y, [-1, 1])
            ], axis=-1)

            pre_xy_center = tf.sigmoid(pre_xy_center) + tf.cast(xy_offset, tf.float32)
            pre_xy_center = pre_xy_center * radio
            pre_xy_min, pre_xy_max = pre_xy_center - pre_wh / 2, pre_xy_center + pre_wh / 2,
            pre_xy_min = tf.clip_by_value(pre_xy_min, 0, self.image_size)
            pre_xy_max = tf.clip_by_value(pre_xy_max, 0, self.image_size)
            pre_oc = tf.sigmoid(pre_oc)
            pre_cate1_arg = tf.argmax(pre_cate, axis=-1)
            pre_cate1_value = tf.reduce_max(pre_cate, axis=-1)
            pre_cate1_value = tf.sigmoid(pre_cate1_value)
            if self.joint_classes > 0:
                pre_cate2_arg = tf.argmax(pre_sec_cate, axis=-1)
                pre_cate2_value = tf.reduce_max(pre_sec_cate, axis=-1)
                pre_cate2_value = tf.sigmoid(pre_cate2_value)
                second_labels_list.append(pre_cate2_arg)
                second_label_values.append(pre_cate2_value)
            boxes_list.append(tf.concat([pre_xy_min, pre_xy_max], axis=-1))
            scores_list.append(pre_oc)
            first_labels_list.append(pre_cate1_arg)
            first_label_values.append(pre_cate1_value)
            obj_nums.append(batch_i)
        boxes = tf.concat(boxes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        first_labels = tf.concat(first_labels_list, axis=0)
        first_label_values = tf.concat(first_label_values, axis=0)
        if self.joint_classes > 0:
            second_labels = tf.concat(second_labels_list, axis=0)
            second_label_values = tf.concat(second_label_values, axis=0)
        else:
            second_labels = []
            second_label_values = []
        obj_nums = tf.concat(obj_nums, axis=0)

        return boxes, scores, first_labels, second_labels, first_label_values, second_label_values, obj_nums

    def decode_boxes_train(self):

        pre_boxes_list = []
        for fp_index, feature_map in enumerate(self.feature_maps):
            ratio = self.sampled / (2 ** fp_index)
            pre_xy_center, pre_wh, _ = tf.split(
                feature_map,
                [2, 2, -1],
                axis=-1
            )
            # avoid crossing the boundary
            pre_wh = tf.clip_by_value(pre_wh, -50, 50)
            pre_wh = tf.exp(pre_wh) * self.box_matrix_list[fp_index]
            pre_xy_center = (tf.sigmoid(pre_xy_center) + self.offset_list[fp_index]) * ratio
            pre_xy = pre_xy_center - pre_wh / 2
            pre_box = tf.concat([pre_xy, pre_wh], axis=-1)
            pre_boxes_list.append(pre_box)

        return pre_boxes_list


    def get_loss(self, classes_weights, boxes_weights, oc_weights, nooc_weights):
        # input: label: self.boxes_label_encode, self.masks_label_encode, self.ojs_bool
        # input: pre: self.feature_maps
        num_feature_map = len(self.boxes_label_encode)

        # transfer with boolean_mask and calculate loss
        # regs order: 0 1: x y; 2 3 w h; coco_classes; joint_classes
        loss_oc = 0.
        loss_nooc = 0.
        loss_xy = 0.
        loss_wh = 0.
        loss_cate = 0.
        batch_size = tf.py_func(get_batch_size, [self.feature_maps[0]], [tf.float32])
        batch_size = tf.reshape(batch_size, ())
        for i in range(num_feature_map):
            boxes_pre = tf.boolean_mask(self.feature_maps[i], self.ojs_bool[i][..., 0])
            boxes_pre_noob = tf.boolean_mask(self.feature_maps[i], tf.logical_not(self.ojs_bool[i][..., 1]))
            # split
            pre_nooc = boxes_pre_noob[..., 4]
            pre_xy, pre_wh, pre_oc, pre_cate = tf.split(boxes_pre, [2, 2, 1, -1], axis=-1)
            label_xy, label_wh, label_oc, label_cate = tf.split(
                self.boxes_label_encode[i], [2, 2, 1, -1], axis=-1
            )
            #print("pre_wh shape", pre_wh.shape)
            #print("label_wh shape", label_wh.shape)
            #loss
            loss_oc += oc_weights * cross_entropy_binary_loss(pre_oc, label_oc, if_sigmoid=True)
            loss_wh += boxes_weights * mse_loss(pre_wh, label_wh, if_sigmoid=False, if_exp=True,
                                                weights=self.scale_weights[i])
            loss_xy += boxes_weights * mse_loss(pre_xy, label_xy, if_sigmoid=True, weights=self.scale_weights[i])
            loss_cate += classes_weights * cross_entropy_binary_loss(
                pre_cate, label_cate,
                if_sigmoid=True)
            loss_nooc += nooc_weights * cross_entropy_binary_loss(
                pre_nooc, 0,
                if_sigmoid=True,
                boolean=False)
        loss_list = [loss_oc, loss_xy, loss_wh, loss_cate, loss_nooc]
        loss_list = [x / batch_size for x in loss_list]

        total_loss = sum(loss_list)
        self.total_loss = total_loss
        loss_list.append(total_loss)
        self.loss = loss_list
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(total_loss)

    def train_one_batch(self, imgs, boxes, obj_nums, lr):

        if not self.debug_model:
            _, loss = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.input_tensor: imgs[..., ::-1],
                           self.boxes_label: boxes,
                           self.objs_num: obj_nums,
                           self.lr: lr}
            )

        else:
            loss = self.sess.run(
                [self.loss],
                feed_dict={self.input_tensor: imgs[..., ::-1],
                           self.boxes_label: boxes,
                           self.objs_num: obj_nums,
                           self.lr: lr}
            )

        return loss

    def infer_simple(self, imgs, positive_threshold, save_path, show_second_cate, image_size, sec_thresh=None):
        try:
            assert self.trainable == False
        except:
            raise ValueError("Please change trainable model to 'False'")
        scale = None
        imgs_o = None
        if isinstance(imgs, str):
            imgs_o, imgs, scale = self.dataset.imread(imgs, image_size)
        imgs = imgs[..., ::-1]
        inference_info = self.sess.run(
            self.inference_info,
            feed_dict={
                self.input_tensor: imgs,
                self.oc_threshold: positive_threshold
            }
        )

        boxes, scores, first_cate, second_cate, fcv, scv = inference_info
        if scale is not None:
            boxes[:, 0], boxes[:, 2] = boxes[:, 0] * scale[0], boxes[:, 2] * scale[0]
            boxes[:, 1], boxes[:, 3] = boxes[:, 1] * scale[1], boxes[:, 3] * scale[1]
            imgs = imgs_o[..., ::-1]
        else:
            imgs = imgs[0]

        self.vis_one_image(imgs, boxes, scores, first_cate, second_cate, fcv, scv, save_path, show_second_cate,
                           sec_thresh)

    def infer_for_eval(self, imgs, positive_threshold):
        try:
            assert self.trainable == False
        except:
            raise ValueError("Please change trainable model to 'False'")
        try:
            if len(imgs.shape) == 3:
                imgs = np.expand_dims(imgs, axis=0)
            assert imgs.shape[0] == 1
        except:
            raise ValueError("Evaluator do not support batch size more than 1!")

        imgs = imgs[..., ::-1]
        inference_info = self.sess.run(
            self.inference_info,
            feed_dict={
                self.input_tensor: imgs,
                self.oc_threshold: positive_threshold
            }
        )
        boxes, scores, first_cate, _, _, _ = inference_info

        #self.vis_one_image(imgs[0], boxes, scores, first_cate, np.array(first_cate), "test.jpg", False)

        return boxes, scores, first_cate

    def vis_one_image(self, imgs, boxes, scores, first_cate, second_cate, fcv, scv, save_file, show_second_cate,
                      second_cate_thresh):
        labels = []
        if second_cate_thresh is None:
            second_cate_thresh = 1
        first_cate = first_cate + 1
        second_cate = second_cate + 1
        for i, cate in enumerate(first_cate):
            score = str(round(scores[i], 2))
            cate_name = self.dataset.get_name(first_cate=cate)
            if cate == 1 and show_second_cate and (fcv[i] * scv[i] >= second_cate_thresh) and second_cate[i] != 1:
                cate2_name = self.dataset.get_name(second_cate=second_cate[i])
                cate2_score = str(round(scv[i], 2))
                label = cate_name + ":" + score + " " + cate2_name + ":" + cate2_score
                labels.append(label)
            else:
                label = ":".join([cate_name, score])
                labels.append(label)
        vis_img = vis.instance_visualizer(imgs, boxes, labels)
        vis_img.save(save_file)


def get_batch_size(input_tensor):
    return np.array(input_tensor.shape[0], dtype=np.float32)





