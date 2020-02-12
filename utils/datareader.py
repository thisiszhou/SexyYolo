# inner cate start from 1
import numpy as np
from collections import defaultdict
from utils.fileio import json_load, json_save
from utils.common import polygon2mask2, Bidict
import random
import cv2
import os


class DataSet(object):

    def __init__(self,
                 coco_annotation_file=None,
                 coco_photo_folder=None,
                 joint_annotation_file=None,
                 joint_photo_folder=None):
        self.coco_annotation_file = coco_annotation_file
        self.coco_photo_folder = coco_photo_folder

        # categories_dict: inner categories to real categories
        # inner cate start from 1
        self.coco_categories_dict = Bidict()

        if joint_annotation_file is not None:
            self.joint_annotation_file = joint_annotation_file
            self.joint_photo_folder = joint_photo_folder
            # categories_dict: inner categories to real categories
            # inner cate start from 1
            self.joint_categories_dict = Bidict()
        else:
            self.joint_annotation_file = None
            self.joint_photo_folder = None
        self.joint_classes = 0
        self.coco_epoch = None
        self.img_ids = None
    def imread(self, img_path, resize_shape):
        # default_shape: w, h
        # scale: w_scale, h_scale
        img_o = cv2.imread(img_path)
        shape = img_o.shape
        img = cv2.resize(img_o, (resize_shape, resize_shape))
        scale = shape[1] / resize_shape, shape[0] / resize_shape
        return img_o, np.expand_dims(img, axis=0), scale

    def create_index(self, first_random=False):
        self.create_coco_index(first_random)
        if self.joint_annotation_file is not None:
            self.create_joint_index()

    def create_coco_index(self, first_random):
        # coco_categories_dict: self category id --> coco id
        print("creating coco index...")
        crowd_num = 0
        json_data = json_load(self.coco_annotation_file)
        self.image_info = defaultdict(dict)
        self.annotation = defaultdict(dict)
        self.categories_name = dict()
        self.categories = json_data["categories"]
        self.img_ids = set()
        # load categories info
        for i, cate_dict in enumerate(json_data["categories"]):
            self.coco_categories_dict[i+1] = cate_dict["id"]
            self.categories_name[i+1] = cate_dict["name"]
        # load images info
        for image in json_data["images"]:
            img_id = image['id']
            self.image_info[img_id]["file_name"] = image["file_name"]
            self.image_info[img_id]["size"] = (image["height"], image["width"])
            self.image_info[img_id]["anno_id"] = []
        # load annotations
        for anno in json_data["annotations"]:
            if anno['iscrowd'] == 1:
                crowd_num += 1
                #continue
            anno_id = anno["id"]
            map_img_id = anno["image_id"]
            self.img_ids.add(map_img_id)
            self.image_info[map_img_id]["anno_id"].append(anno_id)
            self.annotation[anno_id]["polygon"] = anno["segmentation"]
            self.annotation[anno_id]["bbox"] = anno["bbox"]
            self.annotation[anno_id]["category_id"] = anno["category_id"]

        self.img_ids = list(self.img_ids)
        print("find {} crowd things".format(crowd_num))
        print("index created!")
        self.num_current_image = 0
        self.num_total_image = len(self.img_ids)
        self.coco_classes = len(self.coco_categories_dict)

        if first_random:
            random.shuffle(self.img_ids)

    def get_coco_epoch(self, batch_size, img_size=416):
        if_last_batch = False
        if self.coco_epoch is None:
            try:
                assert self.img_ids is not None
            except:
                raise ValueError("Please create coco index first!")
            self.coco_epoch = list(self.img_ids)
            self.coco_epoch_current = 0
        batch_ids = self.coco_epoch[self.coco_epoch_current: self.coco_epoch_current + batch_size]
        if self.coco_epoch_current + batch_size >= len(self.coco_epoch):
            self.coco_epoch = None
            if_last_batch = True
        else:
            self.coco_epoch_current = self.coco_epoch_current + batch_size

        scales = []
        img_list = []
        for img_id in batch_ids:
            img, _, scaling = self.get_img_anno(img_id, img_size, mask_on=False)
            img = np.expand_dims(img, axis=0)
            scales.append(scaling)
            img_list.append(img)
        imgs = np.vstack(img_list)
        scales = np.array(scales, dtype=np.float32)
        return imgs, scales, batch_ids, if_last_batch

    def load_categories_name(self, load_file):
        self.categories_name = dict()
        self.joint_cate_name = dict()
        if load_file is None:
            for i in range(100):
                self.categories_name[i] = str(i)
                self.joint_cate_name[i] = str(i)
            print("category name file doesn't find!")
            return

        json_dict = json_load(load_file)

        for key in json_dict.keys():
            key = int(key)
            if key / 1000 >= 1:
                self.joint_cate_name[int(key / 1000)] = json_dict[str(key)]
            else:
                self.categories_name[key] = json_dict[str(key)]
        print("load category names finished!")

    def get_name(self, first_cate=None, second_cate=None):
        if first_cate is None:
            return self.joint_cate_name[second_cate]
        else:
            return self.categories_name[first_cate]

    def save_categories_name(self, save_file="data/categories_name.json"):
        save_dict = dict()
        for cate in self.categories_name.keys():
            save_dict[cate] = self.categories_name[cate]
        for cate in self.joint_cate_name.keys():
            save_dict[1000 * cate] = self.joint_cate_name[cate]
        json_save(save_file, save_dict)

    def get_classes(self):
        return self.coco_classes, self.joint_classes

    def create_joint_index(self, seed=32):
        print("creating joint index!")
        self.joint_img_ids = set()
        self.joint_image_info = defaultdict(dict)
        # real to real
        self.fir2sec_cate = Bidict()
        self.joint_cate_name = dict()
        json_data = json_load(self.joint_annotation_file)
        for i, cate_info in enumerate(json_data["categories"]):
            self.joint_cate_name[i+1] = cate_info["name"]
            self.joint_categories_dict[i+1] = cate_info['id']
            super_id = cate_info['supercategory']
            self.fir2sec_cate[super_id] = cate_info['id']

        img_id = 0
        for img_info in json_data["images"]:
            self.joint_img_ids.add(img_id)
            self.joint_image_info[img_id]["file_name"] = img_info["file_name"]
            real_cate = img_info["category_id"]
            self.joint_image_info[img_id]["second_category"] = self.joint_categories_dict.value2key(real_cate)
            real_fisrt_cate = self.fir2sec_cate.value2key(real_cate)
            self.joint_image_info[img_id]["first_category"] = self.coco_categories_dict.value2key(real_fisrt_cate)
            img_id += 1
        print("index created!")
        self.num_joint_current_image = 0
        self.num_joint_total_image = len(self.joint_img_ids)
        self.joint_classes = len(self.joint_categories_dict)
        self.joint_img_ids = list(self.joint_img_ids)
        random.seed(seed)
        random.shuffle(self.joint_img_ids)


    def get_batch(self, batch_size, joint_batch_size=0, image_size=480, mask_on=False, seed=None, certain_id=None):
        # return: , boxes, masks, obj_nums
        # batch_imgs shape: [batch_size, image_size, image_size, 3]
        # boxes shape: [num_of_object, 5]; 5: x, y, w, h, categoty
        # masks shape: [num_of_object, image_size, image_size]
        # obj_nums: [batch_size, ]: object_num_of_pic1, object_num_of_pic2, object_num_of_pic3 ...
        # generate batch id
        batch_ids = []
        joint_batch_ids = []
        if certain_id is not None:
            if isinstance(certain_id, list):
                if isinstance(certain_id[0], list):
                    batch_ids = certain_id[0]
                    joint_batch_ids = certain_id[1]
                else:
                    batch_ids = certain_id
            else:
                batch_ids = [certain_id]
        else:
            img_ids = self.img_ids
            if (self.num_current_image + batch_size) > self.num_total_image:
                batch_ids.extend(img_ids[self.num_current_image:])
                if seed is None:
                    random.shuffle(img_ids)
                else:
                    random.seed(seed)
                    random.shuffle(img_ids)
                self.num_current_image = (self.num_current_image + batch_size) % self.num_total_image
                batch_ids.extend(img_ids[:self.num_current_image])
            else:
                batch_ids.extend(img_ids[self.num_current_image: self.num_current_image + batch_size])
                self.num_current_image += batch_size


            if joint_batch_size > 0:

                joint_ids = self.joint_img_ids
                if (self.num_joint_current_image + joint_batch_size) > self.num_joint_total_image:
                    joint_batch_ids.extend(joint_ids[self.num_joint_current_image:])
                    if seed is None:
                        random.shuffle(joint_ids)
                    else:
                        random.seed(seed)
                        random.shuffle(joint_ids)
                    self.num_joint_current_image = (
                            (self.num_joint_current_image + joint_batch_size) % self.num_joint_total_image
                    )
                    joint_batch_ids.extend(joint_ids[:self.num_joint_current_image])
                else:
                    joint_batch_ids.extend(joint_ids[self.num_joint_current_image:
                                                     self.num_joint_current_image + joint_batch_size])
                    self.num_joint_current_image += joint_batch_size

        # load batch images
        batch_imgs = []
        batch_infos = []

        for img_id in batch_ids:
            img, annotation, _ = self.get_img_anno(img_id, image_size, mask_on)
            batch_imgs.append(img)
            batch_infos.append(annotation)

        boxes_cates, masks, obj_nums = self.encode_label(batch_infos, image_size, mask_on)
        if joint_batch_size > 0:
            batch_ids = [batch_ids, joint_batch_ids]
            for img_id in joint_batch_ids:
                img, cates, _ = self.get_joint_img_anno(img_id, image_size)
                batch_imgs.append(img)
                boxes_cates.append(cates)
                obj_nums.append(-1)

        return (
            np.array(batch_imgs, dtype=np.float32),
            np.array(boxes_cates, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(obj_nums, dtype=np.float32),
            batch_ids
        )

    def encode_label(self, batch_info, image_size, mask_on):
        boxes = []
        masks = []
        num_index = []
        for annos in batch_info:
            # per image
            num_index.append(len(annos))
            for anno in annos:
                # per object
                boxes.append(anno[0])
                if mask_on:
                    mask = polygon2mask2([image_size, image_size], anno[1])
                    masks.append(mask)

        return boxes, masks, num_index

    def get_joint_img_anno(self, img_id, img_size):
        first_cate = self.joint_image_info[img_id]["first_category"]
        sec_cate = self.joint_image_info[img_id]["second_category"]
        img_name = self.joint_image_info[img_id]["file_name"]
        img_folder = self.joint_cate_name[sec_cate]
        img_name = os.path.join(self.joint_photo_folder, img_folder, img_name)
        img = cv2.imread(img_name)
        img_shape = img.shape
        img = cv2.resize(img, (img_size, img_size))
        scaling = img_size / img_shape[1], img_size / img_shape[0]

        return img, [first_cate, sec_cate, 0, 0, 0], scaling

    def get_img_anno(self, img_id, img_size, mask_on):
        anno_ids = self.image_info[img_id]["anno_id"]
        assert len(anno_ids) > 0
        # image info
        img_name = self.image_info[img_id]["file_name"]
        img_name = os.path.join(self.coco_photo_folder, img_name)
        img_shape = self.image_info[img_id]["size"]

        img = cv2.imread(img_name)
        img = cv2.resize(img, (img_size, img_size))
        scaling = img_shape[1] / img_size, img_shape[0] / img_size
        # image annotations
        annotations = []
        for anno_id in anno_ids:
            anno = self.annotation[anno_id]
            x1, y1, w, h = anno["bbox"]
            x1, w = x1 / scaling[0], w / scaling[0]
            y1, h = y1 / scaling[1], h / scaling[1]
            coco_cate = anno["category_id"]
            local_cate = self.coco_categories_dict.value2key(coco_cate)
            polygon = []
            if mask_on:
                polygons = anno["polygon"]
                for poly in polygons:
                    poly = np.array(poly, dtype=np.float32).reshape((-1, 2))
                    poly[:, 0] = poly[:, 0] / scaling[0]
                    poly[:, 1] = poly[:, 1] / scaling[1]
                    poly = poly.reshape(-1)
                    polygon.append(poly)
            annotations.append([[x1, y1, w, h, local_cate], polygon])

        return img, annotations, scaling