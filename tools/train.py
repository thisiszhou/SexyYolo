import sys
sys.path.append('.')
from utils.datareader import DataSet
from model.yolov3 import YoloJoint
from utils.common import learning_rate_decay, show_loss
from utils.fileio import load_config, change_tuple
import time
import shutil
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Yolov3 training configs")
    parser.add_argument(
        "--config-file",
        default="configs/Yolov3_coco.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_config(args.config_file)

    # load dataset
    coco_anno = cfg['TRAIN']['DETECTION_TRAIN']['DATESET']['TRAIN_ANNOTATION_FILE']
    coco_pic = cfg['TRAIN']['DETECTION_TRAIN']['DATESET']['TRAIN_PHOTO_FOLDER']
    joint_anno = None
    joint_pic = None
    classify_thresh = 1
    coco_classes, joint_classes = change_tuple(cfg['MODEL']['CATEGORY_NUM'])
    if cfg['TRAIN']['CLASSIFY_TRAIN']['JOINT_TRAIN_ON']:
        try:
            assert joint_classes > 0
        except AssertionError:
            raise AssertionError("joint_classes can't be 0 if using joint training!")
        joint_anno = cfg['TRAIN']['CLASSIFY_TRAIN']['DATESET']['TRAIN_ANNOTATION_FILE']
        joint_pic = cfg['TRAIN']['CLASSIFY_TRAIN']['DATESET']['TRAIN_PHOTO_FOLDER']
        classify_thresh = cfg['TRAIN']['CLASSIFY_TRAIN']['TRAIN_PARA']['CLASSIFY_THRESH']
    dataset = DataSet(coco_anno, coco_pic, joint_anno, joint_pic)
    dataset.create_index(first_random=True)
    yolo_model = YoloJoint(trainable=True)
    image_size = change_tuple(cfg['MODEL']['IMAGE_SIZE'])[0]
    print("image_size:", image_size)
    cfg_train = cfg['TRAIN']['DETECTION_TRAIN']['TRAIN_PARA']
    batch_size = cfg_train['BATCH_SIZE']

    yolo_model.build(image_size=image_size,
                     coco_classes=coco_classes,
                     joint_classes=joint_classes,
                     classes_weights=cfg_train['CLASSES_WEIGHTS'],
                     boxes_weights=cfg_train['BOXES_WEIGHTS'],
                     oc_weights=cfg_train['OBJECT_CONF_WEIGHTS'],
                     nooc_weights=cfg_train['NOOC_WEIGHTS'],
                     ignore_thresh=cfg_train['NOOC_IGNORE_THRESH'],
                     classify_threshold=classify_thresh)
    load_save_model_file = cfg["TRAIN"]["LOAD_TRAIN_MODEL"]
    pre_train = False
    if len(load_save_model_file) == 0:
        load_save_model_file = cfg["TRAIN"]["PRE_TRAIN_MODEL"]
        pre_train = True
    if pre_train:
        yolo_model.init_sess(pre_weights=load_save_model_file)
    else:
        yolo_model.init_sess(load_save_model_file)

    lr = float(cfg_train["START_LEANING_RATE"])
    min_lr = float(cfg_train["MIN_LEANING_RATE"])
    lr_decay = cfg_train["LR_DECAY"]
    lr_decay_step = cfg_train["LR_DECAY_STEP"]
    train_total_step = cfg_train["TRAIN_STEP"]
    show_loss_step = cfg_train["SHOW_LOSS_STEP"]
    coco_save_model_file = cfg_train["MODEL_SAVE_FILE"]
    save_step = cfg_train["WEIGHTS_SAVE_STEP"]
    avg_loss = None
    start_time = time.time()
    for step in range(train_total_step):
        imgs, boxes, _, obj_nums, batch_ids = dataset.get_batch(batch_size=batch_size, image_size=image_size)
        loss = yolo_model.train_one_batch(imgs, boxes, obj_nums, lr)
        if avg_loss is None:
            avg_loss = loss[-1]
        else:
            avg_loss = 0.9 * avg_loss + 0.1 * loss[-1]

        if step % show_loss_step == 0:
            spend_time = round(time.time() - start_time, 2)
            start_time = time.time()
            show_loss("coco train step: "+str(step), loss, avg_loss, spend_time)

        if (step + 1) % lr_decay_step == 0:
            lr = learning_rate_decay(lr, lr_decay, min_lr)

        if (step + 1) % save_step == 0:
            yolo_model.save_model_weights(coco_save_model_file, global_step=step + 1)
    print("finished coco train!")
    if cfg['TRAIN']['CLASSIFY_TRAIN']['JOINT_TRAIN_ON']:
        # save coco weights
        if cfg['TRAIN']['DETECTION_TRAIN']['TRAIN_PARA']['TRAIN_STEP'] > 0:
            coco_model = os.path.dirname(coco_save_model_file)
            if os.path.exists(coco_model):
                shutil.move(coco_model, coco_model+"_train")

        cfg_train = cfg['TRAIN']['CLASSIFY_TRAIN']['TRAIN_PARA']
        batch_size, joint_batch_size = change_tuple(cfg_train["BATCH_SIZE"])
        joint_save_model_file = cfg_train["MODEL_SAVE_FILE"]
        r = float(cfg_train["START_LEANING_RATE"])
        min_lr = float(cfg_train["MIN_LEANING_RATE"])
        lr_decay = cfg_train["LR_DECAY"]
        lr_decay_step = cfg_train["LR_DECAY_STEP"]
        train_total_step = cfg_train["TRAIN_STEP"]
        show_loss_step = cfg_train["SHOW_LOSS_STEP"]
        save_step = cfg_train["WEIGHTS_SAVE_STEP"]
        avg_loss = None
        start_time = time.time()
        for step in range(train_total_step):
            imgs, boxes, _, obj_nums, batch_ids = dataset.get_batch(batch_size=batch_size,
                                                                    joint_batch_size=joint_batch_size,
                                                                    image_size=image_size)
            loss = yolo_model.train_one_batch(imgs, boxes, obj_nums, lr)
            if avg_loss is None:
                avg_loss = loss[-1]
            else:
                avg_loss = 0.9 * avg_loss + 0.1 * loss[-1]

            if step % show_loss_step == 0:
                spend_time = round(time.time() - start_time, 2)
                start_time = time.time()
                show_loss("joint train step: "+str(step), loss, avg_loss, spend_time)

            if (step + 1) % lr_decay_step == 0:
                lr = learning_rate_decay(lr, lr_decay, min_lr)

            if (step + 1) % save_step == 0:
                yolo_model.save_model_weights(joint_save_model_file, global_step=step + 1)
        print("finished joint train!")
    yolo_model.close_sess()
