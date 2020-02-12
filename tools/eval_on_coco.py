import sys
sys.path.append('.')
from utils.datareader import DataSet
from model.yolov3 import YoloJoint
from utils.fileio import json_save, json_load
from pycocotools.coco import COCO
from utils.fileio import load_config, change_tuple
from pycocotools.cocoeval import COCOeval
import os
import time
import argparse


oc_iou = 0.1


def get_parser():
    parser = argparse.ArgumentParser(description="Yolov3 training configs")
    parser.add_argument(
        "--config-file",
        default="configs/Yolov3_coco.yaml",
        metavar="FILE",
        help="path to config file",
    )
    #parser.add_argument("input", nargs="+", help="A list of space separated input images")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_config(args.config_file)

    detec_annotation_file = cfg['EVAL']['COCO_EVAL']['RESULT']['DETEC_RESULT_JSON']
    eval_folder = cfg['EVAL']['COCO_EVAL']['DATESET']['EVAL_PHOTO_FOLDER']
    eval_anno_file = cfg['EVAL']['COCO_EVAL']['DATESET']['EVAL_ANNOTATION_FILE']

    if os.path.exists(detec_annotation_file):
        os.remove(detec_annotation_file)

    

    eval_weights = cfg['MODEL']['MODEL_WEIGHTS']
    image_size = change_tuple(cfg['MODEL']['IMAGE_SIZE'])[0]
    coco_classes, joint_classes = change_tuple(cfg['MODEL']['CATEGORY_NUM'])
    # load data
    cocodata = DataSet(eval_anno_file, eval_folder)
    cocodata.create_index()

    # create model
    yolo_model = YoloJoint(trainable=False)
    yolo_model.build(image_size=image_size, coco_classes=coco_classes, joint_classes=joint_classes)
    yolo_model.init_sess(eval_weights)

    cate_dict = cocodata.coco_categories_dict
    if_last = False
    dt = dict()
    dt["annotations"] = []
    data_length = len(cocodata.img_ids)
    num = 0
    anno_id = 0
    start_time = time.time()
    while not if_last:
        if (num + 1) % 50 == 0:
            spend_time = time.time() - start_time
            start_time = time.time()
            print("precessing: ", str(num), "/", str(data_length), " spend time:", round(spend_time, 3))
        imgs, scales, img_id, if_last = cocodata.get_coco_epoch(1)
        scale = scales[0]
        img_id = img_id[0]
        boxes, scores, first_cate = yolo_model.infer_for_eval(imgs, oc_iou)
        for i, box in enumerate(boxes):
            new_anno = dict()
            box = [box[0] * scale[0], box[1] * scale[1],
                   (box[2] - box[0]) * scale[0], (box[3] - box[1]) * scale[1]]
            new_anno['image_id'] = int(img_id)
            new_anno['bbox'] = [float(x) for x in box]
            new_anno['category_id'] = int(cate_dict[first_cate[i] + 1])
            new_anno['score'] = float(scores[i])
            new_anno['area'] = float(box[2]*box[3])
            new_anno['id'] = anno_id
            dt["annotations"].append(new_anno)
            anno_id += 1
        num += 1
    json_save(detec_annotation_file, dt)

    gt = COCO(eval_anno_file)
    dt = COCO(detec_annotation_file)
    cocoeval = COCOeval(gt, dt, iouType="bbox")
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()