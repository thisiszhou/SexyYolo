import sys
sys.path.append('.')
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
        "input_image",
        type=str,
        help="The path of the input image."
    )
    parser.add_argument(
        "--config-file",
        default="configs/test_local.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--result-folder",
        default="data/demo_result",
        metavar="FILE",
        help="path to detection result file",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_config(args.config_file)

    image_size = change_tuple(cfg['MODEL']['IMAGE_SIZE'])[0]
    coco_classes, joint_classes = change_tuple(cfg['MODEL']['CATEGORY_NUM'])
    load_save_model_file = cfg["MODEL"]["MODEL_WEIGHTS"]
    cate_name_file = cfg["INFER"]["CATEGORY_NAME_FILE"]
    yolo_model = YoloJoint(trainable=False, category=cate_name_file)
    yolo_model.build(image_size=image_size,
                     coco_classes=coco_classes,
                     joint_classes=joint_classes)
    yolo_model.init_sess(load_save_model_file)
    acti_thresh = cfg['INFER']['ACTIVATE_THRESH']
    ret_folder = None
    if os.path.isfile(args.result_folder):
        ret_folder = os.path.dirname(args.result_folder)
    else:
        ret_folder = args.result_folder
    if not os.path.exists(ret_folder):
        os.mkdir(ret_folder)
    if os.path.isfile(args.input_image):
        img_file = args.input_image
        img_name = os.path.basename(img_file)
        start_time = time.time()
        yolo_model.infer_simple(img_file,
                                acti_thresh,
                                os.path.join(ret_folder, img_name),
                                cfg['INFER']['SHOW_SECOND_CATE'],
                                image_size,
                                cfg['INFER']['SECOND_CATE_THRESH'])
        print("finished one img. spend time: ",
              round(time.time() - start_time, 4), "s.")
    elif os.path.isdir(args.input_image):
        folder = args.input_image
        photos_name = [file for file in os.listdir(folder) if file[0] != "."]
        for img_name in photos_name:
            start_time = time.time()
            yolo_model.infer_simple(os.path.join(folder, img_name),
                                    acti_thresh,
                                    os.path.join(ret_folder, img_name),
                                    cfg['INFER']['SHOW_SECOND_CATE'],
                                    image_size,
                                    cfg['INFER']['SECOND_CATE_THRESH'])
            print("finished img: ", img_name,
                  ". spend time: ",
                  round(time.time() - start_time, 4), "s.")
    else:
        raise ValueError("unknown input img info!")
    yolo_model.close_sess()
