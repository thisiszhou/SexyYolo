import sys
sys.path.append('.')
from utils.fileio import update_one_folder, json_save
import os
import argparse


join = os.path.join


def get_parser():
    parser = argparse.ArgumentParser(description="Yolov3 training configs")
    parser.add_argument(
        "data_file",
        metavar="FILE",
        help="path to dataset file",
    )
    parser.add_argument(
        "--update-type",
        default="all",
        metavar="FILE",
        help="update type: 'train' or 'val' or 'all'",
    )

    return parser


if __name__=="__main__":
    args = get_parser().parse_args()
    data_file = args.data_file
    update_data_type = args.update_type
    first_folder = [folder for folder in os.listdir(data_file) if folder[0] != "."]
    try:
        if update_data_type == "train":
            assert 'train' in set(first_folder)
        elif update_data_type == "all":
            assert 'train' in set(first_folder) and "val" in set(first_folder)
    except AssertionError:
        raise OSError("no train or val folder!")
    if not os.path.exists(join(data_file, "annotations")):
        os.mkdir(join(data_file, "annotations"))

    # update train
    if update_data_type == "train" or update_data_type == "all":
        save_json = update_one_folder(join(data_file, "train"), "train", 1)
        json_save(join(data_file, "annotations", "train.json"), save_json)
    # update val
    if update_data_type == "val" or update_data_type == "all":
        save_json = update_one_folder(join(data_file, "val"), "val", 1)
        json_save(join(data_file, "annotations", "val.json"), save_json)

