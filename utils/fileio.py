import json
import os
import shutil
import cv2
import numpy as np
import yaml
join = os.path.join


def json_load(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    print("load json: ", json_file, " succeed")
    return data


def json_save(result_json, json_dict):
    with open(result_json, 'w') as result_file:
        json.dump(json_dict, result_file)
    print("save json: ", result_json, " succeed")


def update_one_folder(photo_folder, train_type, supercategory):
    category_file = [file for file in os.listdir(photo_folder) if file[0] != "."]
    json_dict = dict()
    json_dict["categories"] = []
    json_dict["images"] = []
    for i, cate in enumerate(category_file):
        cate_dict = dict()
        cate_dict['supercategory'] = supercategory
        cate_dict['id'] = i + 1
        cate_dict['name'] = cate
        json_dict["categories"].append(cate_dict)
        update_one_cate(join(photo_folder, cate),
                        i + 1,
                        json_dict,
                        train_type)
    return json_dict


def update_one_cate(cate_folder, category_id, json_dict, train_type):
    # default: json_dict have no current label_id

    category = os.path.basename(cate_folder)
    print("start to update ", train_type + " :" + category)
    photos_name = [file for file in os.listdir(cate_folder) if file[0] != "."]
    if os.path.exists(cate_folder +"_tem"):
        shutil.rmtree(cate_folder +"_tem")
    os.mkdir(cate_folder +"_tem")
    duplicate_set = set([None])
    image_id = 0
    for i, filename in enumerate(photos_name):
        origin_pic = cv2.imread(join(cate_folder, filename))
        sum_ = np.sum(origin_pic)
        if sum_ in duplicate_set:
            # pass
            print("duplicate:", filename)
        else:
            duplicate_set.add(sum_)
            img_str_id = category_id * 100000000 + image_id
            new_filename = category + "_" + train_type + "_" + str(img_str_id) + ".jpg"
            shutil.move(join(cate_folder, filename), cate_folder +"_tem/" + new_filename)
            append_dict = dict()
            append_dict["file_name"] = new_filename
            append_dict["category_id"] = category_id
            append_dict["image_id"] = img_str_id
            json_dict["images"].append(append_dict)
            image_id += 1
    shutil.rmtree(cate_folder)
    shutil.move(cate_folder + "_tem/", cate_folder)
    print(category, " update finished!")

def load_config(cfg_file):
    f = open(cfg_file)
    cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg

def change_tuple(num):
    assert isinstance(num, str)
    num = num.lstrip("(").rstrip(")").split(",")
    num = [int(x) for x in num if x != ""]
    return num