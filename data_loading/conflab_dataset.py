from ctypes import util
import random
from typing import Dict, List, Optional
from PIL import Image
import os
import json
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from data_loading.utils import *
from pathlib import Path
import logging
import json
from data_loading.utils import KP_TO_OKS
import pickle

from utils import utils_dist

random.seed(0)

logger = logging.getLogger("detectron2")

# EXCLUDE_ANN = [
#     'cam6/vid3-seg4-scaled-denoised', 'cam6/vid3-seg5-scaled-denoised',
#     'cam2/vid3-seg6-scaled-denoised', 'cam4/vid3-seg1-scaled-denoised',
#     'cam4/vid3-seg2-scaled-denoised', 'cam4/vid3-seg3-scaled-denoised',
#     'cam8/vid3-seg1-scaled-denoised', 'cam8/vid3-seg2-scaled-denoised',
#     'cam8/vid3-seg4-scaled-denoised'
# ]

EXCLUDE_ANN = ['cam8/vid3-seg2-scaled-denoised']


def convert_conflab_to_coco(
        img_root_dir: str,
        annotation_dir: str,
        total_ann: Optional[int] = None,
        thresh_null: float = 0.1,
        info_path: Optional[str] = None,
        valid_kv_ratio: Optional[float] = 0.05) -> List[Dict]:

    counter_image = 0
    counter = 0
    coco_data = {"info": {}, "images": [], "annotations": [], "categories": []}

    ann_stat = AnnStat()

    dict_ims = {}  # all_images that have been seen so far

    for ann_file in tqdm(sorted(os.listdir(annotation_dir)),
                         position=0,
                         desc="processing video"):
        parsed_info = extract_file_info(ann_file)

        img_ann_dir = f"cam{parsed_info['cam']}/vid{parsed_info['vid']}-seg{parsed_info['seg']}-scaled-denoised"
        if img_ann_dir in EXCLUDE_ANN:
            logger.info(f"skipping {img_ann_dir}")
            continue
        img_dir = os.path.join(img_root_dir, img_ann_dir)
        if not os.path.exists(img_dir):
            logger.warning(f"Directory {img_dir} does not exist")
            continue

        with open(os.path.join(annotation_dir, ann_file), 'r') as fp:
            full_data = json.load(fp)

        coco_data["info"] = full_data["info"]
        coco_data["categories"] = full_data["categories"]

        data_kp = full_data['annotations']['skeletons']

        data_keys = list(data_kp)
        if total_ann is not None:
            data_keys = random.sample(data_keys, min(total_ann,
                                                     len(data_keys)))
        for _, v in tqdm(enumerate(data_keys), position=3):
            # v contain info for each image
            annotations_for_image = list(v.items())
            filename = os.path.join(
                img_dir,
                f"{annotations_for_image[0][1]['image_id']+1:06d}.jpg")
            if not os.path.exists(filename):
                logger.warning(f"{filename} does not exist")
                continue

            width, height = Image.open(filename).size

            if filename not in dict_ims:
                counter_image += 1

                record_im = dict()
                record_im["file_name"] = os.path.join(
                    img_ann_dir,
                    f"{annotations_for_image[0][1]['image_id']+1:06d}.jpg")
                record_im["id"] = counter_image
                record_im["height"] = height
                record_im["width"] = width
                dict_ims[filename] = record_im

            coco_ann_im = []

            ann_per_file = 0
            for _, anno in annotations_for_image:
                ann_stat.update_ann(filename)
                counter += 1

                record_ann = {}
                record_ann["id"] = counter
                record_ann["image_id"] = dict_ims[filename]["id"]
                record_ann["category_id"] = 1  # NOTE: person category

                null_values = [x is None for x in anno["keypoints"]]
                ann_stat.update_pt(len(anno["keypoints"]), sum(null_values),
                                   filename)

                keypoints, bbox = filter_kp_xy(anno["keypoints"],
                                               width,
                                               height,
                                               valid_kv_ratio=valid_kv_ratio)
                if keypoints is None:
                    continue

                record_ann["bbox"] = bbox
                record_ann["segmentation"] = []
                record_ann["keypoints"] = keypoints
                record_ann["num_keypoints"] = len(keypoints[2::3])
                record_ann["area"] = int(bbox[2] * bbox[3])
                record_ann["iscrowd"] = 0

                coco_ann_im.append(record_ann)

                ann_stat.update_nonnull_person(filename)

            if ann_stat.null_kp(filename) < thresh_null:
                coco_data["annotations"].extend(coco_ann_im)
                coco_data["images"].append(record_im)
                ann_per_file += 1
            else:
                # NOTE: image has too many null values
                ann_stat.remove_filename(filename)
                # logger.warning(f"[REJECTED] {filename}")
            # ann_stat.stats_file(filename)
            # if total_ann is not None and ann_per_file > total_ann:
            #     break
        ann_stat.stats()

    ann_stat.stats()

    with open(info_path, "wb") as fp:
        pickle.dump(dict(ann_stat.info_im), fp)

    return coco_data


def register_conflab_dataset(args: DictConfig) -> None:

    if args.split_path:
        with open(args.split_path, 'r') as fp:
            split_info = json.load(fp)
        args.train_cam = split_info['train_cam']
        args.test_cam = split_info['test_cam']

    keypoints, keypoint_connection_rules, keypoint_flip_map, kp_indices = get_keypoints(
        args.kp_rank)

    args.oks_std = [int(KP_TO_OKS[kp] * 1000) for kp in keypoints]

    args.num_keypoints = len(keypoints)

    if utils_dist.is_main_process():
        if args.create_coco:
            convert_to_coco = (not os.path.exists(
                args.coco_json_path)) or args.force_register
            # convert to coco
            if convert_to_coco:
                coco_info = convert_conflab_to_coco(
                    img_root_dir=args.img_root_dir,
                    annotation_dir=args.ann_dir,
                    total_ann=args.total_im_per_vid,
                    thresh_null=args.thresh_null_kp,
                    info_path=args.info_path,
                    valid_kv_ratio=args.valid_kv_ratio)

                Path(args.coco_json_path).parent.mkdir(exist_ok=True,
                                                       parents=True)
                with open(args.coco_json_path, "w") as fp:
                    json.dump(coco_info, fp, indent=2)

            logger.info("splitting coco dataset")
            coco_split(args.coco_json_path,
                       args.coco_json_path_train,
                       args.coco_json_path_test,
                       test_cam=args.test_cam,
                       train_cam=args.train_cam,
                       filter_kp_ids=kp_indices,
                       total_im_per_vid=args.total_im_per_vid)

    def _register(dataset, ann_path):
        register_coco_instances(dataset, {}, ann_path, args.img_root_dir)
        # set meta data catalog
        MetadataCatalog.get(dataset).keypoint_names = keypoints
        MetadataCatalog.get(
            dataset).keypoint_connection_rules = keypoint_connection_rules
        MetadataCatalog.get(dataset).keypoint_flip_map = keypoint_flip_map

    if utils_dist.is_dist_avail_and_initialized():
        # wait for all processes to finish
        torch.distributed.barrier()

    _register(args.dataset, args.coco_json_path)
    _register(args.train_dataset, args.coco_json_path_train)
    _register(args.test_dataset, args.coco_json_path_test)
