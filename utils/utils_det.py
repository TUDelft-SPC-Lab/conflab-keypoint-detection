from datetime import datetime
import logging
import os
from typing import List, Optional
from detectron2.utils.env import seed_all_rng
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T
from omegaconf.dictconfig import DictConfig
import torch
from tqdm import tqdm
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import rich


def configure_logger(args, fileonly=False):
    now = datetime.now()
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(
        args.log_dir,
        f"{args.name}_{args.log_prefix}{now.month:02d}_{now.day:02d}_{now.hour:03d}.txt"
    )

    name = "detectron2"
    rank = comm.get_rank()

    if not fileonly:
        logger = setup_logger(output=logfile, distributed_rank=rank, name=name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if rank == 0:
            fh = logging.FileHandler(logfile, mode='w')
            fh.setLevel(logging.DEBUG)
            plain_formatter = logging.Formatter(
                "%(asctime)s %(name)s | %(message)s")
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)
            rich.print(f"[red]Log to {logfile}[/red]")

    return logger


def visualize_det2(dataset_dicts: List[dict],
                   predictor: DefaultPredictor,
                   metadata: Optional[Metadata] = None,
                   vis_conf: DictConfig = None) -> None:
    rng = np.random.RandomState(seed=vis_conf.seed)
    samples = rng.choice(dataset_dicts, size=vis_conf.count)
    for i, d in tqdm(enumerate(samples)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        # filter only `person` class
        instances = outputs["instances"]
        instances = instances[instances.pred_classes == 0]

        v = Visualizer(im[:, :, ::-1], metadata=metadata)
        out = v.draw_instance_predictions(instances.to("cpu"))
        if vis_conf.save:
            os.makedirs(vis_conf.save_folder, exist_ok=True)
            filename = os.path.join(vis_conf.save_folder, f"{i:06d}.jpg")
            cv2.imwrite(filename, out.get_image()[:, :, ::-1])
        else:
            cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def create_train_augmentation(cfg):
    augs = [
        T.Resize((cfg.image_h, cfg.image_w)),
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.3),
    ]
    if cfg.half_crop:
        augs.insert(0, MyCustomCrop())
    return augs


def create_test_augmentation(cfg):
    augs = [
        T.ResizeShortestEdge(min(cfg.image_h_test, cfg.image_w_test)),
    ]
    return augs


class MyCustomCrop(T.Augmentation):
    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = h, int(w // 2)
        x0 = cropw
        y0 = 0
        return T.CropTransform(x0, y0, cropw, croph)
