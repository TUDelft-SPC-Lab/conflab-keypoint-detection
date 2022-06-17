import os
from typing import Dict, List

from utils.utils_det import configure_logger
from detectron2.data.catalog import MetadataCatalog
import hydra
from detectron2 import model_zoo
from omegaconf import OmegaConf, DictConfig

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import (DatasetCatalog, DatasetMapper,
                             build_detection_train_loader,
                             build_detection_test_loader)

from detectron2.engine import DefaultTrainer, launch, default_setup, DefaultPredictor
from detectron2.data import transforms as T

from data_loading import conflab_dataset
from utils import (utils_dist, visualize_det2, create_train_augmentation,
                   create_test_augmentation)
import rich
import logging

logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg,
                               is_train=True,
                               augmentations=create_train_augmentation(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg,
                               is_train=False,
                               augmentations=create_test_augmentation(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name,
                             cfg.TASKS,
                             False,
                             output_dir=output_folder,
                             kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                             use_fast_impl=False)


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aug = T.Resize((cfg.image_h, cfg.image_w))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    cfg.DATASETS.TRAIN = (args.train_dataset, )
    cfg.DATASETS.TEST = (args.test_dataset, )
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    cfg.image_w = args.size[0]
    cfg.image_h = args.size[1]
    cfg.image_w_test = args.size_test[0]
    cfg.image_h_test = args.size_test[1]
    cfg.half_crop = args.half_crop

    cfg.TASKS = tuple(args.eval_task)

    cfg.SOLVER.REFERENCE_WORLD_SIZE = args.ref_world_size
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.KEYPOINT_OKS_SIGMAS = ()

    if args.task_name == 'keypoint':
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = args.num_keypoints
        cfg.TEST.KEYPOINT_OKS_SIGMAS = [x / 1000 for x in args.oks_std]

    if args.eval_only is False:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_zoo)
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        cfg.SOLVER.BASE_LR = args.learning_rate
        cfg.SOLVER.MAX_ITER = args.max_iters
        cfg.SOLVER.WARMUP_ITERS = int(args.max_iters / 10)
        cfg.SOLVER.STEPS = (int(args.max_iters / 2),
                            int(args.max_iters * 2 / 3))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    else:
        if args.pretrained:
            cfg.MODEL.WEIGHTS = args.model_zoo_weights
        else:
            cfg.MODEL.WEIGHTS = os.path.join(
                cfg.OUTPUT_DIR,
                "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh  # set a custom testing threshold

    if args.checkpoint is not None:
        logger.debug(f"load checkpoint from {args.checkpoint}")
        cfg.MODEL.WEIGHTS = os.path.expanduser(args.checkpoint)

    if args.eval_only is False:
        default_setup(cfg, args)

    return cfg


def main(args: DictConfig):
    args = OmegaConf.create(OmegaConf.to_yaml(args, resolve=True))

    rich.print("Command Line Args:\n{}".format(
        OmegaConf.to_yaml(args, resolve=True)))

    if args.accelerator == "ddp" and args.ngpus > 1:
        utils_dist.init_distributed_mode(args)

    # register dataset
    conflab_dataset.register_conflab_dataset(args)

    if args.create_coco and args.force_register:
        # only create dataset
        return

    cfg = setup(args)

    print(f"Number of keypoints: {args.num_keypoints}")

    if args.eval_only is False:
        configure_logger(args, fileonly=True)

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()

    else:
        # setup logger
        configure_logger(args)

        if args.visualize is False:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            res = Trainer.test(cfg, model)
            logger.info(res)
            return res
        else:
            test_dataset: List[Dict] = DatasetCatalog.get(args.test_dataset)
            metadata = MetadataCatalog.get(args.test_dataset)
            predictor = Predictor(cfg)
            visualize_det2(test_dataset,
                           predictor,
                           metadata=metadata,
                           vis_conf=args.vis)


def main_spawn(args: DictConfig):
    # ddp spawn
    launch(main,
           args.ngpus,
           machine_rank=args.machine_rank,
           num_machines=args.num_machines,
           dist_url=args.dist_url,
           args=(args, ))


@hydra.main(config_name='config', config_path='conf')
def hydra_main(args: DictConfig):
    if args.launcher_name == "local":
        if args.accelerator == "ddp":
            main(args)
        else:
            args.dist_url = "auto"
            main_spawn(args)
    elif args.launcher_name == "slurm":
        from utils.utils_slurm import submitit_main
        submitit_main(args)


if __name__ == "__main__":
    hydra_main()