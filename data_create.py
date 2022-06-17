from typing import Dict, List
import os
from utils.utils_det import configure_logger
import hydra
import cv2

import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_loading.conflab_dataset import *
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer
from rich.progress import track


@hydra.main(config_name='config', config_path='conf')
def main(args):
    configure_logger(args)
    register_conflab_dataset(args)

    if args.data_plot:
        os.makedirs(args.data_plot_dir, exist_ok=True)
        dataset_dicts: List[Dict] = DatasetCatalog.get(args.test_dataset)
        metadata = MetadataCatalog.get(args.test_dataset)
        samples = random.sample(
            dataset_dicts, min(args.data_create_num_vis, len(dataset_dicts)))
        for d in track(samples):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1],
                                    metadata=metadata,
                                    scale=1.2)
            out = visualizer.draw_dataset_dict(d)
            cv2_im = out.get_image()[:, :, ::-1]

            fname = d['file_name']
            out_filename = '_'.join(fname.split(os.sep)[-3:])
            cv2.imwrite(os.path.join(args.data_plot_dir, out_filename), cv2_im)

            # cv2.imshow(d["file_name"], cv2_im)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
