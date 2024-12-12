# Code for the baseline experiments on **Conflab-dataset**

# Prerequisites

- PyTorch 1.7+
- detectron2

Install the required packages by:
```
pip install requirements.txt
```

or use uv. The packages were added to lock file like so
```bash
uv add scikit-learn seaborn parse rich tqdm opencv-python detectron2 torch==1.10.1+cpu torchvision==0.11.2+cpu \
--prerelease allow \
-f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html \
-f https://download.pytorch.org/whl/cpu/torch_stable.html
```
See the respective websites for the GPU with CUDA packages.

# How to Run

## Dataset

create dataset by:
```
python data_create.py create_coco=true force_register=true
```

## Training

```
bash scripts/script_train.sh -b "R50_FPN" -m train
```

## Evaluation

```
bash scripts/script_train.sh -b "R50_FPN" -m test
```
