# Code for the baseline experiments on **Conflab-dataset**

# Prerequisites

- PyTorch 1.7+
- detectron2

Install the required packages by:
```
pip install requirements.txt
```

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
