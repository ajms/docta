import sys
import os

from docta.datasets.image_classification import (
    ImageDataset10Classes,
)

o_path = os.getcwd()
sys.path.append(o_path)  # set path so that modules from other foloders can be loaded

import torch
from docta.utils.config import Config
from docta.datasets.data_utils import load_embedding

from docta.core.preprocess import Preprocess
from docta.core.report import Report

from docta.apis import DetectLabel
from docta.apis import Diagnose
import h5py

cfg = Config.fromfile("./config/image_classification_data.py")
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocess the dataset, get embeddings
data_path = (
    lambda x: cfg.save_path
    + f"embedded_{cfg.dataset_type}_{cfg.embedding_model.split('/')[-1]}_{x}.pt"
)
dataset = ImageDataset10Classes(cfg, train=True)
if not os.path.exists(data_path(0)):
    test_dataset = None
    pre_processor = Preprocess(cfg, dataset, test_dataset)
    pre_processor.encode_feature()
    print(pre_processor.save_ckpt_idx)
    save_ckpt_idx = pre_processor.save_ckpt_idx
else:
    save_ckpt_idx = list(
        range(
            min(
                len(dataset) // cfg.embedding_cfg.batch_size,
                cfg.embedding_cfg.save_num,
            )
        )
    )


dataset, _ = load_embedding(
    save_ckpt_idx, data_path, duplicate=False
)  # change it to duplicate=true may improve the performance

# initialize report
report = Report()

# diagnose labels
estimator = Diagnose(cfg, dataset, report=report)
estimator.hoc()

# print diagnose reports
import numpy as np

np.set_printoptions(precision=1, suppress=True)
T = report.diagnose["T"]
p = report.diagnose["p_clean"]
print(f"T_est is \n{T * 100}")
print(f"p_est is \n{p * 100}")

# label error detection
detector = DetectLabel(cfg, dataset, report=report)
detector.detect()

# Create a numpy array
label_error = np.array(report.detection["label_error"])
label_curation = np.array(report.curation["label_curation"])


# Save the array as torch tensors
torch.save(
    label_error,
    os.path.join(cfg.save_path, f"{cfg.embedding_model.split('/')[-1]}_label_error.pt"),
)
torch.save(
    label_curation,
    os.path.join(
        cfg.save_path, f"{cfg.embedding_model.split('/')[-1]}_label_curation.pt"
    ),
)

# print results
dataset_raw = ImageDataset10Classes(cfg, train=True)
index_to_class = {v: k for k, v in dataset_raw.class_to_idx.items()}
label_name = [index_to_class[idx] for idx in range(len(index_to_class))]
