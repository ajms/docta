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
data_path = lambda x: cfg.save_path + f"embedded_{cfg.dataset_type}_{x}.pt"
if not os.path.exists(data_path(0)):
    dataset = ImageDataset10Classes(cfg, train=True)
    test_dataset = None
    pre_processor = Preprocess(cfg, dataset, test_dataset)
    pre_processor.encode_feature()
    print(pre_processor.save_ckpt_idx)
    save_ckpt_idx = pre_processor.save_ckpt_idx
else:
    save_ckpt_idx = list(range(19))


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

# Save the array in HDF5 format
with h5py.File(os.path.join(cfg.save_path, "label_error.h5"), "w") as f:
    f.create_dataset("dataset_name", data=label_error, compression="gzip")

with h5py.File(os.path.join(cfg.save_path, "label_curation.h5"), "w") as f:
    f.create_dataset("dataset_name", data=label_curation, compression="gzip")

# print results
dataset_raw = ImageDataset10Classes(cfg, train=True)
index_to_class = {v: k for k, v in dataset_raw.class_to_idx.items()}
label_name = [index_to_class[idx] for idx in range(len(index_to_class))]
