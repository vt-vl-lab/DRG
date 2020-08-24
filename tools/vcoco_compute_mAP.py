# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import json

import numpy as np
from maskrcnn_benchmark.data.datasets.evaluation.vcoco.vsrl_eval import VCOCOeval
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

def main():
    parser = argparse.ArgumentParser(description="PyTorch DRG Detection Inference")
    parser.add_argument(
        "--dataset_name",
        default="vcoco_test",
        help="dataset name, default: vcoco_test",
    )
    parser.add_argument(
        "--detection_file",
        help="The path to the final detection pkl file for test",
        default="../output/VCOCO/detection_merged_human_object_app.pkl",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    data = DatasetCatalog.get(dataset_name)
    data_args = data["args"]
    action_dic = json.load(open(data_args['action_index']))
    action_dic_inv = {y: x for x, y in action_dic.items()}
    vcocoeval = VCOCOeval(data_args['vcoco_test_file'], data_args['ann_file'], data_args['vcoco_test_ids_file'])

    vcocoeval._do_eval(args.detection_file, ovr_thresh=0.5)

if __name__ == "__main__":
    main()