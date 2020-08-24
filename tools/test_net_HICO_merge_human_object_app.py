# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation.hico.hico_compute_mAP import compute_hico_map

def run_test(
            app_detection,
            sp_human_detection,
            sp_object_detection,
            output_file=None,
):
    detection = {}
    for idx, image_id in enumerate(tqdm(sp_human_detection)):
        sp_detect_this_image_list = sp_human_detection[image_id]
        app_detect_this_image_list = app_detection[image_id]
        sp_object_detect_this_image_list = sp_object_detection[image_id]
        this_pair = []
        for sp_this_pair in sp_detect_this_image_list:
            for app_this_pair in app_detect_this_image_list:
                # human box and object box are the same
                if np.all(sp_this_pair[0] == app_this_pair[0]) and np.all(sp_this_pair[1] == app_this_pair[1]):
                    assert sp_this_pair[2][0] == app_this_pair[2][0] # object class is the same
                    sp_this_pair[3] = sp_this_pair[3] * app_this_pair[3]
                    break
            for object_this_pair in sp_object_detect_this_image_list:
                if np.all(sp_this_pair[0] == object_this_pair[0]) and np.all(sp_this_pair[1] == object_this_pair[1]):
                    assert sp_this_pair[2][0] == object_this_pair[2][0]  # object class is the same
                    sp_this_pair[3] = sp_this_pair[3] * object_this_pair[3] * sp_this_pair[5]
                    break

            this_pair.append(sp_this_pair)

        detection[image_id] = this_pair

    pickle.dump(detection, open(output_file, "wb"))


def main():
    #     apply_prior   prior_mask
    # 0        -             -
    # 1        Y             -
    # 2        -             Y
    # 3        Y             Y
    parser = argparse.ArgumentParser(description="PyTorch DRG Detection Inference")
    parser.add_argument(
        "--dataset_name",
        default="hico_test",
        help="hico_test",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--app_pkl",
        help="The path to the app checkpoint for test",
        default="/Users/jiarui/Downloads/hico_checkpoints/detection.pkl",
    )
    parser.add_argument(
        "--sp_human_pkl",
        help="The path to the sp checkpoint for test",
        default="/Users/jiarui/Downloads/hico_checkpoints/detection_sp.pkl",
    )
    parser.add_argument(
        "--sp_object_pkl",
        help="The path to the sp checkpoint for test",
        default="/Users/jiarui/Downloads/hico_checkpoints/detection_sp_object.pkl",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app_detection = pickle.load(open(args.app_pkl, "rb"), encoding='latin1')
    sp_human_detection = pickle.load(open(args.sp_human_pkl, "rb"), encoding='latin1')
    sp_object_detection = pickle.load(open(args.sp_object_pkl, "rb"), encoding='latin1')

    dataset_name = args.dataset_name
    output_folder = os.path.join(ROOT_DIR, "output/HICO/inference_human_object_app", dataset_name)
    mkdir(output_folder)

    output_file = os.path.join(output_folder, 'hico_detection_merged_human_object_app.pkl')
    output_map_folder = os.path.join(output_folder, 'map')

    run_test(
        app_detection,
        sp_human_detection,
        sp_object_detection,
        output_file=output_file,
    )
    compute_hico_map(output_map_folder, output_file, 'test')


if __name__ == "__main__":
    main()
