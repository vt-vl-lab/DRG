# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import json
import logging
import numpy as np
from tqdm import tqdm
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation.vcoco.vsrl_eval import VCOCOeval
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.apply_prior import apply_prior_Graph

RNG_SEED = 0

def bbox_iou(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def run_test(
        detect_object_centric_dict,
        detect_human_centric_dict,
        detect_app_dict,
        test_image_id_list=None,
        dataset_name=None,
        action_dic_inv=None,
        output_file=None,
):

    logger = logging.getLogger("DRG.inference")
    logger.info("Start evaluation on {} dataset.".format(dataset_name))

    np.random.seed(RNG_SEED)
    detection = []

    for count, image_id in enumerate(tqdm(test_image_id_list)):
        all_lists = detect_human_centric_dict[image_id]
        object_centric_lists = detect_object_centric_dict[image_id]
        app_lists = detect_app_dict[image_id]
        for detect_idx, human_dict in enumerate(all_lists):
            human_box = human_dict['person_box']
            person_score = human_dict['person_score']
            prediction_sp = human_dict['prediction_sp']
            O_class = human_dict['o_class']
            object_boxes_cpu = human_dict['object_boxes_cpu']
            O_score = human_dict['O_score']

            app_detect_dict = app_lists[detect_idx]
            assert (app_detect_dict['person_box'] == human_box).all()
            assert app_detect_dict['person_score'] == person_score
            assert (app_detect_dict['object_boxes_cpu'] == object_boxes_cpu).all()

            prediction_H = app_detect_dict['prediction_H']
            prediction_O = app_detect_dict['prediction_O']

            prediction_sp_obj = np.zeros(prediction_sp.shape, dtype=np.float32)
            for obj_detected_human in object_centric_lists:
                if bbox_iou(obj_detected_human['person_box'], human_box) > 0.98:
                    for idx, object_box in enumerate(object_boxes_cpu):
                        if bbox_iou(obj_detected_human['object_box'], object_box) > 0.98:
                            prediction_sp_obj[idx] = obj_detected_human['prediction_sp'] * obj_detected_human['O_score']

            # H * O * Sp_H * Sp_O
            prediction_HO = prediction_H * prediction_O * prediction_sp * prediction_sp_obj
            prediction_HO = apply_prior_Graph(O_class, prediction_HO)

            # save image information
            dic = {}
            dic['image_id'] = image_id
            dic['person_box'] = human_box

            Score_obj = prediction_HO * O_score
            Score_obj = np.concatenate((object_boxes_cpu, Score_obj), axis=1)
            max_idx = np.argmax(Score_obj, 0)[4:]

            # agent mAP
            for i in range(29):
                # '''
                # walk, smile, run, stand
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    agent_name = action_dic_inv[i] + '_agent'
                    dic[agent_name] = person_score * prediction_H[0][i]
                    continue

                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = person_score * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                    continue
                if i == 4:
                    continue

                # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = person_score * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                    continue
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = person_score * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                    continue
                if i == 20:
                    continue

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'
                    dic[agent_name] = person_score * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'
                    dic[agent_name] = person_score * Score_obj[max_idx[i]][4 + i]
                    continue

                # all the rest
                agent_name = action_dic_inv[i].split("_")[0] + '_agent'
                dic[agent_name] = person_score * Score_obj[max_idx[i]][4 + i]

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    dic[action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4), person_score * prediction_H[0][i])
                    continue

                # Impossible to perform this action
                if person_score * Score_obj[max_idx[i]][4 + i] == 0:
                    dic[action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       person_score * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                    dic[action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4],
                                                       person_score * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)

    pickle.dump(detection, open(output_file, "wb" ) )

def main():
    parser = argparse.ArgumentParser(description="PyTorch DRG Detection Inference")
    parser.add_argument(
        "--dataset_name",
        default="vcoco_test",
        help="dataset name, default: vcoco_test",
    )
    parser.add_argument(
        "--app_detection",
        help="The path to the app detection pkl for test",
        default=None,
    )
    parser.add_argument(
        "--sp_human_detection",
        help="The path to the sp human detection pkl  for test",
        default=None,
    )
    parser.add_argument(
        "--sp_object_detection",
        help="The path to the sp object detection pkl for test",
        default=None,
    )

    args = parser.parse_args()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))

    detect_app_dict = pickle.load(open(args.app_detection, "rb"), encoding='latin1')
    detect_human_centric_dict = pickle.load(open(args.sp_human_detection, "rb"), encoding='latin1')
    detect_object_centric_dict = pickle.load(open(args.sp_object_detection, "rb"), encoding='latin1')

    output_folder = os.path.join(ROOT_DIR, 'output')
    mkdir(output_folder)
    output_file = os.path.join(output_folder, 'detection_vcoco_test_human_object_app.pkl')

    dataset_name = args.dataset_name
    data = DatasetCatalog.get(dataset_name)
    data_args = data["args"]
    action_dic = json.load(open(data_args['action_index']))
    action_dic_inv = {y: x for x, y in action_dic.items()}
    vcoco_test_ids = open(data_args['vcoco_test_ids_file'], 'r')
    test_image_id_list = [int(line.rstrip()) for line in vcoco_test_ids]
    vcocoeval = VCOCOeval(data_args['vcoco_test_file'], data_args['ann_file'], data_args['vcoco_test_ids_file'])

    run_test(
        detect_object_centric_dict,
        detect_human_centric_dict,
        detect_app_dict,
        test_image_id_list=test_image_id_list,
        dataset_name=dataset_name,
        action_dic_inv=action_dic_inv,
        output_file=output_file
    )

    vcocoeval._do_eval(output_file, ovr_thresh=0.5)

if __name__ == "__main__":
    main()
