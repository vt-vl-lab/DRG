# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import json
import logging

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation.vcoco.vsrl_eval import VCOCOeval
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.apply_prior import apply_prior_Graph

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def bbox_trans(human_box_ori, object_box_ori, size=64):
    human_box = human_box_ori.copy()
    object_box = object_box_ori.copy()

    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)

    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2

        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def generate_spatial(human_box, object_box):
    H, O = bbox_trans(human_box, object_box)

    Pattern = np.zeros((2, 64, 64))
    Pattern[0, int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1] = 1
    Pattern[1, int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1] = 1

    return Pattern


def im_detect(model, im_dir, image_id, Test_RCNN, fastText, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection, detect_app_dict, device, cfg):

    # im_orig, im_shape = get_blob(image_id, cfg)

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    if "train" in im_dir:
        im_file = os.path.join(DATA_DIR, im_dir, 'COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg')
    else:
        im_file = os.path.join(DATA_DIR, im_dir, 'COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg')
    img_original = Image.open(im_file)
    img_original = img_original.convert('RGB')
    # when using Image.open to read images, img.size= (640, 480), while using cv2.imread, im.shape = (480, 640)
    # to be consistent with previous code, I used img.height, img.width here
    im_shape = (img_original.height, img_original.width)  # (480, 640)
    transforms = build_transforms(cfg, is_train=False)


    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human

            O_box   = np.empty((0, 4), dtype=np.float32)
            O_vec   = np.empty((0, 300), dtype=np.float32)
            Pattern = np.empty((0, 2, 64, 64), dtype=np.float32)
            O_score = np.empty((0, 1), dtype=np.float32)
            O_class = np.empty((0, 1), dtype=np.int32)
            Weight_mask = np.empty((0, 29), dtype=np.float32)

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    O_box_ = np.array([Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1,4)
                    O_box  = np.concatenate((O_box, O_box_), axis=0)

                    O_vec_ = fastText[Object[4]]
                    O_vec  = np.concatenate((O_vec, O_vec_), axis=0)

                    # Pattern_ = Get_next_sp(Human_out[2], Object[2]).reshape(1, 64, 64, 2)
                    # Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
                    Pattern_ = generate_spatial(Human_out[2], Object[2]).reshape(1, 2, 64, 64)
                    Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

                    O_score  = np.concatenate((O_score, np.max(Object[5]).reshape(1,1)), axis=0)
                    O_class  = np.concatenate((O_class, np.array(Object[4]).reshape(1,1)), axis=0)

                    Weight_mask_ = prior_mask[:,Object[4]].reshape(1,29)
                    Weight_mask  = np.concatenate((Weight_mask, Weight_mask_), axis=0)

            H_box = np.array([Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,4)

            if len(O_box) == 0:
                continue

            blobs = {}
            pos_num = len(O_box)
            blobs['pos_num'] = pos_num
            # blobs['dropout_is_training'] = False
            human_boxes_cpu = np.tile(H_box, [len(O_box), 1]).reshape(pos_num, 4)
            human_boxes = torch.FloatTensor(human_boxes_cpu)
            object_boxes_cpu = O_box.reshape(pos_num, 4)
            object_boxes = torch.FloatTensor(object_boxes_cpu)

            blobs['object_word_embeddings']  = torch.FloatTensor(O_vec).reshape(pos_num, 300)


            human_boxlist = BoxList(human_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)
            object_boxlist = BoxList(object_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)

            img, human_boxlist, object_boxlist = transforms(img_original, human_boxlist, object_boxlist)

            spatials = []
            for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
                ho_spatial = generate_spatial(human_box.numpy(), object_box.numpy()).reshape(1, 2, 64, 64)
                spatials.append(ho_spatial)
            blobs['spatials'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
            blobs['human_boxes'], blobs['object_boxes'] = (human_boxlist,), (object_boxlist,)

            for key in blobs.keys():
                if not isinstance(blobs[key], int) and not isinstance(blobs[key], tuple):
                    blobs[key] = blobs[key].to(device)
                elif isinstance(blobs[key], tuple):
                    blobs[key] = [boxlist.to(device) for boxlist in blobs[key]]

            image_list = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)
            image_list = image_list.to(device)

            # compute predictions
            model.eval()
            with torch.no_grad():
                prediction_HO, prediction_H, prediction_O, prediction_sp = model(image_list, blobs)

            #convert to np.array
            # test h + o branch Only
            prediction_HO = prediction_H * prediction_O

            prediction_HO = prediction_HO.data.cpu().numpy()
            prediction_H = prediction_H.data.cpu().numpy()
            prediction_O = prediction_O.data.cpu().numpy()
            # prediction_sp = prediction_sp.data.cpu().numpy()

            dic_save = {}
            dic_save['image_id'] = image_id
            dic_save['person_box'] = Human_out[2]
            dic_save['person_score'] = np.max(Human_out[5])
            dic_save['prediction_HO'] = prediction_HO
            dic_save['prediction_H'] = prediction_H
            dic_save['prediction_O'] = prediction_O
            dic_save['o_class'] = O_class
            dic_save['object_boxes_cpu'] = object_boxes_cpu
            dic_save['O_score'] = O_score

            detect_app_dict[image_id].append(dic_save)

            if prior_flag == 1:
                prediction_HO  = apply_prior_Graph(O_class, prediction_HO)
            if prior_flag == 2:
                prediction_HO  = prediction_HO * Weight_mask
            if prior_flag == 3:
                prediction_HO  = apply_prior_Graph(O_class, prediction_HO)
                prediction_HO  = prediction_HO * Weight_mask

            # save image information
            dic = {}
            dic['image_id']   = image_id
            dic['person_box'] = Human_out[2]

            Score_obj = prediction_HO * O_score
            Score_obj = np.concatenate((object_boxes_cpu, Score_obj), axis=1)

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj,0)[4:]

            # agent mAP
            for i in range(29):
                #'''
                # walk, smile, run, stand
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    agent_name      = Action_dic_inv[i] + '_agent'
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][i]
                    continue

                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                    continue
                if i == 4:
                    continue

                # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                    continue
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                    continue
                if i == 20:
                    continue

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                # all the rest
                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'
                dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * prediction_H[0][i])
                    continue

                # Impossible to perform this action
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i] == 0:
                   dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                   dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)


def run_test(
        model,
        im_dir=None,
        dataset_name=None,
        test_detection=None,
        word_embeddings=None,
        test_image_id_list=None,
        prior_mask=None,
        action_dic_inv=None,
        output_file=None,
        output_dict_file=None,
        object_thres=0.4,
        human_thres=0.6,
        prior_flag=1,
        device=torch.device("cuda"),
        cfg=None,
):

    logger = logging.getLogger("DRG.inference")
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(test_image_id_list)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    np.random.seed(cfg.TEST.RNG_SEED)
    detection = []
    detect_app_dict = {}

    for count, image_id in enumerate(tqdm(test_image_id_list)):
        detect_app_dict[image_id] = []
        im_detect(model, im_dir, image_id, test_detection, word_embeddings, prior_mask, action_dic_inv, object_thres, human_thres,
                  prior_flag, detection, detect_app_dict, device, cfg)

    pickle.dump(detect_app_dict, open(output_dict_file, "wb"))
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    num_devices = 1
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(test_image_id_list), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(test_image_id_list),
            num_devices,
        )
    )

    pickle.dump(detection, open(output_file, "wb" ) )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument('--num_iteration', dest='num_iteration',
            help='Specify which weight to load',
            default=-1, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.1, type=float)  # used to be 0.4 or 0.05
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--prior_flag', dest='prior_flag',
                        help='whether use prior_flag',
                        default=1, type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1 and torch.cuda.is_available()

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    args.config_file = os.path.join(ROOT_DIR, args.config_file)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("DRG", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    # model.to(cfg.MODEL.DEVICE)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)

    if args.num_iteration != -1:
        args.ckpt = os.path.join(cfg.OUTPUT_DIR, 'model_%07d.pth' % args.num_iteration)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    logger.info("Testing checkpoint {}".format(ckpt))
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    # iou_types = ("bbox",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            if args.num_iteration != -1:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_ho", dataset_name, "model_%07d" % args.num_iteration)
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_ho", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    opt = {}
    opt['word_dim'] = 300
    for output_folder, dataset_name in zip(output_folders, dataset_names):
        data = DatasetCatalog.get(dataset_name)
        data_args = data["args"]
        im_dir = data_args['im_dir']
        test_detection = pickle.load(open(data_args['test_detection_file'], "rb"), encoding='latin1')
        prior_mask = pickle.load(open(data_args['prior_mask'], "rb"), encoding='latin1')
        action_dic = json.load(open(data_args['action_index']))
        action_dic_inv = {y: x for x, y in action_dic.items()}
        vcoco_test_ids = open(data_args['vcoco_test_ids_file'], 'r')
        test_image_id_list = [int(line.rstrip()) for line in vcoco_test_ids]
        vcocoeval = VCOCOeval(data_args['vcoco_test_file'], data_args['ann_file'], data_args['vcoco_test_ids_file'])
        word_embeddings = pickle.load(open(data_args['word_embedding_file'], "rb"), encoding='latin1')
        output_file = os.path.join(output_folder, 'detection.pkl')
        output_dict_file = os.path.join(output_folder, 'detection_app_{}_new.pkl'.format(dataset_name))

        logger.info("Output will be saved in {}".format(output_file))
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(test_image_id_list)))

        run_test(
            model,
            dataset_name=dataset_name,
            im_dir=im_dir,
            test_detection=test_detection,
            word_embeddings=word_embeddings,
            test_image_id_list=test_image_id_list,
            prior_mask=prior_mask,
            action_dic_inv=action_dic_inv,
            output_file=output_file,
            output_dict_file=output_dict_file,
            object_thres=args.object_thres,
            human_thres=args.human_thres,
            prior_flag=args.prior_flag,
            device=device,
            cfg=cfg
        )

        synchronize()

        vcocoeval._do_eval(output_file, ovr_thresh=0.5)

    # data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    # for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
    #     inference(
    #         model,
    #         data_loader_val,
    #         dataset_name=dataset_name,
    #         iou_types=iou_types,
    #         box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
    #         bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
    #         device=cfg.MODEL.DEVICE,
    #         expected_results=cfg.TEST.EXPECTED_RESULTS,
    #         expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    #         output_folder=output_folder,
    #     )
    #     synchronize()


if __name__ == "__main__":
    main()
