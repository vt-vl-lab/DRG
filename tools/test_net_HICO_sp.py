# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import logging
import glob

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
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.Generate_HICO_detection import Generate_HICO_detection
from maskrcnn_benchmark.data.datasets.evaluation.hico.hico_compute_mAP import compute_hico_map

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
    # human_box = human_box.numpy()
    # object_box = object_box.numpy()
    H, O = bbox_trans(human_box, object_box)

    Pattern = np.zeros((2, 64, 64))
    Pattern[0, int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1] = 1
    Pattern[1, int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1] = 1

    return Pattern


def im_detect(model, image_id, Test_RCNN, word_embeddings, object_thres, human_thres, detection, device, opt):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    im_file = os.path.join(DATA_DIR, 'hico_20160224_det', 'images', 'test2015', 'HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg')
    img_original = Image.open(im_file)
    img_original = img_original.convert('RGB')
    im_shape = (img_original.height, img_original.width)  # (480, 640)
    transforms = build_transforms(cfg, is_train=False)

    This_human = []

    for Human in Test_RCNN[image_id]:

        if (np.max(Human[5]) > human_thres) and (Human[1] == 'Human'):  # This is a valid human

            O_box = np.empty((0, 4), dtype=np.float32)
            O_vec = np.empty((0, 300), dtype=np.float32)
            Pattern = np.empty((0, 2, 64, 64), dtype=np.float32)
            O_score = np.empty((0, 1), dtype=np.float32)
            O_class = np.empty((0, 1), dtype=np.int32)

            for Object in Test_RCNN[image_id]:
                if opt['use_thres_dic'] == 1:
                    object_thres_ = opt['thres_dic'][Object[4]]
                else:
                    object_thres_ = object_thres

                if (np.max(Object[5]) > object_thres_) and not (np.all(Object[2] == Human[2])):  # This is a valid object

                    O_box_ = np.array([Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 4)
                    O_box = np.concatenate((O_box, O_box_), axis=0)

                    O_vec_ = word_embeddings[Object[4]]
                    O_vec = np.concatenate((O_vec, O_vec_), axis=0)

                    Pattern_ = generate_spatial(Human[2], Object[2]).reshape(1, 2, 64, 64)
                    Pattern = np.concatenate((Pattern, Pattern_), axis=0)

                    O_score = np.concatenate((O_score, np.max(Object[5]).reshape(1, 1)), axis=0)
                    O_class = np.concatenate((O_class, np.array(Object[4]).reshape(1, 1)), axis=0)

            if len(O_box) == 0:
                continue
            H_box = np.array([Human[2][0], Human[2][1], Human[2][2], Human[2][3]]).reshape(1, 4)

            blobs = {}
            blobs['pos_num'] = len(O_box)
            pos_num = len(O_box)
            human_boxes_cpu = np.tile(H_box, [len(O_box), 1]).reshape(pos_num, 4)
            human_boxes = torch.FloatTensor(human_boxes_cpu)
            object_boxes_cpu = O_box.reshape(pos_num, 4)
            object_boxes = torch.FloatTensor(object_boxes_cpu)

            human_boxlist = BoxList(human_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)
            object_boxlist = BoxList(object_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)

            img, human_boxlist, object_boxlist = transforms(img_original, human_boxlist, object_boxlist)

            spatials = []
            for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
                ho_spatial = generate_spatial(human_box.numpy(), object_box.numpy()).reshape(1, 2, 64, 64)
                spatials.append(ho_spatial)
            blobs['spatials'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
            blobs['human_boxes'], blobs['object_boxes'] = (human_boxlist,), (object_boxlist,)
            blobs['object_word_embeddings'] = torch.FloatTensor(O_vec).reshape(pos_num, 300)

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

            # convert to np.array
            prediction_HO = prediction_HO.data.cpu().numpy()
            prediction_H = prediction_H.data.cpu().numpy()
            # prediction_O = prediction_O.data.cpu().numpy()
            prediction_sp = prediction_sp.data.cpu().numpy()

            for idx in range(len(prediction_HO)):
                temp = []
                temp.append(Human[2])  # Human box
                temp.append(O_box[idx])  # Object box
                temp.append(O_class[idx])  # Object class
                temp.append(prediction_HO[idx])  # Score
                temp.append(Human[5])  # Human score
                temp.append(O_score[idx])  # Object score
                This_human.append(temp)

    detection[image_id] = This_human



def run_test(
            model,
            dataset_name=None,
            test_detection=None,
            word_embeddings=None,
            output_file=None,
            object_thres=0.4,
            human_thres=0.6,
            device=None,
            cfg=None,
            opt=None
):
    logger = logging.getLogger("DRG.inference")
    logger.info("Start evaluation on {} dataset.".format(dataset_name))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))

    image_list = glob.glob(os.path.join(DATA_DIR, 'hico_20160224_det', 'images', 'test2015', '*.jpg'))
    np.random.seed(cfg.TEST.RNG_SEED)
    detection = {}

    for idx, line in enumerate(tqdm(image_list)):

        image_id = int(line[-9:-4])

        if image_id in test_detection:
            im_detect(model, image_id, test_detection, word_embeddings, object_thres, human_thres, detection, device, opt)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    num_devices = 1
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(image_list), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(image_list),
            num_devices,
        )
    )

    pickle.dump(detection, open(output_file, "wb"))


def main():
    #     apply_prior   prior_mask
    # 0        -             -
    # 1        Y             -
    # 2        -             Y
    # 3        Y             Y
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="",
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
                        default=0.4, type=float)  # used to be 0.4 or 0.05
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.6, type=float)
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

    print('prior flag: {}'.format(args.prior_flag))

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args.config_file = os.path.join(ROOT_DIR, args.config_file)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("DRG.inference", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
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

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            if args.num_iteration != -1:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_sp", dataset_name,
                                             "model_%07d" % args.num_iteration)
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_sp", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    opt = {}
    opt['word_dim'] = 300
    opt['use_thres_dic'] = 1
    for output_folder, dataset_name in zip(output_folders, dataset_names):
        data = DatasetCatalog.get(dataset_name)
        data_args = data["args"]
        test_detection = pickle.load(open(data_args['test_detection_file'], "rb"), encoding='latin1')
        word_embeddings = pickle.load(open(data_args['word_embedding_file'], "rb"), encoding='latin1')
        opt['thres_dic'] = pickle.load(open(data_args['threshold_dic'], "rb"), encoding='latin1')
        output_file = os.path.join(output_folder, 'detection.pkl')
        # hico_folder = os.path.join(output_folder, 'HICO')
        output_map_folder = os.path.join(output_folder, 'map')

        logger.info("Output will be saved in {}".format(output_file))
        logger.info("Start evaluation on {} dataset.".format(dataset_name))

        run_test(
            model,
            dataset_name=dataset_name,
            test_detection=test_detection,
            word_embeddings=word_embeddings,
            output_file=output_file,
            object_thres=args.object_thres,
            human_thres=args.human_thres,
            device=device,
            cfg=cfg,
            opt=opt
        )

        # Generate_HICO_detection(output_file, hico_folder)
        compute_hico_map(output_map_folder, output_file, 'test')


if __name__ == "__main__":
    main()
