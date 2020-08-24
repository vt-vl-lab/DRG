# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    DATASETS = {
        "vcoco_train": {
            "im_dir": "v-coco/coco/images/train2014",
            "ann_dir": "Train_Graph_GT_Hcentric_VCOCO_dict.pkl",
            "split": "train",
            "trainval_neg_dir": "Trainval_Neg_VCOCO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        "vcoco_val": {
            "im_dir": "v-coco/coco/images/train2014",
            "ann_dir": "Val_Graph_GT_Hcentric_VCOCO_dict.pkl",
            "split": "val",
            "trainval_neg_dir": "Trainval_Neg_VCOCO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        "vcoco_test": {
            "im_dir": "v-coco/coco/images/val2014",
            "ann_dir": "v-coco/data/instances_vcoco_all_2014.json",
            "split": "test",
            "test_detection_file": "Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl",
            "prior_mask": "prior_mask_1.pkl",
            "action_index": "action_index.json",
            "word_embedding_dir": "fastText_new.pkl",
            "vcoco_test":"v-coco/data/vcoco/vcoco_test.json",
            "vcoco_test_ids": "v-coco/data/splits/vcoco_test.ids",
            "type": "human"
        },
        ################################################
        "vcoco_train_object_centric": {
            "im_dir": "v-coco/coco/images/train2014",
            "ann_dir": "Train_Graph_GT_Ocentric_VCOCO_dict.pkl",
            "split": "train",
            "trainval_neg_dir": "Trainval_Neg_VCOCO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        },
        "vcoco_val_object_centric": {
            "im_dir": "v-coco/coco/images/train2014",
            "ann_dir": "Val_Graph_GT_Ocentric_VCOCO_dict.pkl",
            "split": "val",
            "trainval_neg_dir": "Trainval_Neg_VCOCO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        },
        "vcoco_test_object_centric": {
            "im_dir": "v-coco/coco/images/val2014",
            "ann_dir": "v-coco/data/instances_vcoco_all_2014.json",
            "split": "test",
            "test_detection_file": "Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl",
            "prior_mask": "prior_mask_1.pkl",
            "action_index": "action_index.json",
            "word_embedding_dir": "fastText_new.pkl",
            "vcoco_test": "v-coco/data/vcoco/vcoco_test.json",
            "vcoco_test_ids": "v-coco/data/splits/vcoco_test.ids",
            "type": "object"
        },
        ################################################
        "hico_train": {
            "im_dir": "hico_20160224_det/images/train2015",
            "ann_dir": "Train_Graph_GT_Hcentric_HICO_dict.pkl",
            "split": "train",
            "trainval_neg_dir": "Trainval_Neg_HICO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        "hico_val": {
            "im_dir": "hico_20160224_det/images/train2015",
            "ann_dir": "Val_Graph_GT_Hcentric_HICO_dict.pkl",
            "split": "val",
            "trainval_neg_dir": "Trainval_Neg_HICO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        "hico_test": {
            "im_dir": "hico_20160224_det/images/test2015",
            "split": "test",
            "test_detection_file": "Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl",
            "threshold_dic": "threshold.pkl",
            "hico_list_vb": "hico_list_vb.txt",
            "hico_list_hoi": "hico_list_hoi.txt",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        "hico_test_finetuned": {
            "im_dir": "hico_20160224_det/images/test2015",
            "split": "test",
            "test_detection_file": "test_HICO_finetuned_v3.pkl",
            "threshold_dic": "threshold.pkl",
            "hico_list_vb": "hico_list_vb.txt",
            "hico_list_hoi": "hico_list_hoi.txt",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "human"
        },
        ################################################
        "hico_train_object_centric": {
            "im_dir": "hico_20160224_det/images/train2015",
            "ann_dir": "Train_Graph_GT_Ocentric_HICO_dict.pkl",
            "split": "train",
            "trainval_neg_dir": "Trainval_Neg_HICO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        },
        "hico_val_object_centric": {
            "im_dir": "hico_20160224_det/images/train2015",
            "ann_dir": "Val_Graph_GT_Ocentric_HICO_dict.pkl",
            "split": "val",
            "trainval_neg_dir": "Trainval_Neg_HICO_dict.pkl",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        },
        "hico_test_object_centric": {
            "im_dir": "hico_20160224_det/images/test2015",
            "split": "test",
            "test_detection_file": "Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl",
            "threshold_dic": "threshold.pkl",
            "hico_list_vb": "hico_list_vb.txt",
            "hico_list_hoi": "hico_list_hoi.txt",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        },
        "hico_test_object_centric_finetuned": {
            "im_dir": "hico_20160224_det/images/test2015",
            "split": "test",
            "test_detection_file": "test_HICO_finetuned_v3.pkl",
            "threshold_dic": "threshold.pkl",
            "hico_list_vb": "hico_list_vb.txt",
            "hico_list_hoi": "hico_list_hoi.txt",
            "word_embedding_dir": "fastText_new.pkl",
            "type": "object"
        }  
    }

    @staticmethod
    def get(name):
        if "vcoco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs["split"] == "test" or attrs["split"] == "val_test":
                args = dict(
                    im_dir=os.path.join(data_dir, attrs["im_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_dir"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"],
                    test_detection_file=os.path.join(data_dir, attrs["test_detection_file"]),
                    prior_mask=os.path.join(data_dir, attrs["prior_mask"]),
                    action_index=os.path.join(data_dir, attrs["action_index"]),
                    vcoco_test_file=os.path.join(data_dir, attrs["vcoco_test"]),
                    vcoco_test_ids_file=os.path.join(data_dir, attrs["vcoco_test_ids"])

                )
            else:
                args = dict(
                    root=os.path.join(data_dir, attrs["im_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_dir"]),
                    train_val_neg_file=os.path.join(data_dir, attrs["trainval_neg_dir"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"]
                )
            if attrs["type"] == "human":
                return dict(
                    factory="VCOCODataset",
                    args=args,
                )
            else:
                return dict(
                    factory="VCOCODatasetObject",
                    args=args,
                )
        elif "hico" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs["split"] == "test":
                args = dict(
                    root=os.path.join(data_dir, attrs["im_dir"]),
                    threshold_dic=os.path.join(data_dir, attrs["threshold_dic"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"],
                    test_detection_file=os.path.join(data_dir, attrs["test_detection_file"]),
                    hico_list_vb=os.path.join(data_dir, attrs["hico_list_vb"]),
                    hico_list_hoi=os.path.join(data_dir, attrs["hico_list_hoi"]),

                )
            else:
                args = dict(
                    root=os.path.join(data_dir, attrs["im_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_dir"]),
                    train_val_neg_file=os.path.join(data_dir, attrs["trainval_neg_dir"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"]
                )
            if attrs["type"] == "human":
                return dict(
                    factory="HICODataset",
                    args=args,
                )
            else:
                return dict(
                    factory="HICODatasetObject",
                    args=args,
                )
        elif "hcvrd" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs["split"] == "test":
                args = dict(
                    root=os.path.join(data_dir, attrs["im_dir"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"],
                    test_detection_file=os.path.join(data_dir, attrs["test_detection_file"])
                )
            else:
                args = dict(
                    root=os.path.join(data_dir, attrs["im_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_dir"]),
                    train_val_neg_file=os.path.join(data_dir, attrs["trainval_neg_dir"]),
                    word_embedding_file=os.path.join(data_dir, attrs["word_embedding_dir"]),
                    split=attrs["split"]
                )
            if attrs["type"] == "human":
                return dict(
                    factory="HCVRDDataset",
                    args=args,
                )
            else:
                return dict(
                    factory="HCVRDDatasetObject",
                    args=args,
                )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
