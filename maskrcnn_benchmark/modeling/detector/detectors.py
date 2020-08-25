# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .object_det_rcnn import ObjectDetRCNN
from .generalized_rcnn import GeneralizedRCNN
from .h_centric_spatial_branch import human_centric_spatial_branch
from .appearance_branch import appearance_branch
from .o_centric_spatial_branch import object_centric_spatial_branch

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "human_centric_spatial_branch":human_centric_spatial_branch,
                                 "object_centric_spatial_branch":object_centric_spatial_branch,
                                 "appearance_branch": appearance_branch,
                                 "ObjectDetRCNN": ObjectDetRCNN,}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
