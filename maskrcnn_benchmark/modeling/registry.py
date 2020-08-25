# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.utils.registry import Registry

BACKBONES = Registry()
# Object detector
RPN_HEADS = Registry()
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()
# HOI
HUMAN_FEATURE_EXTRACTORS = Registry()
HUMAN_PREDICTOR = Registry()
OBJECT_FEATURE_EXTRACTORS = Registry()
OBJECT_PREDICTOR = Registry()
