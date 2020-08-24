# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.modeling import registry
from .object_feature_extractors import make_object_feature_extractor

@registry.OBJECT_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.DATASETS.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        fc7 = self.avgpool(x) # head_to_tail
        fc7 = fc7.view(x.size(0), -1)
        cls_logit = self.cls_score(fc7)
        return cls_logit


@registry.OBJECT_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.DATASETS.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        return scores

def make_object_predictor(cfg, in_channels):
    func = registry.OBJECT_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ObjectHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ObjectHead, self).__init__()
        self.feature_extractor = make_object_feature_extractor(cfg, in_channels)
        self.predictor = make_object_predictor(
            cfg, self.feature_extractor.out_channels)

        self.num_classes = cfg.DATASETS.NUM_CLASSES

    def forward(self, features, boxes, label_HO=None, pos_num=1, HO_mask=None, split='train'):
        x = self.feature_extractor(features, boxes)
        class_logits = self.predictor(x)
        class_prob = torch.sigmoid(class_logits)

        if self.training or split == 'val':
            criterion = nn.BCEWithLogitsLoss(reduction='none')

            cross_entropy_O = criterion(class_logits[:pos_num, :].view(-1, self.num_classes),
                                        label_HO[:pos_num, :]) * HO_mask[:pos_num, :]
            cross_entropy_O = cross_entropy_O.mean()
            return class_logits, class_prob, cross_entropy_O
        else:
            return class_logits, class_prob, {}

def build_object_head(cfg, in_channels):
    return ObjectHead(cfg, in_channels)
