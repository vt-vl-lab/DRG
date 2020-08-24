# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
from torch import nn
from ..backbone import build_backbone
from ..ho_heads.human_head.human_head import build_human_head
from ..ho_heads.object_head.object_head import build_object_head

class appearance_branch(nn.Module):
    def __init__(self, cfg):
        super(appearance_branch, self).__init__()

        self.backbone = build_backbone(cfg)
        self.human_head = build_human_head(cfg, self.backbone.out_channels)
        self.object_head = build_object_head(cfg, self.backbone.out_channels)

    def forward(self, images, blobs, split='train'):
        if self.training or split == 'val':
            label_h = blobs['human_labels']
            label_o = blobs['object_labels']
            ho_mask = blobs['mask_ho']
        else:
            label_h = None
            label_o = None
            ho_mask = None

        features = self.backbone(images.tensors)
        h_boxes = blobs['human_boxes']
        o_boxes = blobs['object_boxes']
        pos_num = blobs['pos_num']

        human_class_logits, human_class_prob, cross_entropy_H = self.human_head(features, h_boxes, label_h, pos_num, split=split)
        object_class_logits, object_class_prob, cross_entropy_O = self.object_head(features, o_boxes, label_o, pos_num, ho_mask, split=split)

        if self.training or split == 'val':
            losses = {}
            losses['h_loss'] = cross_entropy_H
            losses['o_loss'] = cross_entropy_O
            total_loss = cross_entropy_H + cross_entropy_O
            losses['total_loss'] = total_loss

            score_summaries = {}
            score_summaries['human_class_prob'] = human_class_prob
            score_summaries['object_class_prob'] = object_class_prob
            score_summaries['cls_prob_final'] = object_class_prob * human_class_prob

            return losses, score_summaries

        cls_prob_final = object_class_prob * human_class_prob

        return cls_prob_final, human_class_prob, object_class_prob, {}
