# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn
from ..backbone import build_backbone
from ..spatial_heads.build_spatial_object_centric import build_spatial_object_centric

class object_centric_spatial_branch(nn.Module):
    def __init__(self, cfg):
        super(object_centric_spatial_branch, self).__init__()
        self.backbone = build_backbone(cfg)
        self.sp_branch = build_spatial_object_centric(cfg, self.backbone.out_channels)

    def forward(self, images, blobs, split='train'):
        if self.training or split == 'val':
            label_ho = blobs['ho_pair_labels_object_centric']
            ho_mask = blobs['mask_ho']
        else:
            label_ho = None
            ho_mask = None

        object_word_embedding = blobs['object_word_embeddings_object_centric']
        spatial = blobs['spatials_object_centric']
        spatial_class_logits, spatial_class_prob, cross_entropy_sp = self.sp_branch(spatial, object_word_embedding, label_ho, ho_mask, split=split)

        if self.training or split == 'val':
            losses = {}
            losses['total_loss'] = cross_entropy_sp

            score_summaries = {}
            score_summaries['spatial_object_centric_class_prob'] = spatial_class_prob
            score_summaries['cls_prob_final'] = spatial_class_prob
            return losses, score_summaries

        cls_prob_final = spatial_class_prob

        return cls_prob_final, spatial_class_prob, {}, spatial_class_prob
