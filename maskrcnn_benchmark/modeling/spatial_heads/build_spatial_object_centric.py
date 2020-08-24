import torch
import torch.nn.functional as F
from torch import nn
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SPModuleObjectCentric(torch.nn.Module):
    """
    Module for spatial branch computation.
    """
    def __init__(self, cfg, in_channels):
        super(SPModuleObjectCentric, self).__init__()

        self.cfg = cfg.clone()
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.sp_to_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            Flatten()
        )  # output size, [N, 5408]

        sp_head_channels = 5408 + cfg.DATASETS.WORD_DIM
        self.sp_head_channels = sp_head_channels
        self.ssar_key_net = nn.Linear(sp_head_channels, 1024, bias=None)  # 5708, 1024 (5708 = 5408 + cfg.DATASETS.WORD_DIM)
        self.ssar_value_net = nn.Linear(sp_head_channels, sp_head_channels, bias=None)  # 5708, 5708
        self.ssar_attn_dropout = nn.Dropout(p=0.7, inplace=True)
        self.ssar_layernorm = nn.LayerNorm(sp_head_channels)  # 5708

        self.sp_head_to_tail = nn.Sequential(
            nn.Linear(sp_head_channels, 1024),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.5, inplace=True)
        )
        self.cls_score_net_sp = nn.Linear(1024, self.num_classes)

        self.init_weights()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def init_weights(self):
        def normal_init(m, mean, stddev):
            """
            weight initalizer: random normal.
            """
            # x is a parameter
            m.weight.data.normal_(mean, stddev)
            if m.bias is not None:
                m.bias.data.zero_()

        for layer in self.sp_to_head:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01)
        normal_init(self.cls_score_net_sp, 0, 0.01)
        normal_init(self.ssar_key_net, 0, 0.01)
        # initialize ssar_value_net with identity function
        self.ssar_value_net.weight.data.copy_(torch.eye(self.sp_head_channels))

        for layer in self.sp_head_to_tail:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01)

    def refinement(self, net_human_embed):
        '''
        refinement module
        :param net_human_embed: spatial branch embedding, size [N, 7756], N is batch size
        :return: weights: size [N, 7756]
        '''
        key_embed = self.ssar_key_net(net_human_embed) # key and query share weights, n*1024
        query_embed = self.ssar_key_net(net_human_embed)# n*1024
        value_embed = self.ssar_value_net(net_human_embed) # n*5708

        weights = torch.matmul(query_embed, torch.t(key_embed)) # key_embed.permute(1, 0), n*n
        weights = weights / math.sqrt(key_embed.size(-1)) # math.sqrt(key_embed.size(-1)) = sqrt(1024) = 32
        softmax_attention = nn.Softmax(dim=1)(weights) # n*n

        weights = torch.matmul(softmax_attention, value_embed) # n*5708
        # weights = F.relu(weights, inplace=True)
        weights = F.relu(weights)
        weights = self.ssar_attn_dropout(weights)
        weights = self.ssar_layernorm(net_human_embed + weights)

        return weights

    def forward(self, spatial, object_word_embedding, label_ho=None, ho_mask=None, split='train'):
        sp = self.sp_to_head(spatial) # [-1, 5408]
        net_word_sp = torch.cat((object_word_embedding, sp), 1) # [-1, 5708]

        X_1 = self.refinement(net_word_sp)
        X_2 = self.refinement(X_1)

        fc9_sp = self.sp_head_to_tail(X_2)

        cls_score_sp = self.cls_score_net_sp(fc9_sp)
        cls_prob_sp = torch.sigmoid(cls_score_sp)

        if self.training or split == 'val':
            cross_entropy_sp = self.criterion(cls_score_sp.view(-1, self.num_classes),
                                              label_ho) * ho_mask
            cross_entropy_sp = cross_entropy_sp.mean()
            return cls_score_sp, cls_prob_sp, cross_entropy_sp
        else:
            return cls_score_sp, cls_prob_sp, {}

def build_spatial_object_centric(cfg, in_channels):
    return SPModuleObjectCentric(cfg, in_channels)
