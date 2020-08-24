#!/bin/bash

# train on appearance stream
python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_app_only.yaml
# train on spatial human stream
python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_human_only.yaml
# train on spatial object stream
python tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_object_only.yaml