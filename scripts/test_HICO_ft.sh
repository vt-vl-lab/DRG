#!/bin/bash

# test on appearance stream
python tools/test_net_HICO_app.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_app_only_ft.yaml \
        --ckpt output/HICO/model_app.pth
# test on spatial human stream
python tools/test_net_HICO_sp.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_human_only_ft.yaml \
        --ckpt output/HICO/model_sp_human.pth
# test on spatial object stream
python tools/test_net_HICO_sp_object_centric.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_object_only_ft.yaml \
        --ckpt output/HICO/model_sp_object.pth
# merge results
APP_PKL="output/HICO_app_only_ft/inference_ho/hico_test_finetuned/detection_times.pkl"
HUMAN_SP_PKL="output/HICO_sp_human_only_ft/inference_sp/hico_test_finetuned/detection.pkl"
OBJECT_SP_PKL="output/HICO_sp_object_only_ft/inference_sp/hico_test_object_centric_finetuned/detection.pkl"

python tools/test_net_HICO_merge_human_object_app.py \
       --dataset_name hico_test_finetuned \
       --app_pkl $APP_PKL \
       --sp_human_pkl $HUMAN_SP_PKL \
       --sp_object_pkl $OBJECT_SP_PKL
