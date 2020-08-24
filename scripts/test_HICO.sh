#!/bin/bash
# Usage: ./test_HICO.sh APP_ITER_NUMBER HUMAN_SP_ITER_NUMBER OBJECT_SP_ITER_NUMBER

APP_ITER_NUMBER=$1
HUMAN_SP_ITER_NUMBER=$2
OBJECT_SP_ITER_NUMBER=$3

# test on appearance stream
python tools/test_net_HICO_app.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_app_only.yaml \
        --num_iteration $APP_ITER_NUMBER
# test on spatial human stream
python tools/test_net_HICO_sp.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_human_only.yaml \
        --num_iteration $HUMAN_SP_ITER_NUMBER
# test on spatial object stream
python tools/test_net_HICO_sp_object_centric.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_HICO_sp_object_only.yaml \
        --num_iteration $OBJECT_SP_ITER_NUMBER
# merge results
APP_INTER_NUMBER_PAD="$( printf '%07d' "$APP_ITER_NUMBER" )"
HUMAN_SP_INTER_NUMBER_PAD="$( printf '%07d' "$HUMAN_SP_ITER_NUMBER" )"
OBJECT_SP_INTER_NUMBER_PAD="$( printf '%07d' "$OBJECT_SP_ITER_NUMBER" )"
APP_PKL="output/HICO_app_only/inference_ho/hico_test/model_${APP_INTER_NUMBER_PAD}/detection_times.pkl"
HUMAN_SP_PKL="output/HICO_sp_human_only/inference_sp/hico_test/model_${HUMAN_SP_INTER_NUMBER_PAD}/detection.pkl"
OBJECT_SP_PKL="output/HICO_sp_object_only/inference_sp/hico_test_object_centric/model_${OBJECT_SP_INTER_NUMBER_PAD}/detection.pkl"

python tools/test_net_HICO_merge_human_object_app.py \
       --dataset_name hico_test \
       --app_pkl $APP_PKL \
       --sp_human_pkl $HUMAN_SP_PKL \
       --sp_object_pkl $OBJECT_SP_PKL