#!/bin/bash
# Usage: ./test_VCOCO.sh APP_ITER_NUMBER HUMAN_SP_ITER_NUMBER OBJECT_SP_ITER_NUMBER

APP_ITER_NUMBER=$1
HUMAN_SP_ITER_NUMBER=$2
OBJECT_SP_ITER_NUMBER=$3

# test on appearance stream
python tools/test_net_VCOCO_app.py \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_VCOCO_app_only.yaml \
        --num_iteration $APP_ITER_NUMBER
# test on spatial human stream
python tools/test_net_VCOCO_sp.py \
        --dataset_name vcoco_test \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_VCOCO_sp_human_only.yaml \
        --num_iteration $HUMAN_SP_ITER_NUMBER
# test on spatial object stream
python tools/test_net_VCOCO_sp_object_centric.py \
        --dataset_name vcoco_test_object_centric \
        --config-file configs/e2e_faster_rcnn_R_50_FPN_1x_VCOCO_sp_object_only.yaml \
        --num_iteration $OBJECT_SP_ITER_NUMBER

# merge results
APP_INTER_NUMBER_PAD="$( printf '%07d' "$APP_ITER_NUMBER" )"
HUMAN_SP_INTER_NUMBER_PAD="$( printf '%07d' "$HUMAN_SP_ITER_NUMBER" )"
OBJECT_SP_INTER_NUMBER_PAD="$( printf '%07d' "$OBJECT_SP_ITER_NUMBER" )"
python tools/test_net_VCOCO_merge_human_object_app.py \
    --dataset_name vcoco_test \
    --app_detection output/VCOCO_app_only/inference_ho/vcoco_test/model_${APP_INTER_NUMBER_PAD}/detection_app_vcoco_test_new.pkl \
    --sp_human_detection output/VCOCO_sp_human_only/inference_sp/vcoco_test/model_${HUMAN_SP_INTER_NUMBER_PAD}/detection_human_vcoco_test_new.pkl \
    --sp_object_detection output/VCOCO_sp_object_only/inference_sp/vcoco_test_object_centric/model_${OBJECT_SP_INTER_NUMBER_PAD}/detection_object_centric_all_pairs_vcoco_test_object_centric_new.pkl