#!/bin/bash

# Download Faster R-CNN pre-trained weight
echo "Downloading Faster R-CNN pre-trained weight"
mkdir -p pretrained_model
cd pretrained_model
wget https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth

