#!/bin/bash

# Download VCOCO detection
echo "Downloading VCOCO detection results"
mkdir -p output/VCOCO
python scripts/Download_data.py 1BdLQzS8Ebdkj7geJ3DnT_j4ZHlXc_2wb output/VCOCO/detection_merged_human_object_app.pkl

# Download HICO-DET detection results
echo "Downloading HICO-DET detection results"
mkdir -p output/HICO/matlab
python scripts/Download_data.py 1ZPRgQRiVfLZGI08Wby-2YLAV8wB3o4Pv output/HICO/matlab/HICO.zip
python scripts/Download_data.py 1K_6BejSNtjPSr2PlN83GRnfWMgQfMtlY output/HICO/matlab/HICO_finetune.zip

unzip output/HICO/matlab/HICO.zip -d output/HICO/matlab/
unzip output/HICO/matlab/HICO_finetune.zip -d output/HICO/matlab/
rm output/HICO/matlab/HICO.zip
rm output/HICO/matlab/HICO_finetune.zip