#!/bin/bash

# Download COCO dataset
echo "Downloading V-COCO"

# Download V-COCO
mkdir Data
git clone --recursive https://github.com/s-gupta/v-coco.git Data/v-coco/
cd Data/v-coco/coco

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip
URL_2014_Trainval_annotation=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images
wget -N $URL_2014_Test_images
wget -N $URL_2014_Trainval_annotation

mkdir images

unzip train2014.zip -d images/
unzip val2014.zip -d images/
unzip test2014.zip -d images/
unzip annotations_trainval2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip
rm annotations_trainval2014.zip

# Pick out annotations from the COCO annotations to allow faster loading in V-COCO
echo "Picking out annotations from the COCO annotations to allow faster loading in V-COCO"

cd ../
python script_pick_annotations.py coco/annotations

# Build
echo "Building"
cd coco/PythonAPI/
2to3 . -w
make install
cd ../../
2to3 . -w
make
cd ../../

# Download HICO-DET dataset
echo "Downloading HICO-DET"

python scripts/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

# Download HICO-DET evaluation code
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp scripts/Generate_detection.m Data/ho-rcnn/
cp scripts/save_mat.m Data/ho-rcnn/
cp scripts/load_mat.m Data/ho-rcnn/
cp scripts/eval_one.m Data/ho-rcnn/evaluation/
cp scripts/eval_run.m Data/ho-rcnn/evaluation/

mkdir Data/ho-rcnn/data/hico_20160224_det/
python scripts/Download_data.py 1cE10X9rRzzqeSPi-BKgIcDgcPXzlEoXX Data/ho-rcnn/data/hico_20160224_det/anno_bbox.mat
python scripts/Download_data.py 1ds_qW9wv-J3ESHj_r_5tFSOZozGGHu1r Data/ho-rcnn/data/hico_20160224_det/anno.mat

mkdir -p Data/ho-rcnn/cache/det_base_caffenet/train2015
python scripts/Download_data.py 1agWZrvHi9arGj_RrUan04dXvhrNVirjY Data/ho-rcnn/cache/det_base_caffenet/train2015/HICO_train2015_00000001.mat

# Download Pre-trained weights
echo "Downloading Fast RCNN Pre-trained weights..."

mkdir pretrained_model/
cd pretrained_model
wget -N https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth
cd ../

# output folder
mkdir output