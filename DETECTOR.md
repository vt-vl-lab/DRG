# Fine-tuning object detectors

## How to use your own detector
**NOTE:** If you wish the use the same detector for a fair comparison, you can replace the provided fine-tuned detector with your own detector with the following steps:

1. Generate a detection file in `.pkl`, following the iCAN format [here](https://github.com/vt-vl-lab/iCAN/blob/master/misc/Object_Detector.py).
2. Update the path of your detector file in default config [here](https://github.com/vt-vl-lab/DRG/blob/master/maskrcnn_benchmark/config/paths_catalog.py#L98) and [here](https://github.com/vt-vl-lab/DRG/blob/master/maskrcnn_benchmark/config/paths_catalog.py#L135).


## Results
We here provide some numbers of our method by replacing the object detector. Note that we simply replace the detector file without re-training our method.

Detector | Obj det. mAP on HICO test | Full (python) | Rare (python) | Non Rare (python) | Full (Matlab) | Rare (Matlab) | Non Rare (Matlab) |
---------------|------------|------------|------------|------------|------------|------------|------------|
Our fine-tuned | - | 24.51 | 19.4 | 26.03 | 24.53 | 19.47 | 26.04 |
R101-FPN Faster R-CNN (from [VCL](https://github.com/zhihou7/VCL)) [[ckpt]](https://drive.google.com/file/d/1RgWNoc-lk8HMlcttzLghPg8LAPCdmCCG/view?usp=sharing) [[pkl]](https://drive.google.com/file/d/1QI1kcZJqI-ym6AGQ2swwp4CKb39uLf-4/view?usp=sharing) | 30.79 | 20.71 | 16.37 | 22.01 | - | - | - |
R50 DETR (from [UPT](https://github.com/fredzzhang/upt))[[ckpt]](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing) [[pkl]](https://filebox.ece.vt.edu/~ylzou/HOI/detection/Test_HICO_detr-r50-hicodet.pkl) | - | 21.52 | 19.31 | 22.19 | - | - | - |
R101 DETR (from [UPT](https://github.com/fredzzhang/upt))[[ckpt]](https://drive.google.com/file/d/1pZrRp8Qcs5FNM9CJsWzVxwzU7J8C-t8f/view?usp=sharing) [[pkl]](https://filebox.ece.vt.edu/~ylzou/HOI/detection/Test_HICO_detr-r101-hicodet.pkl) | - | 21.98 | 19.5 | 22.72 | - | - | - |
R101-DC5 DETR (from [UPT](https://github.com/fredzzhang/upt))[[ckpt]](https://drive.google.com/file/d/1kkyVeoUGb8rT9b5J5Q3f51OFmm4Z73UD/view?usp=sharing) [[pkl]](https://filebox.ece.vt.edu/~ylzou/HOI/detection/Test_HICO_detr-r101-dc5-hicodet.pkl) | - | 21.66 | 19.66 | 22.25 | - | - | - |
*X152-FPN Cascade R-CNN [[ckpt]](https://filebox.ece.vt.edu/~ylzou/HOI/ckpt/cascade_faster_rcnn_X152_FPN_lr1e-3_20k.pth) [[pkl]](https://filebox.ece.vt.edu/~ylzou/HOI/detection/Test_HICO_cascade_rcnn_X152_FPN_lr1e-3_20k.pkl) | 37.61 | 24.89 | 20.2 | 26.28 | - | - | - |

*We re-train a model for reproducibility only, please use our original detection pkl file if you wish to compare to other methods using the same detector.


## Fine-tuning w/ detectron2

1. Set up your detectron2 environment following the [official instructions](https://detectron2.readthedocs.io/tutorials/install.html).
2. Download HICO detection annotations from VCL: [[training set]](https://drive.google.com/file/d/1qyUURe978WuZRm1s-VWoC_TpTInYTUXd/view?usp=sharing) [[test set]](https://drive.google.com/file/d/1M4j5-rHcdfHYVfHQToccO0SsEGP4nGC1/view?usp=sharing), and register a new dataset accordingly in detectron2.
3. Choose a config file and download the COCO pre-trained weights, then you can start. Usually a learning rate 1e-3 is good.
4. To reproduce the new X152-FPN Cascade R-CNN results, you can use this [config file](https://filebox.ece.vt.edu/~ylzou/HOI/config/cascade_faster_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml) and download the [pre-trained weight](https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl). We trained it using 2 V100 GPUs and take the 20k step checkpoint. Note that it has not converged, so you may get even better results if you select a later checkpoint.

We also provide a simple script to convert detectron2 test json file to iCAN format pkl file. Please find it [here](misc/convert_json_to_pkl.py).


## Fine-tuning w/ DETR
Set up a forked version of DETR [here](https://github.com/fredzzhang/hicodet). You will need to follow the instructions to install the pocket library.

To get iCAN format pkl file from this repo:
1. Replace the original `pocket/data/hicodet.py` in the installed pocket directory, with [this file](misc/hicodet.py). The original pocket implementation skips ~100 images in the test set, while our current python evaluation requires evaluating on **all** the test images even if they have no annotations.
2. Test your model and output iCAN format pkl file using [this script](misc/custom_detr.py).


## Conclusion
Object detector can largely affect the final HOI detection results. Sometimes a 0.02 difference in object detection mAP can lead to 0.3~0.4 difference in HOI detection. Thus, we highly recommend using the **same** detector for a fair comparison. If you use your own fine-tuned detector, please try to use it for your competing methods as well.
