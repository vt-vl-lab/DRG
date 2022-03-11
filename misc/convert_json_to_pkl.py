# Convert detectron2 json result to iCAN pkl file

import os
import json
import pickle
import numpy as np

path = 'cascade_rcnn_X152_FPN_lr1e-3'
full_path = 'coco_instances_results_X152.json'

data = json.load(open(full_path))

# Create a mapping from all 91 to 80 training classes
id2train_id = {}
with open('coco_labels_paper.txt') as f:
    names = f.read().rstrip().split('\n')
with open('coco_train_labels.txt') as f:
    train_names = f.read().rstrip().split('\n')
train_id = 1
for id, name in enumerate(names):
    if name in train_names:
        id2train_id[id+1] = train_id
        train_id += 1

out_dict = dict()
for det in data:
    im_id = det['image_id']
    cat_id = det['category_id']
    cat_id = id2train_id[cat_id]
    bbox = np.array(det['bbox']).astype(np.float32)    # seems to be xywh format?
    bbox[2] = bbox[0]+bbox[2]
    bbox[3] = bbox[1]+bbox[3]
    score = np.array(det['score']).astype(np.float32)

    if im_id not in out_dict:
        out_dict[im_id] = []

    new_item = []
    new_item.append(im_id)
    if cat_id == 1:
        new_item.append('Human')
    else:
        new_item.append('Object')
    new_item.append(bbox)
    new_item.append(np.nan)
    new_item.append(cat_id)
    new_item.append(score)
    out_dict[im_id].append(new_item)

pickle.dump(out_dict, open('Test_HICO_{}.pkl'.format(path), 'wb'))

