# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import json
import logging
import os
import numpy as np
import gzip
from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def read(file_name, mode='rb'):
    with open(file_name, mode) as f:
        return f.read()


def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)


def load_json_object(file_name, compress=False):
    if compress:
        return json.loads(gzip.decompress(read(file_name)).decode('utf8'))
    else:
        return json.loads(read(file_name, 'r'))


def dump_json_object(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')