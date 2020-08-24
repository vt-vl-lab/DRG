# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def load_merge_checkpoints(self, app_file, sp_file):
        self.logger.info("Loading app checkpoint from {}".format(app_file))
        self.logger.info("Loading sp checkpoint from {}".format(sp_file))
        checkpoint_app = torch.load(app_file, map_location=torch.device("cpu"))
        checkpoint_sp = torch.load(sp_file, map_location=torch.device("cpu"))
        weights_app = checkpoint_app['model']
        new_dict_app = {k.replace('module.', ''): v for k, v in weights_app.items() if 'sp_branch' not in k}
        weights_sp = checkpoint_sp['model']
        new_dict_sp = {k.replace('module.', ''): v for k, v in weights_sp.items() if 'sp_branch' in k}
        this_state = self.model.state_dict()
        this_state.update(new_dict_app)
        this_state.update(new_dict_sp)
        # self.model.load_state_dict(this_state)
        load_state_dict(self.model, this_state)
        # return any further checkpoint data
        return {}

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    # transfer pretrained weights for SSAR
    def _transfer_pretrained_weights(self, loaded_weights):
        pretrained_weights = loaded_weights['model']
        # new_dict = {k.replace('module.', ''): v for k, v in pretrained_weights.items()
        #             if 'rpn' not in k and 'roi_heads' not in k}
        new_dict = {k.replace('module.', ''): v for k, v in pretrained_weights.items() if 'rpn' not in k}

        new_dict_ours = {}
        for k, v in new_dict.items():
            if 'roi_heads' not in k:
                new_dict_ours[k] = v
            elif 'box.feature_extractor' in k:
                new_dict_ours[k.replace('roi_heads.box', 'human_head')] = v
                new_dict_ours[k.replace('roi_heads.box', 'object_head')] = v
        # this_state = model.state_dict()
        # this_state.update(new_dict)
        # model.load_state_dict(this_state)
        return new_dict_ours

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        # transfer pretrained weights for SSAR
        loaded = self._transfer_pretrained_weights(loaded)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
