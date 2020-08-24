# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import tensorboardX as tb

from apex import amp

def add_score_summary(key, tensor):
    return tb.summary.histogram(
        'SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

def run_summary_op(event_summaries, score_summaries=None, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # # Add image gt
    # summaries.append(add_gt_image_summary())
    # Add event_summaries
    for key, var in event_summaries.items():
        summaries.append(tb.summary.scalar(key, var.item()))
    if not val:
        # Add score summaries
        for key, var in score_summaries.items():
            summaries.append(add_score_summary(key, var))

    return summaries

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
):
    # tensorboard directory
    tb_dir = cfg.TB_DIR
    tbval_dir = cfg.TB_DIR + '_val'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    if not os.path.exists(tbval_dir):
        os.makedirs(tbval_dir)

    # Remove previous events
    filelist = [ f for f in os.listdir(tb_dir)]
    for f in filelist:
        os.remove(os.path.join(tb_dir, f))

    filelist = [ f for f in os.listdir(tbval_dir)]
    for f in filelist:
        os.remove(os.path.join(tbval_dir, f))

    # tensorboard initialization
    writer = tb.writer.FileWriter(tb_dir)
    valwriter = tb.writer.FileWriter(tbval_dir)

    logger = logging.getLogger("DRG.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, blobs, idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        blobs = blobs[0]
        for key in blobs.keys():
            if not isinstance(blobs[key], int) and not isinstance(blobs[key], tuple):
                blobs[key] = blobs[key].to(device)
            elif isinstance(blobs[key], tuple):
                blobs[key] = [boxlist.to(device) for boxlist in blobs[key]]

        model.train()
        loss_dict, score_summaries = model(images, blobs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        meters.update(**loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        with amp.scale_loss(loss_dict['total_loss'], optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration == 1 or iteration % cfg.SOLVER.SUMMARY_INTERVAL == 0 or iteration == max_iter:
            summaries = run_summary_op(loss_dict_reduced, score_summaries, val=False)
            for _sum in summaries:
                writer.add_summary(_sum, float(iteration))

        if iteration == 1 or iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and (iteration % test_period == 0 or iteration == 1):
            meters_val = MetricLogger(delimiter="  ")
            loss_dict_val_total = {}
            for k in loss_dict.keys():
                loss_dict_val_total[k+'_val'] = torch.tensor(0.0).to(device)

            synchronize()
            model.eval()
            # Should be one image for each GPU:
            with torch.no_grad():
                for iteration_val, (images_val, blobs_val, _) in enumerate(tqdm(data_loader_val)):
                    blobs_val = blobs_val[0]
                    for key in blobs_val.keys():
                        if not isinstance(blobs_val[key], int) and not isinstance(blobs_val[key], tuple):
                            blobs_val[key] = blobs_val[key].to(device)
                        elif isinstance(blobs_val[key], tuple):
                            blobs_val[key] = [boxlist.to(device) for boxlist in blobs_val[key]]
                    images_val = images_val.to(device)
                    # targets_val = [target.to(device) for target in targets_val
                    loss_dict_val, score_summaries_val = model(images_val, blobs_val, 'val')
                    # losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced_val = reduce_loss_dict(loss_dict_val)
                    # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    # meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                    meters_val.update(**loss_dict_reduced_val)

                    for k in loss_dict_reduced_val.keys():
                        loss_dict_val_total[k+'_val'] += loss_dict_reduced_val[k]

            synchronize()

            for k in loss_dict_val_total.keys():
                loss_dict_val_total[k] = loss_dict_val_total[k] / len(data_loader_val)
            # add tensorboard summary
            summaries_val = run_summary_op(loss_dict_val_total, score_summaries_val, val=True)
            for _sum in summaries_val:
                valwriter.add_summary(_sum, float(iteration))

            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    writer.close()
    valwriter.close()
