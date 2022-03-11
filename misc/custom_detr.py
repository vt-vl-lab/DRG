import os
import torch
import random
import pocket
import argparse
import torchvision
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from torchvision.ops.boxes import batched_nms
from torch.utils.data import (
    Dataset, DataLoader,
    DistributedSampler, BatchSampler
)

import sys
sys.path.append('detr')

from util import box_ops
from models import build_model
import datasets.transforms as T


class Engine(pocket.core.DistributedLearningEngine):
    def __init__(self, net, criterion, dataloader, max_norm, **kwargs):
        super().__init__(net, criterion, dataloader, **kwargs)
        self.max_norm = max_norm

    def _on_start_epoch(self):
        self._state.epoch += 1
        self._state.net.train()
        self._train_loader.batch_sampler.sampler.set_epoch(self._state.epoch)

    def _on_each_iteration(self):
        self._state.output = self._state.net(*self._state.inputs)
        loss_dict = self._criterion(self._state.output, self._state.targets)
        weight_dict = self._criterion.weight_dict
        self._state.loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def eval(self, postprocessors, thresh=0.1):
        self._state.net.eval()
        associate = pocket.utils.BoxAssociation(min_iou=0.5)
        meter = pocket.utils.DetectionAPMeter(
            80, algorithm='INT', nproc=10
        )
        num_gt = torch.zeros(80)

        out_dict = dict()

        if self._train_loader.batch_size != 1:
            raise ValueError(f"The batch size shoud be 1, not {self._train_loader.batch_size}")
        for image, target, id in tqdm(self._train_loader):
            im_id = id[0][0]
            out_dict[im_id] = []

            image = pocket.ops.relocate_to_cuda(image)
            output = self._state.net(image)
            output = pocket.ops.relocate_to_cpu(output)
            scores, labels, boxes = postprocessors(
                output, target[0]['size'].unsqueeze(0)
            )[0].values()
            keep = torch.nonzero(scores >= thresh).squeeze(1)
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]    # NOTE: These bboxes are scaled up (800x1333)

            ori_im_path = os.path.join('../hico_20160224_det/images/test2015/HICO_test2015_{:08d}.jpg'.format(im_id))
            from PIL import Image
            ori_im = np.array(Image.open(ori_im_path))

            curr_h, curr_w = image[0].shape[1:3]
            ori_h, ori_w = ori_im.shape[:2]
            #print(curr_h, curr_w, ori_h, ori_w)

            boxes[:, 0] = boxes[:, 0]/curr_w*ori_w
            boxes[:, 1] = boxes[:, 1]/curr_h*ori_h
            boxes[:, 2] = boxes[:, 2]/curr_w*ori_w
            boxes[:, 3] = boxes[:, 3]/curr_h*ori_h

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            cat_ids = labels_np+1    # 0-80 -> 1-81
            boxes_np = boxes.cpu().numpy()

            for o_id in range(scores_np.shape[0]):
                new_item = []
                new_item.append(im_id)

                score = np.array(scores_np[o_id]).astype(np.float32)
                box = boxes_np[o_id].astype(np.float32)
                cat_id = cat_ids[o_id]
                if cat_ids[o_id] == 1:
                    new_item.append('Human')
                else:
                    new_item.append('Object')
                new_item.append(box)
                new_item.append(np.nan)
                new_item.append(cat_id)
                new_item.append(score)
                out_dict[im_id].append(new_item)

        import pickle
        path = 'detr-r101-dc5-hicodet'
        pickle.dump(out_dict, open('Test_HICO_{}.pkl'.format(path), 'wb'))


class HICODetObject(Dataset):
    def __init__(self, dataset, transforms, nms_thresh=0.7):
        self.dataset = dataset
        self.transforms = transforms
        self.nms_thresh = nms_thresh
        self.conversion = [
             4, 47, 24, 46, 34, 35, 21, 59, 13,  1, 14,  8, 73, 39, 45, 50,  5,
            55,  2, 51, 15, 67, 56, 74, 57, 19, 41, 60, 16, 54, 20, 10, 42, 29,
            23, 78, 26, 17, 52, 66, 33, 43, 63, 68,  3, 64, 49, 69, 12,  0, 53,
            58, 72, 65, 48, 76, 18, 71, 36, 30, 31, 44, 32, 11, 28, 37, 77, 38,
            27, 70, 61, 79,  9,  6,  7, 62, 25, 75, 40, 22
        ]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, target, im_id = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        boxes[:, :2] -= 1
        labels = torch.cat([
            49 * torch.ones_like(target['object']),
            target['object']
        ])
        # Remove overlapping ground truth boxes
        keep = batched_nms(
            boxes, torch.ones(len(boxes)),
            labels, iou_threshold=self.nms_thresh
        )
        boxes = boxes[keep]
        labels = labels[keep]
        # Convert HICODet object indices to COCO indices
        converted_labels = torch.as_tensor([self.conversion[i.item()] for i in labels])
        # Apply transform
        image, target = self.transforms(image, dict(boxes=boxes, labels=converted_labels))
        return image, target, im_id

def initialise(args):
    # Load model and loss function
    detr, criterion, postprocessors = build_model(args)
    class_embed = torch.nn.Linear(256, 81, bias=True)
    if os.path.exists(args.pretrained):
        print(f"Load pre-trained model from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained)['model_state_dict'])
        w, b = detr.class_embed.state_dict().values()
        keep = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90, 91
        ]
        class_embed.load_state_dict(dict(
            weight=w[keep], bias=b[keep]
        ))
    detr.class_embed = class_embed
    if os.path.exists(args.resume):
        print(f"Resume from model at {args.resume}")
        detr.load_state_dict(torch.load(args.resume)['model_state_dict'])

    # Prepare dataset transforms
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if args.partition == 'train2015':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if args.partition == 'test2015':
        transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    # Load dataset
    dataset = HICODetObject(
        pocket.data.HICODet(
            root=os.path.join(args.data_root, f'hico_20160224_det/images/{args.partition}'),
            anno_file=os.path.join(args.data_root, f'instances_{args.partition}.json'),
            target_transform=pocket.ops.ToTensor(input_format='dict')
        ), transforms
    )
    return detr, criterion, postprocessors['bbox'], dataset

def collate_fn(batch):
    #images = []; targets = []
    #for img, tgt in batch:
    #    images.append(img)
    #    targets.append(tgt)
    #return images, targets
    images = []; targets = []; ids = []
    for img, tgt, id in batch:
        images.append(img)
        targets.append(tgt)
        ids.append(id)
    return images, targets, ids

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    model, criterion, postprocessors, dataset = initialise(args)
    if args.eval:
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler,
            batch_size=1, collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=False
        )
    else:
        sampler = DistributedSampler(dataset)
        batch_sampler = BatchSampler(
            sampler, args.batch_size,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers
        )

    engine = Engine(
        model, criterion, dataloader,
        max_norm=args.clip_max_norm,
        print_interval=args.print_interval,
        cache_dir=args.output_dir,
    )

    if args.eval:
        engine.eval(postprocessors)
        #print(f"The mAP is {ap.mean().item():.4f}, the mRec is {rec.mean().item():.4f}")
    else:
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad]
            }, {
                "params": [p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr,
            weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        engine.update_state_key(optimizer=optimizer, lr_scheduler=lr_scheduler)
        engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    model, criterion, postprocessors, dataset = initialise(args)
    image, target = dataset[0]
    print("\nPrinting out the detection target =>")
    for k, v in target.items():
        print(f"{k}: {v}")
    output = model([image])
    loss_dict = criterion(output, [target])
    print("\nPrinting out the computed losses =>")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")

    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    print("\nPrinting out the total loss =>")
    print(losses.item())

    scores, labels, boxes = postprocessors(output, target['size'].unsqueeze(0))[0].values()
    keep = torch.nonzero(scores >= 0.5).squeeze()
    print("\nPrinting out the detected instances =>")
    for c, s in zip(labels[keep], scores[keep]):
        print(f"Class {c.item()}: {s.item():.4f}")

    image = torchvision.transforms.ToPILImage()(image)
    image_copy = image.copy()
    pocket.utils.draw_boxes(image, boxes[keep], width=3)
    image.show(title='Detected boxes')

    _, _, boxes = postprocessors(
        dict(
            pred_logits=torch.rand(1, 3, 81),
            pred_boxes=target['boxes'].unsqueeze(0)
        ), target['size'].unsqueeze(0)
    )[0].values()
    pocket.utils.draw_boxes(image_copy, boxes, width=3)
    image_copy.show(title='Ground truth boxes')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--partition', default='train2015')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--data_root', default='../')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained', default='', help='Start from a pre-trained model')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--print-interval', default=1000, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sanity', action='store_true')

    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
