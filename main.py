import json
from contextlib import suppress
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils.distribute import synchronize, all_gather, is_main_process
from utils.misc import SmoothedValue, MetricLogger
from collections import defaultdict
from datetime import timedelta
from data import build_dataloader
from solver import build_optimizer
from modeling import cfg, build_model
from torch.nn import functional as F
from torch.nn import SyncBatchNorm
import cv2
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from metrics import calculate_precision_at_k_and_iou_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'

SMOOTH = 1e-6
def calculate_IoU(pred, gt):
    IArea = (pred & (gt==1.0)).astype(float).sum()
    OArea = (pred | (gt==1.0)).astype(float).sum()
    # IArea = (pred * gt).astype(float).sum()
    # OArea = (pred + gt).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


def make_inputs(inputs, device):
    keys = ['img_seq', 'key_frame', 'word_ids', 'attention_mask', 'label', 'instance_mask', 'instance_coord',
            'instance_attention_mask', 'motions', 'residuals', 'resized_frame_size', 'original_frame_size',
            'valid_indices', 'img_mask', 'logits', 'valid', 'image_id', 'ori_size']
    results = {}
    if isinstance(inputs, dict):
        # print(inputs.keys())
        for key in keys:
            if key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor):
                    val = val.to(device)
                results[key] = val
    elif isinstance(inputs, list):
        targets = defaultdict(list)
        for item in inputs:
            for key in keys:
                if key in item:
                    val = item[key]
                    targets[key].append(val)

        for key in targets:
            results[key] = torch.stack(targets[key], dim=0).to(device)
    else:
        raise NotImplementedError
    return results


def train_one_epoch(cfg, args, model, device, optimizer, data_loader, summary_writer, auto_cast, loss_scaler, epoch):
    model.train()

    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    start = time.time()
    for i, inputs in enumerate(data_loader):
        # ------------ inputs ----------
        samples = make_inputs(inputs, device)
        # ------------ inputs ----------

        model_start = time.time()
        with auto_cast():
            loss_dict = model(samples)
            total_loss = sum(loss_dict.values())

        # ------------ training operations ----------
        optimizer.zero_grad()
        if cfg.SOLVER.AMPE:
            loss_scaler.scale(total_loss).backward()
            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            total_loss.backward()
            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            optimizer.step()
        # ------------ training operations ----------

        # ------------ logging ----------
        if is_main_process():
            summary_writer.global_step += 1

            if len(loss_dict) > 1:
                summary_writer.update(**loss_dict)

            summary_writer.update(lr=optimizer.param_groups[0]['lr'], total_loss=total_loss,
                                  total_time=time.time() - start, model_time=time.time() - model_start)
            start = time.time()

            speed = summary_writer.total_time.avg
            eta = str(timedelta(seconds=int((len(data_loader) - i - 1) * speed)))
            if i % 10 == 0:
                print('Epoch{:02d} ({:04d}/{:04d}): {}, Eta:{}'.format(epoch,
                                                                       i,
                                                                       len(data_loader),
                                                                       str(summary_writer),
                                                                       eta
                                                                       ), flush=True)

@torch.no_grad()
def validate_coco(cfg, model, device, data_loader, epoch):
    model.eval()
    # start_time = time.time()
    # metrics = {}
    MeanIoU, IArea, OArea, Overlap = [], [], [], []
    predictions = []
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        output = model(samples)
        gt_mask = samples['label']
        # print(samples.keys())
        # output = F.interpolate(output, gt_mask.shape[-2:], mode='bilinear', align_corners=True)[:, 0, ...]
        output = [F.interpolate(output[j][None, ...], (samples['ori_size'][0][j], samples['ori_size'][1][j]), mode='bilinear', align_corners=True)[0, 0, ...].cpu().numpy() for j in range(output.shape[0])]

        # print(gt_mask.shape)
        # print(samples['ori_size'])
        # print(inputs['ori_size'])
        image_ids = samples['image_id']
        # print(image_ids)

        # output = output * 255
        # output = output.cpu().numpy()

        pred = [(output[j] > 0.5).astype(np.uint8) for j in range(len(output))]

        # print(pred[0].shape)
        f_pred_rle_masks = [mask_util.encode(np.array(mask[..., np.newaxis], order='F'))[0] for mask in pred]
        # predictions.extend(f_pred_rle_masks)
        for i in range(len(f_pred_rle_masks)):
            predictions.append({'image_id': image_ids[i], 'category_id': 1, 'segmentation': f_pred_rle_masks[i], 'score': 1.})

        gt_mask = gt_mask.cpu().numpy()

        for i in range(len(output)):
            iou, iarea, oarea = calculate_IoU(pred[i], gt_mask[i])

            MeanIoU.append(iou)
            IArea.append(iarea)
            OArea.append(oarea)
            Overlap.append(iou)

    metric_dict = {'MeanIoU': MeanIoU, 'IArea':IArea, 'OArea':OArea, 'Overlap': Overlap, 'coco': predictions}

    synchronize()
    data_list = all_gather(metric_dict)
    if not is_main_process():
        metrics = {
            'mean_iou': 0,
            'overall_iou': 0,
            'precision5': 0,
            'precision6': 0,
            'precision7': 0,
            'precision8': 0,
            'precision9': 0,
            'precision_mAP': 0
        }
        return metrics

    MeanIoU, IArea, OArea, Overlap, Predictions = [], [], [], [], []
    # print(data_list[0][0])
    for p in data_list:
        MeanIoU.extend(p['MeanIoU'])
        IArea.extend(p['IArea'])
        OArea.extend(p['OArea'])
        Overlap.extend(p['Overlap'])
        Predictions.extend(p['coco'])

    coco_gt = COCO('./data/a2d_sentences_test_annotations_in_coco_format.json')
    coco_pred = coco_gt.loadRes(Predictions)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
    coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
    coco_eval.evaluate()
    coco_eval.accumulate()
    # coco_eval.summarize()
    # eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
    eval_metrics = {}
    if True:
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        precision_at_k, overall_iou, mean_iou, coco_eval = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred, coco_eval)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        ap_metrics = coco_eval.stats[:6]
        eval_metrics.update({l: m for l, m in zip(ap_labels, ap_metrics)})
        # eval_metrics.update()
    print(eval_metrics)

    # prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros(
    #     (len(Overlap), 1)), \
    #                                     np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    # for i in range(len(Overlap)):
    #     if Overlap[i] >= 0.5:
    #         prec5[i] = 1
    #     if Overlap[i] >= 0.6:
    #         prec6[i] = 1
    #     if Overlap[i] >= 0.7:
    #         prec7[i] = 1
    #     if Overlap[i] >= 0.8:
    #         prec8[i] = 1
    #     if Overlap[i] >= 0.9:
    #         prec9[i] = 1
    #
    # # maybe different with coco style as we could not get detailed response about the way to calculate it.
    # # it is confuse to define precision and recall for me, if we follow the prior definition of precision.
    # # anyone is welcome to pull request
    # mAP_thres_list = list(range(50, 95 + 1, 5))
    # mAP = []
    # for i in range(len(mAP_thres_list)):
    #     tmp = np.zeros((len(Overlap), 1))
    #     for j in range(len(Overlap)):
    #         if Overlap[j] >= mAP_thres_list[i] / 100.0:
    #             tmp[j] = 1
    #     mAP.append(tmp.sum() / tmp.shape[0])
    #
    # mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP = \
    #     np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), prec5.sum() / prec5.shape[0], \
    #     prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], prec8.sum() / prec8.shape[0], \
    #     prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))
    #
    # metrics = {
    #     'mean_iou': mean_iou,
    #     'overall_iou': overall_iou,
    #     'precision5': precision5,
    #     'precision6': precision6,
    #     'precision7': precision7,
    #     'precision8': precision8,
    #     'precision9': precision9,
    #     'precision_mAP': precision_mAP
    # }
    # print(metrics)
    return eval_metrics

@torch.no_grad()
def validate(cfg, model, device, data_loader, epoch, auto_cast=None):
    model.eval()
    # start_time = time.time()
    # metrics = {}
    MeanIoU, IArea, OArea, Overlap = [], [], [], []
    predictions = []
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        # start = time.time()

        output = model(samples)
        # print(time.time() - start)
        gt_mask = samples['label']
        # print(samples.keys())
        output = F.interpolate(output, gt_mask.shape[-2:], mode='bilinear', align_corners=True)[:, 0, ...]
        # print(image_ids)

        # output = output * 255
        output = output.cpu().numpy()

        pred = [(output[j] > 0.5).astype(np.uint8) for j in range(len(output))]

        # print(pred[0].shape)

        # predictions.extend(f_pred_rle_masks)
        gt_mask = gt_mask.cpu().numpy()

        for i in range(len(output)):
            iou, iarea, oarea = calculate_IoU(pred[i], gt_mask[i])

            MeanIoU.append(iou)
            IArea.append(iarea)
            OArea.append(oarea)
            Overlap.append(iou)

    metric_dict = {'MeanIoU': MeanIoU, 'IArea':IArea, 'OArea':OArea, 'Overlap': Overlap, 'coco': predictions}

    synchronize()
    data_list = all_gather(metric_dict)
    if not is_main_process():
        metrics = {
            'mean_iou': 0,
            'overall_iou': 0,
            'precision5': 0,
            'precision6': 0,
            'precision7': 0,
            'precision8': 0,
            'precision9': 0,
            'precision_mAP': 0
        }
        return metrics

    MeanIoU, IArea, OArea, Overlap = [], [], [], []
    # print(data_list[0][0])
    for p in data_list:
        MeanIoU.extend(p['MeanIoU'])
        IArea.extend(p['IArea'])
        OArea.extend(p['OArea'])
        Overlap.extend(p['Overlap'])


    prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros(
        (len(Overlap), 1)), \
                                        np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    for i in range(len(Overlap)):
        if Overlap[i] >= 0.5:
            prec5[i] = 1
        if Overlap[i] >= 0.6:
            prec6[i] = 1
        if Overlap[i] >= 0.7:
            prec7[i] = 1
        if Overlap[i] >= 0.8:
            prec8[i] = 1
        if Overlap[i] >= 0.9:
            prec9[i] = 1

    # maybe different with coco style as we could not get detailed response about the way to calculate it.
    # it is confuse to define precision and recall for me, if we follow the prior definition of precision.
    # anyone is welcome to pull request
    mAP_thres_list = list(range(50, 95 + 1, 5))
    mAP = []
    for i in range(len(mAP_thres_list)):
        tmp = np.zeros((len(Overlap), 1))
        for j in range(len(Overlap)):
            if Overlap[j] >= mAP_thres_list[i] / 100.0:
                tmp[j] = 1
        mAP.append(tmp.sum() / tmp.shape[0])

    mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP = \
        np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), prec5.sum() / prec5.shape[0], \
        prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], prec8.sum() / prec8.shape[0], \
        prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))

    metrics = {
        'mean_iou': mean_iou,
        'overall_iou': overall_iou,
        'precision5': precision5,
        'precision6': precision6,
        'precision7': precision7,
        'precision8': precision8,
        'precision9': precision9,
        'precision_mAP': precision_mAP
    }
    # print(metrics)
    return metrics


@torch.no_grad()
def validate_youtube_vos(cfg, model, device, data_loader, epoch):
    if cfg.MODEL.END_TO_END:
        return validate_youtube_vos_end_to_end(cfg, model, device, data_loader, epoch)
    model.eval()
    target_root = os.path.join(cfg.OUTPUT_DIR, 'output', str(epoch))
    if is_main_process():
        os.makedirs(target_root, exist_ok=True)
        valid_root = os.path.join(cfg.DATASETS.ROOT, 'valid/JPEGImages')
        all_video_names = os.listdir(valid_root)
        expression_path = os.path.join(cfg.DATASETS.ROOT, 'meta_expressions/valid/meta_expressions.json')
        with open(expression_path, 'r') as f:
            exp_dict = json.load(f)['videos']
        for video_name in all_video_names:
            num_expressions = len(exp_dict[video_name]['expressions'])
            for exp_id in range(num_expressions):
                os.makedirs(os.path.join(target_root, video_name, str(exp_id)), exist_ok=True)
    synchronize()

    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        output = model(samples).cpu()
        video_names, ori_sizes, cur_frames, exp_ids = inputs['video_name'], inputs['ori_size'], inputs['cur_frame'], inputs['exp_id']

        for j in range(len(output)):
            video_name, ori_size, cur_frame, exp_id = video_names[j], ori_sizes[j], cur_frames[j], exp_ids[j]
            predict = F.interpolate(output[j][None, ...], size=[ori_size[0].int(), ori_size[1].int()], mode='bilinear', align_corners=True)[0]
            predict = (predict.squeeze() > 0.5).numpy().astype(np.uint8) * 255
            target_path = os.path.join(target_root, video_name, str(exp_id.item()), cur_frame+'.png')
            # print(target_path)
            cv2.imwrite(target_path, predict)

    return {'J': 0, 'F': 0, 'Overall': 0}

@torch.no_grad()
def validate_youtube_vos_end_to_end(cfg, model, device, data_loader, epoch):
    model.eval()
    target_root = os.path.join(cfg.OUTPUT_DIR, 'output', str(epoch))
    expression_path = os.path.join(cfg.DATASETS.ROOT, 'meta_expressions/valid/meta_expressions.json')
    with open(expression_path, 'r') as f:
        exp_dict = json.load(f)['videos']
    if is_main_process():
        os.makedirs(target_root, exist_ok=True)
        valid_root = os.path.join(cfg.DATASETS.ROOT, 'valid/JPEGImages')
        all_video_names = os.listdir(valid_root)

        for video_name in all_video_names:
            num_expressions = len(exp_dict[video_name]['expressions'])
            for exp_id in range(num_expressions):
                os.makedirs(os.path.join(target_root, video_name, str(exp_id)), exist_ok=True)
    synchronize()

    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        output = model(samples)
        output = [o.cpu() for o in output]
        video_names, ori_sizes, exp_ids, cur_frames = inputs['video_name'], inputs['original_frame_size'], inputs['exp_id'], inputs['cur_frame']

        for j in range(len(output)):
            video_name, ori_size, exp_id, cur_frame_indices = video_names[j], ori_sizes[j], exp_ids[j], cur_frames[j]
            all_frame_names = exp_dict[video_name]['frames']
            # all_frames = exp_dict[video_name]['frames']
            # predict = F.interpolate(output[j][None, ...], size=[ori_size[0].int(), ori_size[1].int()], mode='bilinear', align_corners=True)[0]
            predict = output[j]
            predict = (predict.squeeze() > 0.5).numpy().astype(np.uint8) * 255
            target_paths = [os.path.join(target_root, video_name, str(exp_id.item()), all_frame_names[cur_frame_idx.item()]+'.png') for cur_frame_idx in cur_frame_indices]
            for k, target_path in enumerate(target_paths):
                cv2.imwrite(target_path, predict[k])

    return {'J': 0, 'F': 0, 'Overall': 0}


def main(cfg, args):
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TRAIN, is_train=True)
    val_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)
    # seed = 4
    if cfg.MODEL.SYNC_BN:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    # if is_main_process():
    #     print(model)
    start_epoch = 0
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}'.format(args.resume), flush=True)

    if args.test_only:
        if cfg.DATASETS.NAME != 'youtube':
            validate(cfg, model, device, val_data_loader, start_epoch)
        else:
            validate_youtube_vos(cfg, model, device, val_data_loader, start_epoch)
        return

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)
    # scheduler.step(start_epoch)

    summary_writer = MetricLogger(log_dir=os.path.join(output_dir, 'logs')) if is_main_process() else None
    if summary_writer is not None:
        summary_writer.add_meter('lr', SmoothedValue(fmt='{value:.5f}'))
        summary_writer.add_meter('total_time', SmoothedValue(fmt='{avg:.3f}s'))
        summary_writer.add_meter('model_time', SmoothedValue(fmt='{avg:.3f}s'))

    auto_cast = torch.cuda.amp.autocast if cfg.SOLVER.AMPE else suppress
    loss_scaler = torch.cuda.amp.GradScaler() if cfg.SOLVER.AMPE else None

    # metrics = validate(cfg, model, device, val_data_loader, -1)
    # print(metrics)
    # print(metrics)
    if cfg.DATASETS.NAME != 'youtube':
        metrics = validate(cfg, model, device, val_data_loader, 0)
    else:
        metrics = validate_youtube_vos(cfg, model, device, val_data_loader, 0)
    best_metric = 0
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, args, model, device, optimizer, train_data_loader, summary_writer, auto_cast, loss_scaler, epoch)
        if cfg.DATASETS.NAME != 'youtube':
            metrics = validate(cfg, model, device, val_data_loader, epoch)
        else:
            metrics = validate_youtube_vos(cfg, model, device, val_data_loader, epoch)
        scheduler.step()

        if is_main_process():
            model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            if cfg.DATASETS.NAME != 'youtube':
                save_path = os.path.join(output_dir, f'model_best_{epoch}.pth')
                # if metrics['mean_iou'] > best_metric:
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'metrics': metrics
                }, save_path)
                    # best_metric = metrics['mean_iou']
            else:
                save_path = os.path.join(output_dir, f'model_epoch{epoch}.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'metrics': metrics
                }, save_path)

            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                write_str = ['Epoch: {:02d}'.format(epoch)]
                for key in metrics:
                    write_str.append(f'{key}: {metrics[key]}')
                write_str = '\t'.join(write_str)
                f.write(write_str+'\n')
                print(write_str)

            print('Saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--t", type=float, default=0.06)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    # parser.add_argument('--root', default='/home/tiger/videos-cuts/GEBD-end-2-end/data/dongchedi')

    args = parser.parse_args()
    local_rank = args.local_rank
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

    main(cfg, args)





