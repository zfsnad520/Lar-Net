import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


# 注意：在这个版本中，所有损失计算都在模型内部完成
# engine.py 不再需要定义损失函数

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    # --- 新增一个 AverageMeter 来跟踪梯度范数 ---
    gnorm_meter = AverageMeter('GradNorm', ':2.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter, gnorm_meter], # <-- 添加到 progress bar
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        # 你的代码中 target 需要 unsqueeze，这里保留
        target = target.cuda(non_blocking=True).unsqueeze(1) 

        with amp.autocast():
            pred, resized_target, total_loss = model(image, text, target)

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        
        # --- 核心修改：在 scaler.step 之前 unscale 并计算梯度范数 ---
        # 1. Unscale 梯度，这样我们可以观察到真实的梯度值
        scaler.unscale_(optimizer)
        
        # 2. 计算并裁剪梯度范数 (如果 args.max_norm 设置了的话)
        #    torch.nn.utils.clip_grad_norm_ 会返回裁剪前的总范数
        if args.max_norm:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        else:
            # 如果没有设置 max_norm，我们仍然可以计算它以便监控
            total_norm = torch.sqrt(sum(p.grad.data.norm(2).pow(2) for p in model.parameters() if p.grad is not None))
        # -----------------------------------------------------------

        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, resized_target, 0.35, 0.5)
        dist.all_reduce(total_loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = total_loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()
        
        # 我们也需要同步 total_norm，以便所有GPU的日志一致
        dist.all_reduce(total_norm)
        total_norm_avg = total_norm / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        # --- 更新梯度范数的 meter ---
        gnorm_meter.update(total_norm_avg.item(), image.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                        "training/grad_norm": gnorm_meter.avg, # <-- 将平均梯度范数记录到wandb
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args, len_train_loader):
    iou_list = []
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        # inference
        # 在推理模式下，模型返回 detach 过的 logits
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=False).squeeze(1)
        # process one batch
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, args.epochs, 100. * iou.item())
    logger.info(head + temp)
    if dist.get_rank() in [-1, 0]:
        wandb.log(
            {
                "val/iou": iou.item(),
                "val/prec@50": prec.get('Pr@50', 0),
                "val/prec@60": prec.get('Pr@60', 0),
                "val/prec@70": prec.get('Pr@70', 0),
                "val/prec@80": prec.get('Pr@80', 0),
                "val/prec@90": prec.get('Pr@90', 0),
            },
            step=epoch * len_train_loader
        )
    return iou.item(), prec


@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
        # multiple sentences
        for sent in param['sents']:
            mask = mask / 255.
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=False).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if args.visualize:
                pred = np.array(pred*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred)
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec