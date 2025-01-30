import os
import time
import json
import random
import inspect
import argparse
import datetime
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

# configs
from configs.default import get_config
# datasets
from datasets.build_dataset import build_train_loader
# models
import models
# utils
from utils.logger import create_logger
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from utils.criterion import Losses
from utils.anomaly_score import psnr_park
from utils.checkpoint import save_checkpoint
# visualization


def parse_option():
    parser = argparse.ArgumentParser('HSTforU training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--gpus', type=int, help='GPU index')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, logger):
    dataset_train, data_loader_train = build_train_loader(config)
    logger.info(f"The number of input sequences of {config.DATA.DATASET.upper()} dataset: {len(dataset_train)}")

    # logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger.info(f"-------------------------------------------------")
    logger.info(f"{inspect.getsourcefile(models.build_model)}")
    logger.info(f"-------------------------------------------------")
    model = models.build_model(config, logger=logger)
    # logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters:,}")

    model.cuda()
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = Losses(config).cuda()
    mse = torch.nn.MSELoss(reduction='none')

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, mse, logger)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger, resume=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, mse, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, samples in enumerate(data_loader):
        samples = samples['video']
        samples, targets = samples[:-1], samples[-1]
        targets = targets.cuda(non_blocking=True)

        outputs = model(samples)

        int_l, gra_l, mss_l, l2_l = criterion(outputs, targets)
        loss = int_l + gra_l + mss_l + l2_l

        # compute PSNR
        mse_imgs = torch.mean(mse((outputs + 1) / 2, (targets + 1) / 2)).item()
        psnr = psnr_park(mse_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'lr {lr:.6f}\t'
                f'psnr {psnr:.2f}\t'
                f'int_l {int_l:.6f}\t'
                f'gra_l {gra_l:.6f}\t'
                f'mss_l {mss_l:.6f}\t'
                f'l2_l {l2_l:.6f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed_type = '9xx'
    if seed_type == '9xx':
        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed)
    elif seed_type == 'swin':
        # Swin
        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True
    elif seed_type == 'pvt':
        # PVT
        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, logger)
