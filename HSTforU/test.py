import os
import time
import json
import inspect
import datetime

import torch
import torch.backends.cudnn as cudnn

# datasets
from datasets.build_dataset import build_test_loader
from datasets.label import Label
# models
import models
# utils
from utils.logger import create_test_logger
from utils.anomaly_score import calculate_auc, psnr_park
# visualization
from visualization.progress import progress_bar

# train
from train import parse_option

# python test.py --cfg configs/scripts/ped2.yaml --batch-size 4 --gpus 0 --pretrained ckpt_ped2.pth
# python test.py --cfg configs/scripts/avenue.yaml --batch-size 4 --gpus 0 --pretrained ckpt_avenue.pth
# python test.py --cfg configs/scripts/shanghaitech.yaml --batch-size 4 --gpus 0 --pretrained ckpt_shanghaitech.pth

def main(config, logger):
    dataset_test, data_loader_test = build_test_loader(config)
    logger.info(f"The number of input sequences of {config.DATA.DATASET.upper()} dataset: {len(dataset_test)}")

    logger.info(f"--------------------------------------------")
    logger.info(f"{inspect.getsourcefile(models.build_model)}")
    logger.info(f"--------------------------------------------")
    model = models.build_model(config, logger=logger, is_trained=False)
    model.cuda(device=config.GPUS)

    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    pretrained = os.path.join(config.OUTPUT, config.MODEL.PRETRAINED)
    if isinstance(pretrained, str):
        checkpoint = torch.load(pretrained, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

        del checkpoint
        torch.cuda.empty_cache()

    mse = torch.nn.MSELoss(reduction='none')
    psnrs = validate(config, data_loader_test, model, mse, logger)

    mat_loader = Label(config)
    mat = mat_loader()
    assert len(psnrs) == len(mat), f'Ground truth has {len(mat)} frames, BUT got {len(psnrs)} detected frames!'

    auc, fpr, tpr = calculate_auc(config, psnrs, mat)

    ckpt_epoch = os.path.splitext(os.path.basename(config.MODEL.PRETRAINED))[0]
    logger.info(f"AUC of {ckpt_epoch}: {auc * 100:.1f}%")
    print(f'-------------------------------------------------------------------------------')


def validate(config, data_loader, model, mse, logger):
    model.eval()

    if config.DATA.DATASET.lower() in ['ped1', 'ped2', 'avenue', 'shanghaitech', 'drone',
                                       'drone/railway', 'drone/highway', 'drone/crossroads', 'drone/bike',
                                       'drone/vehicle', 'drone/solar', 'drone/farmland']:
        dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, 'testing', "frames")
        if config.DATA.DATASET_SCENE != '':  # Drone dataset has evaluated on different scenes
            dir = os.path.join(dir, config.DATA.DATASET_SCENE)
        assert (os.path.exists(dir))
        dirs = [x[0] for x in os.walk(dir, followlinks=True)]  # if not x[0].startswith('.')]
        num_videos = len(dirs) - 1
    else:
        assert config.DATA.DATASET.lower(), f' dataset is not evaluated!'
    logger.info(f"The number of videos ({config.DATA.DATASET}/{config.DATA.DATASET_SCENE}): {num_videos}")

    psnrs = [[] for i in range(num_videos)]
    start = time.time()
    for i, (frames, i_video, idx) in enumerate(data_loader):
        progress_bar((i + 1) * config.DATA.BATCH_SIZE, len(data_loader) * config.DATA.BATCH_SIZE)
        frames = [frame.cuda(device=config.GPUS) for frame in frames]
        target = frames[-1]
        frames = frames[:-1]

        # compute output
        output = model(frames)
        bs = output.shape[0]

        # compute PSNR
        for j in range(bs):
            mse_imgs = torch.mean(mse((output[j] + 1) / 2, (target[j] + 1) / 2)).item()
            psnr = psnr_park(mse_imgs)

            psnrs[i_video[j].item()].append(psnr)

    print('')  # Enter to a new line
    test_time = time.time() - start
    logger.info(f'Dataset {config.DATA.DATASET} takes {datetime.timedelta(seconds=int(test_time))}')

    return psnrs


if __name__ == '__main__':
    args, config = parse_option()

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_test_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # print config
    # logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, logger)
