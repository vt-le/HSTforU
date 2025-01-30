import os
import natsort
from PIL import Image
from random import randrange

import torch
import torch.distributed as dist
import torch.utils.data as data
from torchvision import transforms

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_transform(is_train, config):
    if is_train:
        transform = create_transform(
            input_size=(config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            is_training=True,
            hflip=config.AUG.HFLIP,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation=config.DATA.INTERPOLATION,
        )
        transform.transforms[0] = transforms.Resize((config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
                                                    interpolation=_pil_interp(config.DATA.INTERPOLATION))
        return transform
    else:
        t = [transforms.Resize((config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
                               interpolation=_pil_interp(config.DATA.INTERPOLATION)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]

        return transforms.Compose(t)


def collect_files(root):
    include_ext = [".png", ".jpg", "jpeg", ".bmp"]
    # collect subfolders
    dirs = [x[0] for x in os.walk(root, followlinks=True)]  # if not x[0].startswith('.')]

    # sort both dirs and individual images
    dirs = natsort.natsorted(dirs)

    dataset = [
        [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
         if os.path.isfile(os.path.join(fdir, el))
         and not el.startswith('.')
         and any([el.endswith(ext) for ext in include_ext])]
        for fdir in dirs
    ]

    return [el for el in dataset if el]


def get_training_frames(videos, frame_steps, num_input_frames, min_frames):
    output = [[] for i in range(frame_steps)]
    for video in videos:
        if frame_steps == 1:
            step = 1
            max_frames = (len(video) - num_input_frames)

            for j in range(0, max_frames + 1, step):
                for s in range(0, frame_steps):
                    output[s].append(video[j:j + num_input_frames])
        elif len(video) > min_frames and frame_steps != 1:
            max_frames = (len(video) - frame_steps - num_input_frames + 1) // frame_steps * frame_steps

            for j in range(0, max_frames + 1, frame_steps):
                for s in range(0, frame_steps):
                    output[s].append(video[j + s:j + s + num_input_frames])
        else:
            step = frame_steps // 3 if frame_steps >= 3 else 1
            max_frames = (len(video) - num_input_frames - step + 1) // step * step

            for j in range(0, max_frames + 1, step):
                for s in range(0, frame_steps):
                    si = s % step
                    output[s].append(video[j+si:j+si+num_input_frames])

    return output


class BuildTrainDataset(data.Dataset):
    def __init__(self, config, is_train=True):
        super(BuildTrainDataset, self).__init__()
        self.transform_samples = build_transform(is_train=is_train, config=config)
        self.transform_targets = build_transform(is_train=not is_train, config=config)

        dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, 'training')
        assert (os.path.exists(dir))

        videos = collect_files(dir)

        num_input_frames = config.DATA.NUM_INPUT_FRAMES
        frame_steps = config.DATA.FRAME_STEP
        min_frames = config.DATA.MIN_FRAMES
        self.frame_steps = frame_steps

        self.videos = get_training_frames(videos, frame_steps, num_input_frames, min_frames)

    def __len__(self):
        return len(self.videos[-1])     # or return len(self.videos[0])

    def __getitem__(self, index):
        if self.frame_steps == 1:
            video_name = self.videos[0][index]
        else:
            i_list = randrange(self.frame_steps)
            video_name = self.videos[i_list][index]

        raw_frames = [Image.open(f).convert('RGB') for f in video_name]

        video = []
        for i, f in enumerate(raw_frames):
            if i == len(raw_frames) - 1:
                f = self.transform_targets(f)
            else:
                f = self.transform_samples(f)
            video.append(f)

        return {'video': video, 'video_name': video_name}


def build_train_loader(config):
    dataset_train = BuildTrainDataset(config=config, is_train=True)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    return dataset_train, data_loader_train


def build_train_par_loader(config):
    dataset_train = BuildTrainDataset(config=config, is_train=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE * len(config.GPUS),
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    return dataset_train, data_loader_train


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
def build_indexes(videos):
    # frames = []
    indexes = []
    start, end = 0, -1
    for i, video in enumerate(videos):
        start = end + 1
        end = start + len(video) - 1
        indexes.append([start, end])
        # for j in range(len(video)):
        #     frames.append(video[j])
    return indexes


def get_video_index(indexes, ind):
    i = -1
    for i, index in enumerate(indexes):
        if index[0] <= ind <= index[len(index) - 1]:
            return i
    return i


class BuildTestDataset(data.Dataset):
    def __init__(self, config):
        super(BuildTestDataset, self).__init__()
        self.transform = build_transform(is_train=False, config=config)

        dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, 'testing', "frames")
        if config.DATA.DATASET_SCENE != '':     ### Drone
            dir = os.path.join(dir, config.DATA.DATASET_SCENE)
        assert (os.path.exists(dir))

        self.videos = collect_files(dir)

        # Each video is excluded (num_input_frames - 1)
        self.num_input_frames = config.DATA.NUM_INPUT_FRAMES
        self.cut_videos = []

        self.num_videos = 0  # number of videos after exclude (num_input_frames) frames
        for video in self.videos:
            self.num_videos += (len(video) - (self.num_input_frames - 1))
            cut_video = []
            for i in range(len(video) - (self.num_input_frames - 1)):
                cut_video.append(video[i])
            self.cut_videos.append(cut_video)

        self.indexes = build_indexes(self.cut_videos)

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        i_video = get_video_index(self.indexes, index)
        idx = index - self.indexes[i_video][0]
        video = self.videos[i_video]

        output = []
        for i in range(self.num_input_frames):
            frame = Image.open(video[idx + i]).convert('RGB')
            output.append(self.transform(frame))
        return output, i_video, idx
        # return {'output': output, 'i_video': i_video}


def build_test_loader(config):
    dataset_test = BuildTestDataset(config=config)
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build test dataset")

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset_test, data_loader_test
