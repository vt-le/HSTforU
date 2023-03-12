# HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net
This is the code for "HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net".

 * [Project Page](https://vt-le.github.io/HSTforU)
 * [Paper](https://arxiv.org)
 * [Video](https://youtu.be/s5wmFbDnBCI)
 
This codebase is implemented using [PyTorch](https://pytorch.org/).

![Code coming soon](static/images/code_coming_soon.png "Code coming soon")

## Setup
The code can be run under any environment with Python 3.7 and above.
(It may run with lower versions, but we have not tested it).

Install the required packages:

    pip install -r requirements.txt
  

## Training

Download `weights`, and put them in a folder `pretrained/`

| Dataset | Link                                                                                   |
|--|----------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://colab.research.google.com/assets/colab-badge.svg)](https://)  |
| CUHK Avenue | [![Google drive](https://colab.research.google.com/assets/colab-badge.svg)](https://) |
| ShanghaiTech | [![Google drive](https://colab.research.google.com/assets/colab-badge.svg)](https://) |
| Drone-Anomaly | [![Google drive](https://colab.research.google.com/assets/colab-badge.svg)](https://) |
 

After preparing a dataset, you can train a model by running:
    
    python train.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin
 

## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

 
## Datasets
A dataset is a directory with the following structure:

    dataset
        ├── train
        │   └── ${video_id}
        │   └── └── ${frame_id}.jpg
        ├── test
        │   ├── ${video_id}
        │   └── └── ${frame_id}.jpg
        └── dataset.mat

At a high level, a dataset is simply the following:
 * A collection of images (e.g., from a video).
 * XXX for each image.
 
In each video folder, we have a unique identifier for each image which we call `frame_id`, which is extracted from the 
video. An `frame_id` can be any string, but typically it is some alphanumeric string such as `0250`.


## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{le2023hstforu
  author    = {Le, Viet-Tuan and Jin, Hulin and Kim, Yong-Guk},
  title     = {HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net},
  journal   = {IEEE},
  year      = {2023},
  publisher = {IEEE}
}
```

## Related Work
> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).