# HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net
This is the code for **[HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net]()**.

### [Project](https://vt-le.github.io/HSTforU) | [Video](https://youtu.be/s5wmFbDnBCI) | [Paper (coming soon...)](#)
 
## Related Work
> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).



## Updates
* [8/19/2023] Code of [HSTforU](https://vt-le.github.io/HSTforU/) is released (with password). We will remove the password after this paper is accepted.

## Setup
The code can be run under any environment with Python 3.7 and above.
(It may run with lower versions, but we have not tested it).

Install the required packages:

    pip install -r requirements.txt
  
Clone this repo:

    git clone https://github.com/vt-le/HSTforU.git
    cd HSTforU/

We evaluate `HSTforU` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Homepage/Ped2/blue)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Homepage/Avenue/cyan)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Homepage/ShanghaiTech/green?)](https://svip-lab.github.io/dataset/campus_dataset.html) |
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Homepage/Drone-Anomaly/yellow)](https://github.com/Jin-Pu/Drone-Anomaly)    |

A dataset is a directory with the following structure:
  ```bash
  $ tree data
  ped2/avenue
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  ├── testing
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 000.jpg
  │       │   └── ...
  │       └── ...
  └── ped2/avenue.mat
  
  shanghaitech
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  ├── testing
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 000.jpg
  │       │   └── ...
  │       └── ...
  └── test_frame_mask
      ├── 01_0014.npy
      ├── 01_0015.npy
      └── ...

  droneanomaly
  ├──bike
  │  ├──training
  │  │  └── frames
  │  │      ├── ${video_1}$
  │  │      │   ├── 0.jpg
  │  │      │   ├── 1.jpg
  │  │      │   └── ...
  │  │      ├── ${video_2}$
  │  │      │   ├── 00.jpg
  │  │      │   └── ...
  │  │      └── ...
  │  ├── testing
  │  │   └── frames
  │  │       ├── ${video_1}$
  │  │       │   ├── 000.jpg
  │  │       │   ├── 001.jpg
  │  │       │   └── ...
  │  │       ├── ${video_2}$
  │  │       │   ├── 000.jpg
  │  │       │   └── ...
  │  │       └── ...
  │  └── annotation
  │      ├── 01.npy
  │      ├── 0.npy
  │      └── ...
  ├── highway
  │   ├── ...
  └── ...
  
  ```

## Evaluation
Please first download the pre-trained model

| Dataset | Pretrained Model                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Link/Ped2/blue?icon=chrome)](https://drive.google.com/file/d/1ALB6RjqqNN8Pgl6Xl50jD5CZPW2OFPOW/view?usp=sharing) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Link/Avenue/blue?icon=chrome)](https://drive.google.com/file/d/1fEZdStInRLad2d2qAGZjBw6r7oAMgHBN/view?usp=sharing) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Link/ShanghaiTech/blue?icon=chrome)](https://drive.google.com/file/d/1pYhyBGmDYpxjCtKZRbD8-gdF3MkjZOio/view?usp=sharing) |
| Drone-Anomaly | [![Google drive](https://badgen.net/badge/Link/Drone-Anomaly/blue?icon=chrome)](https://drive.google.com/file/d/1pYhyBGmDYpxjCtKZRbD8-gdF3MkjZOio/view?usp=sharing)    |

To evaluate a pretrained `HSTforU` on a dataset, run:

```bash
 python test.py \
    --cfg <path/to/config/file> \
    --pretrained </path/to/pre-trained/model> \
    [--batch-size <batch-size> --tag <job-tag>]
```      
 
 For example, to evaluate `HSTforU` on Ped2:

```bash
python test.py \
    --cfg config/scripts/ped2/ped2_pvt2_hst.yaml \
    --model-file output/HSTforU/ped2_pvt2_hst/ped2/ckpt_ped2.pth
```
<!-- 
## Training from scratch
To train `HSTforU` on a dataset, run:
```bash
python -m torch.distributed.launch \
    --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345  main.py \ 
    --cfg <path/to/config/file> \
    [--batch-size <batch-size-per-gpu> --tag <job-tag>]
```

For example, to train `HSTforU` on Ped2:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --cfg configs/scripts/ped2/ped2_pvt2_hst.yaml 
``` 
-->
## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{LI2024121093,
title = {HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net},
author = {Le, Viet-Tuan and Jin, Hulin and Kim, Yong-Guk},
journal = {Expert Systems with Applications},
year = {2023},
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/HSTforU/issues) or contact:

    Viet-Tuan Le: tuanlv@sju.ac.kr or vtle0011@gmail.com