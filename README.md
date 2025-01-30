# HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net
This is the code for **[HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net]()** [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fvt-le%2FHSTforU&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com).

### [Project](https://vt-le.github.io/HSTforU) | [Video](https://youtu.be/s5wmFbDnBCI) | [Paper](https://link.springer.com/article/10.1007/s10489-024-06042-4?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20250103&utm_content=10.1007%2Fs10489-024-06042-4)
 
## Related Works
> **CrossAnomaly**: See [CrossAnomaly: A Contextual Cross-Modality Framework for Video Anomaly Detection](https://github.com/vt-le/CrossAnomaly).

> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).

<a href="static/img/shanghai/shanghai_curve_full.gif" target="_blank">
    <image style="border: 2px solid rgb(201, 196, 196);" src="static/images/03_0031.gif" width="100%">
</a>

## Updates
* [8/19/2023] Code of [HSTforU](https://vt-le.github.io/HSTforU/) is released.

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

  drone
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

## Training
To train `HSTforU` on a dataset, run:
```bash
 python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  train.py --cfg <config-file>
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
    --model-file output/HSTforU/ped2_pvt2_hst/ckpt_ped2.pth
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
﻿@Article{le2025hstforu,
author={Le, Viet-Tuan
and Jin, Hulin
and Kim, Yong-Guk},
title={HSTforU: anomaly detection in aerial and ground-based videos with hierarchical spatio-temporal transformer for U-net},
journal={Applied Intelligence},
year={2025},
month={Jan},
day={03},
volume={55},
number={4},
pages={261},
issn={1573-7497},
doi={10.1007/s10489-024-06042-4},
url={https://doi.org/10.1007/s10489-024-06042-4}
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/HSTforU/issues) or contact:

    Viet-Tuan Le: vt-le@outlook.com
