# WSESeg: Introducing a Dataset for the Segmentation of Winter Sports Equipment with a Baseline for Interactive Segmetation
This paper contains the code and dataset download instructions for the paper "WSESeg: Introducing
a Dataset for the Segmentation of Winter Sports Equipment with a Baseline for Interactive Segmetation" by 
Robin Schön, Daniel Kienzle and Rainer Lienhart. The paper is part of the proceedings of the CBMI 2024. 

## Obtaining the WSESeg dataset
WSESeg is a dataset which contains segmentation masks for winter sports equipment. The types of winter 
sports equipment are: Bobsleigh, Curling Broom, Curling Stone, Ski Goggles, Ski Helmet, Slalom Gate Poles, 
Skis (in the context of ski jumping), Skis (in miscellaneous contexts), Snowboard and Snowkite. 

The dataset only contains the segmentation masks, lists of pairs of the form (user ID, image ID), and a 
python script to automatically download the images from Flickr. In order to obtain the images themselves, 
you will have to obtain an API key and an API secret, and accept Flickrs terms of service. 

The dataset download can be found at [this adress](https://myweb.rz.uni-augsburg.de/~schoerob/datasets/wseseg/wseseg.html). 

## Running the code 
First you will have to install the dependencies. To our knowledge these dependencies are at least made up of:
- torch
- pillow
- numpy
- tqdm
- scipy
- opencv-python

Before you can replicate the rows of the tables in the paper, you may want to make modifications to `run_experiment_series_sam.py`
- In order to select the table row that you want to run, set the `RUN_ID` to the index of a run in the list `RUNS`. 
- In order to select whether to run SAM or HQ-SAM, set `BACKBONE_SIZE` to `b` or `b_hq` respectively. 
- You may want to change the variable `DEVICE` to the respective device-id you want to run your code on.

You should download the following two weight files and store them in the folder `./weights/`: 
- The SAM ViT-B weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
- The HQ-SAM ViT-B weights: https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing 

In case you want to change the paths to store the weights, these are available in the file `config.py`. 
The code itself can be run by `python3 run_experiment_series_sam.py`. 

## Citation
In case you want to use our code or dataset, please cite our paper 
(citation to be replaced with CBMI proceedings citation): 
```
@misc{schoen2024wseseg,
      title={WSESeg: Introducing a Dataset for the Segmentation of Winter Sports Equipment with a Baseline for Interactive Segmentation},
      author={Robin Schön and Daniel Kienzle and Rainer Lienhart},
      year={2024},
      eprint={2407.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.09288},
}
``` 
The weights and architectures of SAM and HQ-SAM belong to the following repositories: 
- SAM: https://github.com/facebookresearch/segment-anything
- HQ-SAM: https://github.com/SysCV/sam-hq

In case you want to use those parts of the code, please cite their papers as well: 
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```