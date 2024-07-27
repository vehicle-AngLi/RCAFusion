
#  Intro

The code and pre-trained model of "RCAFusion: Cross Rubik Cube Attention Network for Multi-modal Image Fusion of Intelligent Vehicles" now have been available. Thank you for the download and interest! This outcome has been guided by many previous papers, thanks to authors in each of the references.

## Environment

 - torch==2.0.1
 - torchvision==0.15.2
 - pandas==2.0.3
 - pillow==9.5.0
 - numpy==1.24.4
 - ultralytics==8.0.188
 
## To Train

if you wanna train your own network, Please: 

1. Copy your infrared images of dataset to "**./train_data/inf/**", visible images of dataset to "**./train_data/vis/**"
2. Run "**python train_rca.py --batch_size ... --gpu 0 --epochs ...**", please set proper batch size and epochs for your train.

## To Test

if you wanna run the pre-trained model of RCAFusion

1. Copy your infrared images for fusion to "**./test_img/ir/**", visible images for fusion to "**./test_img/vi/**"
2. Run "**python test.py**", Fused images will be placed in "**./Save/**"


if you wanna run your model

Run "**python test.py --model_path ... --ir_dir ... --vi_dir ... --gpu 0**". Please point out the absolute path of your model in model_path, and the path of infrared images in ir_dir, visible images in vi_dir.



## If this work is helpful to you, please cite our paper! Thanks! Feel free to contact us at email: vehicle_liang@163.com

## cite us at:

@INPROCEEDINGS{10588756,
  author={Li, Ang and Yin, Guodong and Wang, Ziwei and Liang, Jinhao and Wang, Fanxun and Bai, Xin and Liu, Zhichao},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={RCAFusion: Cross Rubik Cube Attention Network for Multi-modal Image Fusion of Intelligent Vehicles}, 
  year={2024},
  volume={},
  number={},
  pages={2848-2854},
  doi={10.1109/IV55156.2024.10588756}}

## This work is developed from Linfeng Tang's team, i am grateful to their outstanding work and recommend you to read or cite their works, such as:

@article{Tang2022ImageFI,
  title={Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
  author={Linfeng Tang and Jiteng Yuan and Jiayi Ma},
  journal={Inf. Fusion},
  year={2022},
  volume={82},
  pages={28-42},
  url={https://api.semanticscholar.org/CorpusID:245643604}
}
