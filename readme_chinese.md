
## 介绍

论文《RCAFusion: Cross Rubik Cube Attention Network for Multi-modal Image Fusion of Intelligent Vehicles》的代码与预训练模型现在已经开源啦！感谢您的下载与关注。本成果受益于很多前人的研究，向参考文献中的每个作者致以最诚挚的感谢。

## 环境配置

 - torch==2.0.1
 - torchvision==0.15.2
 - pandas==2.0.3
 - pillow==9.5.0
 - numpy==1.24.4
 - ultralytics==8.0.188
 
## 训练

如果你要训练自己的网络模型，请：

1. 将训练数据集中的红外图像粘贴到"**./train_data/ir/**"，可见光图像粘贴到"**./train_data/vi/**"。
2. 运行指令"**python train_rca.py --batch_size ... --gpu 0 --epochs ...**"，请根据自己的显卡情况调整训练的batchsize与epochs。

## 测试

如果你要直接运行预训练的RCAFusion模型，请：

1. 将待融合的红外图像放置在"**./train_data/ir/**"，可见光图像放在"**./train_data/vi/**"。
2. 运行指令"**python test.py**"，融合后的图像将会放在"**./Save/**"。


如果你要运行你自己训练的模型：

运行指令"**python test.py --model_path ... --ir_dir ... --vi_dir ... --gpu 0**"，三个参数分别需要指明模型、红外图像与可见光图像的路径。

## 希望本项工作对您有所帮助，如果您喜欢我们的代码，请您引用我们的文章！谢谢！如果您有任何疑问请联系：vehicle_liang@163.com。

## 您可以通过以下方式引用本文章：（会议论文集暂未出版）
@INPROCEEDINGS{RCAFusion,
  author={Li, Ang and Yin, Guodong and Wang, Ziwei and Liang, Jinhao and Wang, Fanxun and Bai, Xin and Liu, Zhichao},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={RCAFusion: Cross Rubik Cube Attention Network for Multi-modal Image Fusion of Intelligent Vehicles}, 
  year={2024},
  volume={},
  number={},
  pages={1-7},
  doi={}}

## 本文优化基于Linfeng Tang的融合架构，感谢作者提出了优秀的融合算法，我推荐您阅读、引用他们的作品，如：
@article{Tang2022ImageFI,
  title={Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
  author={Linfeng Tang and Jiteng Yuan and Jiayi Ma},
  journal={Inf. Fusion},
  year={2022},
  volume={82},
  pages={28-42},
  url={https://api.semanticscholar.org/CorpusID:245643604}
}