[简体中文](./README.md) | English

# Sound classification system implemented in Pytorch

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/AudioClassification-Pytorch)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/AudioClassification-Pytorch)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/AudioClassification-Pytorch)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

**Disclaimer, this document was obtained through machine translation, please check the original document [here](./README.md).**


# Introduction

This project is a sound classification project based on Pytorch, aiming to realize the recognition of various environmental sounds, animal calls and languages. Several sound classification models such as EcapaTdnn, PANNS, ResNetSE, CAMPPlus, and ERes2Net are provided to support different application scenarios. In addition, the project also provides the commonly used Urbansound8K dataset test report and some dialect datasets download and use examples. Users can choose suitable models and datasets according to their needs to achieve more accurate sound classification. The project has a wide range of application scenarios, and can be used in outdoor environmental monitoring, wildlife protection, speech recognition and other fields. At the same time, the project also encourages users to explore more usage scenarios to promote the development and application of sound classification technology.


# Environment

 - Anaconda 3
 - Python 3.11
 - Pytorch 2.0.1
 - Windows 11 or Ubuntu 22.04

# Project Features

1. Supporting models: EcapaTdnn、PANNS、TDNN、Res2Net、ResNetSE、CAMPPlus、ERes2Net
2. Supporting pooling: AttentiveStatsPool(ASP)、SelfAttentivePooling(SAP)、TemporalStatisticsPooling(TSP)、TemporalAveragePooling(TAP)
3. Support preprocessing methods: MelSpectrogram、Spectrogram、MFCC、Fbank、Wav2vec2.0、WavLM

**Model Paper：**

- EcapaTdnn：[ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143v3)
- PANNS：[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211v5)
- TDNN：[Prediction of speech intelligibility with DNN-based performance measures](https://arxiv.org/abs/2203.09148)
- Res2Net：[Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)
- ResNetSE：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- CAMPPlus：[CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking](https://arxiv.org/abs/2303.00332v3)
- ERes2Net：[An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification](https://arxiv.org/abs/2305.12838v1)

# Model Test

|    Model     | Params(M) | Preprocessing method |   Dataset    | Number Class | Accuracy |
|:------------:|:---------:|:--------------------:|:------------:|:------------:|:--------:|
|   ResNetSE   |    7.8    |        Flank         | UrbanSound8K |      10      | 0.96233  |
|  ERes2NetV2  |    5.4    |        Flank         | UrbanSound8K |      10      | 0.95662  |
|   CAMPPlus   |    7.1    |        Flank         | UrbanSound8K |      10      | 0.95454  |
|  EcapaTdnn   |    6.4    |        Flank         | UrbanSound8K |      10      | 0.95227  |
|   ERes2Net   |    6.6    |        Flank         | UrbanSound8K |      10      | 0.94292  |
|     TDNN     |    2.6    |        Flank         | UrbanSound8K |      10      | 0.93977  |
| PANNS（CNN10） |    5.2    |        Flank         | UrbanSound8K |      10      | 0.92954  |
|   Res2Net    |    5.0    |        Flank         | UrbanSound8K |      10      | 0.92580  |

## Installation Environment

 - The GPU version of Pytorch will be installed first, please skip it if you already have it installed.
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

 - Install macls.
 
Install it using pip with the following command:
```shell
python -m pip install macls -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Source installation is recommended**, which ensures that the latest code is used.
```shell
git clone https://github.com/yeyupiaoling/AudioClassification_Pytorch.git
cd AudioClassification_Pytorch/
python setup.py install
```

## Preparing Data

The `audio_path` is the audio file path. The user needs to store the audio dataset in the `dataset/audio` directory in advance. Each folder stores a category of audio data, and the length of each audio data is more than 3 seconds. For example, `dataset/audio/ bird song /······`. `audio` is where the data list is stored, and the format of the generated data category is`audio_path\tcategory_label_audio`, and the audio path and label are separated by a TAB character `\t`. You can also modify the following functions depending on how you store your data:

Taking Urbansound8K as an example, it is a widely used public dataset for automatic urban environmental sound classification research. Urbansound8K contains 10 categories: air condition sound, car whistle sound, children playing sound, dog bark, drilling sound, engine idling sound, gun sound, jackdrill, siren sound, and street music sound. Data set download address: [UrbanSound8K](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz). Here is the function to generate a list of data for Urbansound8K. If you want to use this dataset, please download and unzip it into the `dataset` directory and change the code to generate the list of data as follows.

`create_data.py` can be used to generate a list of data sets. There are many ways to generate a list of data sets.
```shell
python create_data.py
```

The resulting list looks like this, with the path to the audio followed by the tag for that audio, starting at 0, and separated by `\t`.
```shell
dataset/UrbanSound8K/audio/fold2/104817-4-0-2.wav	4
dataset/UrbanSound8K/audio/fold9/105029-7-2-5.wav	7
dataset/UrbanSound8K/audio/fold3/107228-5-0-0.wav	5
dataset/UrbanSound8K/audio/fold4/109711-3-2-4.wav	3
```

# Change preprocessing methods

By default, the Fbank preprocessing method is used in the configuration file. If you want to use other preprocessing methods, you can modify the following installation in the configuration file, and the specific value can be modified according to your own situation. If it's not clear how to set the parameters, you can remove that section and just use the default values.

```yaml
# 数据预处理参数
preprocess_conf:
  # 是否使用HF上的Wav2Vec2类似模型提取音频特征
  use_hf_model: False
  # 音频预处理方法，也可以叫特征提取方法
  # 当use_hf_model为False时，支持：MelSpectrogram、Spectrogram、MFCC、Fbank
  # 当use_hf_model为True时，指定的是HuggingFace的模型或者本地路径，比如facebook/w2v-bert-2.0或者./feature_models/w2v-bert-2.0
  feature_method: 'Fbank'
  # 当use_hf_model为False时，设置API参数，更参数查看对应API，不清楚的可以直接删除该部分，直接使用默认值。
  # 当use_hf_model为True时，可以设置参数use_gpu，指定是否使用GPU提取特征
  method_args:
    sample_frequency: 16000
    num_mel_bins: 80
```

## 训练

Now we can train the model. We will create `train.py`. The parameters in the configuration file generally do not need to be modified, but these few need to be adjusted according to your actual dataset. The first and most important is the class size `dataset_conf.num_class`, which may be different for each dataset. Then there is` dataset_conf.batch_size `, which can be reduced if memory is insufficient.

```shell
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python train.py
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

Train log:
```
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:14 - ----------- 额外配置参数 -----------
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - configs: configs/ecapa_tdnn.yml
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - local_rank: 0
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - pretrained_model: None
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - resume_model: None
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - save_model_path: models/
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:16 - use_gpu: True
[2023-08-07 22:54:22.148973 INFO   ] utils:print_arguments:17 - ------------------------------------------------
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:19 - ----------- 配置文件参数 -----------
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:22 - dataset_conf:
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:25 - 	aug_conf:
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		noise_aug_prob: 0.2
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		noise_dir: dataset/noise
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		speed_perturb: True
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		volume_aug_prob: 0.2
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		volume_perturb: False
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:25 - 	dataLoader:
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		batch_size: 64
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		num_workers: 4
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:29 - 	do_vad: False
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:25 - 	eval_conf:
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		batch_size: 1
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		max_duration: 20
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:29 - 	label_list_path: dataset/label_list.txt
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:29 - 	max_duration: 3
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:29 - 	min_duration: 0.5
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:29 - 	sample_rate: 16000
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:25 - 	spec_aug_args:
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		freq_mask_width: [0, 8]
[2023-08-07 22:54:22.202166 INFO   ] utils:print_arguments:27 - 		time_mask_width: [0, 10]
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:29 - 	target_dB: -20
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:29 - 	test_list: dataset/test_list.txt
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:29 - 	train_list: dataset/train_list.txt
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:29 - 	use_dB_normalization: True
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:29 - 	use_spec_aug: True
[2023-08-07 22:54:22.203167 INFO   ] utils:print_arguments:22 - model_conf:
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	num_class: 10
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	pooling_type: ASP
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:22 - optimizer_conf:
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	learning_rate: 0.001
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	optimizer: Adam
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	scheduler: WarmupCosineSchedulerLR
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:25 - 	scheduler_args:
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:27 - 		max_lr: 0.001
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:27 - 		min_lr: 1e-05
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:27 - 		warmup_epoch: 5
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	weight_decay: 1e-06
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:22 - preprocess_conf:
[2023-08-07 22:54:22.207167 INFO   ] utils:print_arguments:29 - 	feature_method: Fbank
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:25 - 	method_args:
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:27 - 		num_mel_bins: 80
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:27 - 		sample_frequency: 16000
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:22 - train_conf:
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:29 - 	log_interval: 10
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:29 - 	max_epoch: 30
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:31 - use_model: EcapaTdnn
[2023-08-07 22:54:22.208167 INFO   ] utils:print_arguments:32 - ------------------------------------------------
[2023-08-07 22:54:22.213166 WARNING] trainer:__init__:67 - Windows系统不支持多线程读取数据，已自动关闭！
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
EcapaTdnn                                [1, 10]                   --
├─Conv1dReluBn: 1-1                      [1, 512, 98]              --
│    └─Conv1d: 2-1                       [1, 512, 98]              204,800
│    └─BatchNorm1d: 2-2                  [1, 512, 98]              1,024
├─Sequential: 1-2                        [1, 512, 98]              --
│    └─Conv1dReluBn: 2-3                 [1, 512, 98]              --
│    │    └─Conv1d: 3-1                  [1, 512, 98]              262,144
│    │    └─BatchNorm1d: 3-2             [1, 512, 98]              1,024
│    └─Res2Conv1dReluBn: 2-4             [1, 512, 98]              --
│    │    └─ModuleList: 3-15             --                        (recursive)
│    │    └─ModuleList: 3-16             --                        (recursive)
│    │    └─ModuleList: 3-15             --                        (recursive)
│    │    └─ModuleList: 3-16             --                        (recursive)
│    │    └─ModuleList: 3-15             --                        (recursive)
│    │    └─ModuleList: 3-16             --                        (recursive)
│    │    └─ModuleList: 3-15             --                        (recursive)
│    │    └─ModuleList: 3-16             --                        (recursive)
│    │    └─ModuleList: 3-15             --                        (recursive)
│    │    └─ModuleList: 3-16             --                        (recursive)
···································
│    │    └─ModuleList: 3-56             --                        (recursive)
│    │    └─ModuleList: 3-55             --                        (recursive)
│    │    └─ModuleList: 3-56             --                        (recursive)
│    │    └─ModuleList: 3-55             --                        (recursive)
│    │    └─ModuleList: 3-56             --                        (recursive)
│    └─Conv1dReluBn: 2-13                [1, 512, 98]              --
│    │    └─Conv1d: 3-57                 [1, 512, 98]              262,144
│    │    └─BatchNorm1d: 3-58            [1, 512, 98]              1,024
│    └─SE_Connect: 2-14                  [1, 512, 98]              --
│    │    └─Linear: 3-59                 [1, 256]                  131,328
│    │    └─Linear: 3-60                 [1, 512]                  131,584
├─Conv1d: 1-5                            [1, 1536, 98]             2,360,832
├─AttentiveStatsPool: 1-6                [1, 3072]                 --
│    └─Conv1d: 2-15                      [1, 128, 98]              196,736
│    └─Conv1d: 2-16                      [1, 1536, 98]             198,144
├─BatchNorm1d: 1-7                       [1, 3072]                 6,144
├─Linear: 1-8                            [1, 192]                  590,016
├─BatchNorm1d: 1-9                       [1, 192]                  384
├─Linear: 1-10                           [1, 10]                   1,930
==========================================================================================
Total params: 6,188,490
Trainable params: 6,188,490
Non-trainable params: 0
Total mult-adds (M): 470.96
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 10.28
Params size (MB): 24.75
Estimated Total Size (MB): 35.07
==========================================================================================
[2023-08-07 22:54:26.726095 INFO   ] trainer:train:344 - 训练数据：8644
[2023-08-07 22:54:30.092504 INFO   ] trainer:__train_epoch:296 - Train epoch: [1/30], batch: [0/4], loss: 2.57033, accuracy: 0.06250, learning rate: 0.00001000, speed: 19.02 data/sec, eta: 0:06:43
```

# Eval

At the end of each training round, we can perform an evaluation, which will output the accuracy. We also save the mixture matrix image, and save the path `output/images/` as follows.
![混合矩阵](docs/images/image1.png)

# Inference

At the end of the training, we are given a model parameter file, and we use this model to predict the audio.

```shell
python infer.py --audio_path=dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav
```

# Other Functions

 - In order to read the recorded data and make a dataset easily, we provide the recording program `record_audio.py`, which is used to record audio with a sample rate of 16,000, single channel, 16bit.

```shell
python record_audio.py
```

 - `infer_record.py`This program is used to continuously perform recording recognition, and we can roughly understand this program as recording recognition in real time. And this should allow us to do some interesting things, like put a microphone in a place where birds often come, and recognize it by recording it in real time, and once you recognize that there's a bird calling, if your dataset is powerful enough, and you have a dataset of every bird calling, then you can identify exactly which bird is calling. If the target bird is identified, the procedure is initiated, such as taking photos, etc.

```shell
python infer_record.py --record_seconds=3
```

# Reference

1. https://github.com/PaddlePaddle/PaddleSpeech
2. https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets
3. https://github.com/yeyupiaoling/PPASR
4. https://github.com/alibaba-damo-academy/3D-Speaker
