import os

import numpy as np
import torch
import yaml

from macls import SUPPORT_MODEL
from macls.data_utils.audio import AudioSegment
from macls.data_utils.featurizer import AudioFeaturizer
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE
from macls.models.tdnn import TDNN
from macls.utils.logger import setup_logger
from macls.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class MAClsPredictor:
    def __init__(self,
                 configs,
                 model_path='models/EcapaTdnn_MelSpectrogram/best_model/',
                 use_gpu=True):
        """
        声音分类预测工具
        :param configs: 配置参数
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_conf=self.configs.feature_conf, **self.configs.preprocess_conf)
        self.audio_featurizer.to(self.device)
        # 获取模型
        if self.configs.use_model == 'EcapaTdnn':
            self.predictor = EcapaTdnn(input_size=self.audio_featurizer.feature_dim,
                                       num_class=self.configs.dataset_conf.num_class,
                                       **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN6':
            self.predictor = PANNS_CNN6(input_size=self.audio_featurizer.feature_dim,
                                        num_class=self.configs.dataset_conf.num_class,
                                        **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN10':
            self.predictor = PANNS_CNN10(input_size=self.audio_featurizer.feature_dim,
                                         num_class=self.configs.dataset_conf.num_class,
                                         **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN14':
            self.predictor = PANNS_CNN14(input_size=self.audio_featurizer.feature_dim,
                                         num_class=self.configs.dataset_conf.num_class,
                                         **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Net':
            self.predictor = Res2Net(input_size=self.audio_featurizer.feature_dim,
                                     num_class=self.configs.dataset_conf.num_class,
                                     **self.configs.model_conf)
        elif self.configs.use_model == 'ResNetSE':
            self.predictor = ResNetSE(input_size=self.audio_featurizer.feature_dim,
                                      num_class=self.configs.dataset_conf.num_class,
                                      **self.configs.model_conf)
        elif self.configs.use_model == 'TDNN':
            self.predictor = TDNN(input_size=self.audio_featurizer.feature_dim,
                                  num_class=self.configs.dataset_conf.num_class,
                                  **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        self.predictor.to(self.device)
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pt')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')
        self.predictor.load_state_dict(model_state_dict)
        print(f"成功加载模型参数：{model_path}")
        self.predictor.eval()
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]

    # 预测一个音频的特征
    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频

        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 结果标签和对应的得分
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            input_data = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            input_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            input_data = AudioSegment.from_wave_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        # 重采样
        if input_data.sample_rate != self.configs.dataset_conf.sample_rate:
            input_data.resample(self.configs.dataset_conf.sample_rate)
        # decibel normalization
        if self.configs.dataset_conf.use_dB_normalization:
            input_data.normalize(target_db=self.configs.dataset_conf.target_dB)
        if 'panns' in self.configs.use_model:
            # 对小于训练长度的复制补充
            num_chunk_samples = int(self.configs.dataset_conf.chunk_duration * input_data.sample_rate)
            if input_data.num_samples < num_chunk_samples:
                shortage = num_chunk_samples - input_data.num_samples
                input_data.pad_silence(duration=float(shortage / input_data.sample_rate * 1.1), sides='end')
            # 裁剪需要的数据
            input_data.crop(duration=self.configs.dataset_conf.chunk_duration)
        input_data = torch.tensor(input_data.samples, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio_feature = self.audio_featurizer(input_data)
        # 执行预测
        output = self.predictor(audio_feature)
        result = torch.nn.functional.softmax(output, dim=-1)[0]
        result = result.data.cpu().numpy()
        # 最大概率的label
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data):
        """预测一批音频

        :param audios_data: 经过预处理的一批数据
        :return: 结果标签和对应的得分
        """
        # 找出音频长度最长的
        batch = sorted(audios_data, key=lambda a: a.shape[0], reverse=True)
        freq_size = batch[0].shape[1]
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
        for i, sample in enumerate(batch):
            seq_length = sample.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[i, :seq_length, :] = sample[:, :]
        audios_data = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # 执行预测
        output = self.predictor(audios_data)
        results = torch.nn.functional.softmax(output, dim=-1)
        results = results.data.cpu().numpy()
        labels, scores = [], []
        for result in results:
            lab = np.argsort(result)[-1]
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))
        return labels, scores
