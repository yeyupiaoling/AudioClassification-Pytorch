import random
import sys
from datetime import datetime

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


# 加载并预处理音频
def load_audio(audio_path, mode='train', sr=16000, chunk_duration=3):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    if mode == 'train':
        # 随机裁剪
        num_wav_samples = wav.shape[0]
        # 数据太短不利于训练
        if num_wav_samples < sr:
            raise Exception(f'音频长度不能小于1s，实际长度为：{(num_wav_samples / sr):.2f}s')
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            # 对每次都满长度的再次裁剪
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 2)] = 0
                wav = wav[:-random.randint(1, sr // 2)]
    elif mode == 'eval':
        # 为避免显存溢出，只裁剪指定长度
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
    features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    # 归一化
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    features = features.astype('float32')
    return features


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, model='train', sr=16000, chunk_duration=3):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.sr = sr
        self.chunk_duration = chunk_duration

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            spec_mag = load_audio(audio_path, mode=self.model, sr=self.sr, chunk_duration=self.chunk_duration)
            return spec_mag, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:, :]
    labels = np.array(labels, dtype='int64')
    # 打乱数据
    return torch.tensor(inputs), torch.tensor(labels)
