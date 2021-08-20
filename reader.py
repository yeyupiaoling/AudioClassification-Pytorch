import random

import librosa
import numpy as np
from torch.utils.data import Dataset


# 加载并预处理音频
def load_audio(audio_path, mode='train', spec_len=128):
    # 读取音频数据
    wav, sr = librosa.load(audio_path, sr=16000)
    spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
    if mode == 'train':
        crop_start = random.randint(0, spec_mag.shape[1] - spec_len)
        spec_mag = spec_mag[:, crop_start:crop_start + spec_len]
    else:
        spec_mag = spec_mag[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]
    return spec_mag


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, model='train', spec_len=128):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        return spec_mag, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
