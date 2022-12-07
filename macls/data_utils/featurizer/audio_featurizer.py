import numpy as np
import torch
from torchaudio.compliance.kaldi import fbank, spectrogram
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC


class AudioFeaturizer(object):
    """音频特征器

    :param sample_rate: 预处理方法
    :type sample_rate: str
    :param feature_conf: 预处理方法的参数
    :type feature_conf: dict
    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    :param use_dB_normalization: 是否对音频进行音量归一化
    :type use_dB_normalization: bool
    :param target_dB: 对音频进行音量归一化的音量分贝值
    :type target_dB: float
    """

    def __init__(self,
                 feature_method='MelSpectrogram',
                 feature_conf={},
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        self._feature_conf = feature_conf
        self._feature_method = feature_method
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**feature_conf)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(**feature_conf)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**feature_conf)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def featurize(self, audio_segment):
        """从AudioSegment中提取音频特征

        :param audio_segment: Audio segment to extract features from.
        :type audio_segment: AudioSegment
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        # upsampling or downsampling
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # 获取音频特征
        waveform = torch.from_numpy(np.expand_dims(audio_segment.samples, 0)).float()
        feature = self.feat_fun(waveform).squeeze(0).transpose(1, 0).numpy()
        # 归一化
        mean = np.mean(feature, 1, keepdims=True)
        std = np.std(feature, 1, keepdims=True)
        feature = (feature - mean) / (std + 1e-5)
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'MelSpectrogram':
            return self._feature_conf.n_mels
        elif self._feature_method == 'Spectrogram':
            return 257
        elif self._feature_method == 'MFCC':
            return self._feature_conf.n_mels
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
