import argparse
import functools
import wave

import numpy as np
import pyaudio
import torch

from data_utils.reader import load_audio, CustomDataset
from modules.ecapa_tdnn import EcapaTdnn
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('num_classes',      int,    10,                        '分类的类别数量')
add_arg('label_list_path',  str,    'dataset/label_list.txt',  '标签列表路径')
add_arg('model_path',       str,    'output/models/model.pth','模型保存的路径')
add_arg('feature_method',   str,    'melspectrogram',          '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
args = parser.parse_args()


train_dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取分类标签
with open(args.label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device("cuda")
if args.use_model == 'ecapa_tdnn':
    model = EcapaTdnn(num_classes=args.num_classes, input_size=train_dataset.input_size)
else:
    raise Exception(f'{args.use_model} 模型不存在!')
model.to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "infer_audio.wav"

# 打开录音
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# 获取录音数据
def record_audio():
    print("开始录音......")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音已结束!")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


# 预测
def infer(audio_path):
    data = load_audio(audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output, dim=-1)
    result = result.data.cpu().numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    return class_labels[lab]


if __name__ == '__main__':
    try:
        while True:
            # 加载数据
            audio_path = record_audio()
            # 获取预测结果
            label = infer(audio_path)
            print(f'预测的标签为：{label}')
    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
