import argparse
import functools

import librosa
import numpy as np
import torch

from utils.ecapa_tdnn import EcapaTdnn
from utils.reader import load_audio
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path',       str,    'dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav', '图片路径')
add_arg('num_classes',      int,    10,                        '分类的类别数量')
add_arg('label_list_path',  str,    'dataset/label_list.txt',  '标签列表路径')
add_arg('model_path',       str,    'models/model.pth',        '模型保存的路径')
args = parser.parse_args()


# 获取分类标签
with open(args.label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device("cuda")
model = EcapaTdnn(num_classes=args.num_classes)
model.to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()


def infer():
    data = load_audio(args.audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output, dim=-1)
    result = result.data.cpu().numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    print(f'音频：{args.audio_path} 的预测结果标签为：{class_labels[lab]}')


if __name__ == '__main__':
    infer()
