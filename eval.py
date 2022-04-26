import argparse
import functools

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.ecapa_tdnn import EcapaTdnn
from utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'dataset/label_list.txt', '标签列表路径')
add_arg('model_path',       str,    'models/model.pth',       '模型保存的路径')
args = parser.parse_args()


def evaluate():
    test_dataset = CustomDataset(args.test_list_path, model='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    # 获取分类标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]
    # 获取模型
    device = torch.device("cuda")
    model = EcapaTdnn(num_classes=args.num_classes)
    model.to(device)
    summary(model, (80, 98))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    accuracies, preds, labels = [], [], []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.numpy()
        output = model(spec_mag)
        output = output.data.cpu().numpy()
        pred = np.argmax(output, axis=1)
        preds.extend(pred.tolist())
        labels.extend(label.tolist())
        acc = np.mean((pred == label).astype(int))
        accuracies.append(acc.item())
    acc = float(sum(accuracies) / len(accuracies))
    cm = confusion_matrix(labels, preds)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # 精确率
    precision = TP / (TP + FP + 1e-6)
    # 召回率
    recall = TP / (TP + FN + 1e-6)
    print('分类准确率: {:.4f}, 平均精确率: {:.4f}, 平均召回率: {:.4f}'.format(acc, np.mean(precision), np.mean(recall)))
    plot_confusion_matrix(cm=cm, save_path='log/混淆矩阵_eval.png', class_labels=class_labels)


if __name__ == '__main__':
    print_arguments(args)
    evaluate()
