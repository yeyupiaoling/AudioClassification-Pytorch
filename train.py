import argparse
import functools
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.ecapa_tdnn import EcapaTdnn
from utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, plot_confusion_matrix

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    30,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('label_list_path',   str,   'dataset/label_list.txt', '标签列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
args = parser.parse_args()


# 评估模型
@torch.no_grad()
def evaluate(model, test_loader, device):
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
    model.train()
    acc = float(sum(accuracies) / len(accuracies))
    cm = confusion_matrix(labels, preds)
    return acc, cm


def train(args):
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

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

    # 获取优化方法
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)
    # 获取学习率衰减函数
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    # 恢复训练
    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model.pth')))
        state = torch.load(os.path.join(args.resume, 'model.state'))
        last_epoch = state['last_epoch']
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    # 获取损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        accuracies = []
        for batch_id, (spec_mag, label) in enumerate(train_loader):
            spec_mag = spec_mag.to(device)
            label = label.to(device).long()
            output = model(spec_mag)
            # 计算损失值
            los = loss(output, label)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            if batch_id % 100 == 0:
                print(f'[{datetime.now()}] Train epoch [{epoch}/{args.num_epoch}], batch: {batch_id}/{len(train_loader)}, '
                      f'lr: {scheduler.get_last_lr()[0]:.8f}, loss: {sum(loss_sum) / len(loss_sum):.8f}, '
                      f'accuracy: {sum(accuracies) / len(accuracies):.8f}')
        scheduler.step()
        # 评估模型
        acc, cm = evaluate(model, test_loader, device)
        plot_confusion_matrix(cm=cm, save_path=f'log/混淆矩阵_{epoch}.png', class_labels=class_labels, show=False)
        print('='*70)
        print(f'[{datetime.now()}] Test {epoch}, accuracy: {acc}')
        print('='*70)
        # 保存模型
        os.makedirs(args.save_model, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_model, 'model.pth'))
        torch.save({'last_epoch': torch.tensor(epoch)}, os.path.join(args.save_model, 'model.state'))
        torch.save(optimizer.state_dict(), os.path.join(args.save_model, 'optimizer.pth'))


if __name__ == '__main__':
    print_arguments(args)
    train(args)
