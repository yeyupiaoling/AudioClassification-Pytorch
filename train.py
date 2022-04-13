import argparse
import functools
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from resnet import resnet34
from torch.optim.lr_scheduler import StepLR
from reader import CustomDataset
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('lr_step',          int,    10,                       '学习率衰减步数')
add_arg('input_shape',      str,    '(None, 1, 128, 128)',    '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
args = parser.parse_args()


# 评估模型
def test(model, test_loader, device):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.to(device).long()
        output = model(spec_mag)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc.item())
    model.train()
    return float(sum(accuracies) / len(accuracies))


def train(args):
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train', spec_len=input_shape[3])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[3])
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 获取模型
    device = torch.device("cuda")
    model = resnet34(num_classes=args.num_classes)
    model.to(device)

    # 获取优化方法
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)
    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.8, verbose=True)

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
            output = torch.nn.functional.softmax(output)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            if batch_id % 100 == 0:
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f' % (
                    datetime.now(), epoch, batch_id, len(train_loader), sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies)))
        scheduler.step()
        # 评估模型
        acc = test(model, test_loader, device)
        print('='*70)
        print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
        print('='*70)
        model_path = os.path.join(args.save_model, 'resnet34.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.jit.save(torch.jit.script(model), model_path)


if __name__ == '__main__':
    print_arguments(args)
    train(args)
