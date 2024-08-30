import distutils.util
import os

import matplotlib.pyplot as plt
import numpy as np

from loguru import logger


def print_arguments(args=None, configs=None, title=None):
    if args:
        logger.info("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
    if configs:
        title = title if title else "配置文件参数"
        logger.info(f"----------- {title} -----------")
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info("\t\t%s: %s" % (a1, v1))
                    else:
                        logger.info("\t%s: %s" % (a, v))
            else:
                logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def plot_confusion_matrix(cm, save_path, class_labels, show=False):
    """
    绘制混淆矩阵
    @param cm: 混淆矩阵, 一个二维数组，表示预测结果与真实结果的混淆情况。
    @param save_path: 保存路径, 字符串，指定混淆矩阵图像的保存位置。
    @param class_labels: 类别名称, 一个列表，包含各个类别的名称。
    @param show: 是否显示图像, 布尔值，控制是否在绘图窗口显示混淆矩阵图像。
    """
    # 检测类别名称是否包含中文，是则设置相应字体
    s = ''.join(class_labels)
    is_ascii = all(ord(c) < 128 for c in s)
    if not is_ascii:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    # 初始化绘图参数并绘制混淆矩阵
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中绘制每个格子的概率值
    ind_array = np.arange(len(class_labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val] / (np.sum(cm[:, x_val]) + 1e-6)
        # 忽略概率值太小的格子
        if c < 1e-4: continue
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    m = np.sum(cm, axis=0) + 1e-6
    plt.imshow(cm / m, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion Matrix' if is_ascii else '混合矩阵')
    plt.colorbar()
    # 设置类别标签
    xlocations = np.array(range(len(class_labels)))
    plt.xticks(xlocations, class_labels, rotation=90)
    plt.yticks(xlocations, class_labels)
    plt.ylabel('Actual label' if is_ascii else '实际标签')
    plt.xlabel('Predict label' if is_ascii else '预测标签')

    # 调整刻度标记位置，提高可视化效果
    tick_marks = np.array(range(len(class_labels))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    if show:
        # 显示图片
        plt.show()
