import argparse
import functools
import time

import yaml

from macls.trainer import PPAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/ecapa_tdnn.yml',    "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('save_matrix_path', str,   'output/images/',            "保存混合矩阵的路径")
add_arg('resume_model',     str,   'models/{}_{}/best_model/',  "模型的路径")
args = parser.parse_args()


# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取训练器
trainer = PPAClsTrainer(configs=configs, use_gpu=args.use_gpu)

# 开始评估
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model.format(configs['use_model'],
                                                                        configs['preprocess_conf']['feature_method']),
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('评估消耗时间：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))
