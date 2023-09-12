import argparse
import functools
import time

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/cam++.yml',         "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('save_matrix_path', str,   'output/images/',            "保存混合矩阵的路径")
add_arg('resume_model',     str,   'models/CAMPPlus_Fbank/best_model/',  "模型的路径")
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 开始评估
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('评估消耗时间：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))
