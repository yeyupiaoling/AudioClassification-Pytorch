import os
import platform
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
from loguru import logger
from visualdl import LogWriter

from macls.data_utils.collate_fn import collate_fn
from macls.data_utils.featurizer import AudioFeaturizer
from macls.data_utils.reader import MAClsDataset
from macls.metric.metrics import accuracy
from macls.models import build_model
from macls.optimizer import build_optimizer, build_lr_scheduler
from macls.utils.checkpoint import load_pretrained, load_checkpoint, save_checkpoint
from macls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments

# by placebeyondtheclouds
import mlflow
import mlflow.pytorch
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# by placebeyondtheclouds

class MAClsTrainer(object):
    def __init__(self, configs, use_gpu=True, data_augment_configs=None):
        """ macls集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        :param data_augment_configs: 数据增强配置字典或者其文件路径
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        # 读取数据增强配置文件
        if isinstance(data_augment_configs, str):
            with open(data_augment_configs, 'r', encoding='utf-8') as f:
                data_augment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=data_augment_configs, title='数据增强配置')
        self.data_augment_configs = dict_to_object(data_augment_configs)
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
        if self.configs.preprocess_conf.get('use_hf_model', False):
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('使用HuggingFace模型不支持多线程进行特征提取，已自动关闭！')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False



        # by placebeyondtheclouds
        self.experiment_name = self.configs.mlflow_experiment_name
        self.mlflow_uri = self.configs.mlflow_uri
        self.mlflow_training_parameters = {
                "Batch Size": self.configs.dataset_conf.dataLoader.batch_size,
                "train_loader_num_workers": self.configs.dataset_conf.dataLoader.num_workers,
                "Epochs": self.configs.train_conf.max_epoch,
                "Automatic Mixed Precision": self.configs.train_conf.enable_amp,
                "Optimizer": self.configs.optimizer_conf.optimizer,
                "Learning Rate": self.configs.optimizer_conf.optimizer_args.lr,
                "weight_decay": self.configs.optimizer_conf.optimizer_args.weight_decay,
                "scheduler": self.configs.optimizer_conf.scheduler,
                "scheduler_args.min_lr": self.configs.optimizer_conf.scheduler_args.min_lr,
                "scheduler_args.max_lr": self.configs.optimizer_conf.scheduler_args.max_lr,
                "scheduler_args.warmup_epoch": self.configs.optimizer_conf.scheduler_args.warmup_epoch,
                "Model": self.configs.model_conf.model,
                "feature_method": self.configs.preprocess_conf.feature_method,
                "use_dB_normalization": self.configs.dataset_conf.dataset.use_dB_normalization,
                "speed_perturb_prob": self.data_augment_configs.speed.prob,
                "volume_perturb_prob": self.data_augment_configs.volume.prob,
                "noise_aug_prob": self.data_augment_configs.noise.prob,
                "reverb_aug_prob": self.data_augment_configs.reverb.prob,
                "spec_aug_prob": self.data_augment_configs.spec_aug.prob,
                "data_raw_hours": self.configs.data_description.data_raw_hours,
                "data_used_to_train": self.configs.data_description.data_used_to_train,
                "data_cut_overlap": self.configs.data_description.data_cut_overlap,
                "data_preprocessing": self.configs.data_description.data_preprocessing,
                "data_filtering": self.configs.data_description.data_filtering,
                "data_oversample": self.configs.data_description.data_oversample,
                "data_undersample": self.configs.data_description.data_undersample,
                "test_size": self.configs.data_description.test_size,
                "experiment_run": self.configs.experiment_run,
                "train_audio_files_number": self.configs.data_description.train_audio_files_number,
                "train_audio_files_hours": self.configs.data_description.train_audio_files_hours,
                "comment": self.configs.data_description.comment,
                "model_growth_rate": self.configs.model_conf.growth_rate,
                "train_max_duration": self.configs.dataset_conf.dataset.max_duration,
                "train_min_duration": self.configs.dataset_conf.dataset.min_duration,
                "test_max_duration": self.configs.dataset_conf.eval_conf.max_duration,
                # "test_min_duration": self.configs.dataset_conf.eval_conf.min_duration,
                # "val_max_duration": self.configs.dataset_conf.val_conf.max_duration,
                # "val_min_duration": self.configs.dataset_conf.val_conf.min_duration,
                }
        self.eval_results_all = []
         # by placebeyondtheclouds

    def __setup_dataloader(self, is_train=False):
        """ 获取数据加载器

        :param is_train: 是否获取训练数据
        """
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        if is_train:
            self.train_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.train_list,
                                              audio_featurizer=self.audio_featurizer,
                                              aug_conf=self.data_augment_configs,
                                              mode='train',
                                              **dataset_args)
            # 设置支持多卡训练
            train_sampler = RandomSampler(self.train_dataset)
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           sampler=train_sampler,
                                           **data_loader_args)
        # 获取测试数据
        data_loader_args.drop_last = False
        dataset_args.max_duration = self.configs.dataset_conf.eval_conf.max_duration
        data_loader_args.batch_size = self.configs.dataset_conf.eval_conf.batch_size
        self.test_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.test_list,
                                         audio_featurizer=self.audio_featurizer,
                                         mode='eval',
                                         **dataset_args)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)
        # by placebeyondtheclouds
        self.val_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.validation_list,
                                         audio_featurizer=self.audio_featurizer,
                                         mode='val',
                                         **dataset_args)
        self.val_loader = DataLoader(dataset=self.val_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)
        # by placebeyondtheclouds

    def extract_features(self, save_dir='dataset/features', max_duration=100):
        """ 提取特征保存文件

        :param save_dir: 保存路径
        :param max_duration: 提取特征的最大时长，避免过长显存不足，单位秒
        """
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        for i, data_list in enumerate([self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]):
            # 获取测试数据
            dataset_args = self.configs.dataset_conf.get('dataset', {})
            dataset_args.max_duration = max_duration
            test_dataset = MAClsDataset(data_list_path=data_list,
                                        audio_featurizer=self.audio_featurizer,
                                        mode='extract_feature',
                                        **dataset_args)
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    feature, label = test_dataset[i]
                    feature = feature.numpy()
                    label = int(label)
                    save_path = os.path.join(save_dir, str(label), f'{int(time.time() * 1000)}.npy').replace('\\', '/')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label}\n')
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')

    def __setup_model(self, input_size, is_train=False):
        """ 获取模型

        :param input_size: 模型输入特征大小
        :param is_train: 是否获取训练模型
        """
        # 自动获取列表数量
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)
        # 获取模型
        self.model = build_model(input_size=input_size, configs=self.configs)
        # 打印模型信息，98是长度，这个取决于输入的音频长度
        self.model.to(self.device)
        summary(self.model, input_size=(1, 98, input_size))
        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        # print(self.model)
        # 获取损失函数
        label_smoothing = self.configs.train_conf.get('label_smoothing', 0.0)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.GradScaler(init_scale=1024)
            # 获取优化方法
            self.optimizer = build_optimizer(params=self.model.parameters(), configs=self.configs)
            # 学习率衰减函数
            self.scheduler = build_lr_scheduler(optimizer=self.optimizer, step_per_epoch=len(self.train_loader),
                                                configs=self.configs)

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        """训练一个epoch

        :param epoch_id: 当前epoch
        :param local_rank: 当前显卡id
        :param writer: VisualDL对象
        :param nranks: 所使用显卡的数量
        """
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        for batch_id, (features, label, input_len) in enumerate(self.train_loader):
            if self.stop_train: break
            if nranks > 1:
                features = features.to(local_rank)
                label = label.to(local_rank).long()
            else:
                features = features.to(self.device)
                label = label.to(self.device).long()
            # 执行模型计算，是否开启自动混合精度
            with torch.autocast('cuda', enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)
            # 计算损失值
            los = self.loss(output, label)
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算准确率
            acc = accuracy(output, label)
            accuracies.append(acc)
            loss_sum.append(los.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                mlflow.log_metric('Train/Loss', self.train_loss, self.train_log_step) # by placebeyondtheclouds
                mlflow.log_metric('Train/Accuracy', self.train_acc, self.train_log_step) # by placebeyondtheclouds

                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step)
                mlflow.log_metric('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step) # by placebeyondthec
                train_times, accuracies, loss_sum = [], [], []
                self.train_log_step += 1
            start = time.time()
            self.scheduler.step()

    def train(self,
              save_model_path='models/',
              log_dir='log/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param log_dir: 保存VisualDL日志文件的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        # 获取有多少张显卡训练
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # considering multi-node training # by placebeyondtheclouds
            world_rank = os.environ.get('RANK') # by placebeyondtheclouds
            if world_rank is None or world_rank == '0': # by placebeyondtheclouds
                # 日志记录器
                writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化NCCL环境
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])
            
        # by placebeyondtheclouds
        mlflow.set_tracking_uri(self.mlflow_uri)
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment_id is None and local_rank == 0:
            # considering multi-node training
            world_rank = os.environ.get('RANK')
            if world_rank is None or world_rank == '0':
                experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment_id.experiment_id
        #mlflow.set_experiment(self.experiment_name)
        if local_rank == 0:
            # considering multi-node training
            world_rank = os.environ.get('RANK')
            if world_rank is None or world_rank == '0':
                if mlflow.active_run() is None:
                    mlflow.start_run(experiment_id=experiment_id)
                    mlflow.log_params(self.mlflow_training_parameters)
        # by placebeyondtheclouds


        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)
        # 加载预训练模型
        self.model = load_pretrained(model=self.model, pretrained_model=pretrained_model)
        # 加载恢复模型
        self.model, self.optimizer, self.amp_scaler, self.scheduler, last_epoch, best_acc = \
            load_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, scheduler=self.scheduler, step_epoch=len(self.train_loader),
                            save_model_path=save_model_path, resume_model=resume_model)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        if local_rank == 0:
            # considering multi-node training # by placebeyondtheclouds
            world_rank = os.environ.get('RANK') # by placebeyondtheclouds
            if world_rank is None or world_rank == '0': # by placebeyondtheclouds
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
                mlflow.log_metric('Train/lr', self.scheduler.get_last_lr()[0], last_epoch) # by placebeyondtheclouds
        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                # considering multi-node training # by placebeyondtheclouds
                world_rank = os.environ.get('RANK') # by placebeyondtheclouds
                if world_rank is None or world_rank == '0': # by placebeyondtheclouds
                    if self.stop_eval: continue
                    logger.info('=' * 70)
                    self.eval_loss, self.eval_acc, cm_plot_test = self.evaluate(save_plots_mlflow=str(epoch_id).zfill(2)) # by placebeyondtheclouds
                    val_loss, val_acc, result_f1, result_acc, result_eer_fpr, resut_eer_thr, result_eer_fnr, result_roc_auc_score, result_pr_auc, cm_plot, roc_curve_plot = self.validate(save_plots_mlflow=str(epoch_id).zfill(2)) # by placebeyondtheclouds
                    self.eval_results_all.append([epoch_id, self.eval_loss, self.eval_acc, val_loss, val_acc, result_f1, result_acc, result_eer_fpr, resut_eer_thr, result_eer_fnr, result_roc_auc_score, result_pr_auc]) # by placebeyondtheclouds
                    logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                        epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc))
                    logger.info('=' * 70)
                    writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                    writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                    # by placebeyondtheclouds
                    mlflow.log_metric('Test/Accuracy', self.eval_acc, self.test_log_step) 
                    mlflow.log_metric('Test/Loss', self.eval_loss, self.test_log_step) 
                    mlflow.log_metric('Val/loss', val_loss, self.test_log_step) 
                    mlflow.log_metric('Val/acc', val_acc, self.test_log_step) 
                    mlflow.log_metric('Val/F1 score', result_f1, self.test_log_step) 
                    mlflow.log_metric('Val/Accuracy', result_acc, self.test_log_step) 
                    mlflow.log_metric('Val/EER-fpr', result_eer_fpr, self.test_log_step) 
                    mlflow.log_metric('Val/EER-threshold', resut_eer_thr, self.test_log_step) 
                    mlflow.log_metric('Val/EER-fnr', result_eer_fnr, self.test_log_step) 
                    mlflow.log_metric('Val/ROC AUC score', result_roc_auc_score, self.test_log_step) 
                    mlflow.log_metric('Val/Precision Recall score', result_pr_auc, self.test_log_step) 
                    mlflow.log_figure(cm_plot, 'val_epoch_'+str(epoch_id).zfill(2)+'_cm.png')
                    mlflow.log_figure(roc_curve_plot, 'val_epoch_'+str(epoch_id).zfill(2)+'_roc_curve.png')
                    mlflow.log_figure(cm_plot_test, 'test_epoch_'+str(epoch_id).zfill(2)+'_cm.png')
                    # by placebeyondtheclouds
                self.test_log_step += 1
                self.model.train()
                # # 保存最优模型
                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                    amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                    accuracy=self.eval_acc, best_model=True)
                # 保存模型
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                accuracy=self.eval_acc)
                # by placebeyondtheclouds
        if local_rank == 0: # will only log the last run if training is restarted
            # considering multi-node training
            world_rank = os.environ.get('RANK')
            if world_rank is None or world_rank == '0':
                if mlflow.active_run():
                    results_all = pd.DataFrame(self.eval_results_all, columns=['epoch_id', 'test_loss', 'test_acc', 'val_loss', 'val_acc', 'result_f1', 'result_acc', 'result_eer_fpr', 'resut_eer_thr', 'result_eer_fnr', 'result_roc_auc_score', 'result_pr_auc']) 
                    fname = f'models/{self.configs.experiment_run}.csv'
                    results_all.to_csv(fname, index=None)
                    mlflow.log_table(data=results_all, artifact_file=f'{self.configs.experiment_run}.json')
                    mlflow.log_artifact(local_path=fname)
                    mlflow.end_run()
                    print(f'finished {self.configs.experiment_run}')

    def evaluate(self, resume_model=None, save_matrix_path=None, save_plots_mlflow=None): # by placebeyondtheclouds
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label, input_lens) in enumerate(tqdm(self.test_loader)):
                if self.stop_eval: break
                features = features.to(self.device)
                label = label.to(self.device).long()
                output = eval_model(features)
                los = self.loss(output, label)
                # 计算准确率
                acc = accuracy(output, label)
                accuracies.append(acc)
                # 模型预测标签
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1
        # 保存混合矩阵
        if save_matrix_path is not None:
            try:
                cm = confusion_matrix(labels, preds)
                plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                      class_labels=self.class_labels)
            except Exception as e:
                logger.error(f'保存混淆矩阵失败：{e}')
        
        # by placebeyondtheclouds
        if save_plots_mlflow is not None:
            cm = metrics.confusion_matrix(labels, preds) # variable name was changed from confusion_matrix to cm
            # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            # fig, ax = plt.subplots(figsize=(4,4))
            # cm_display.plot(ax=ax)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_normalized, display_labels = self.class_labels)
            fig, ax = plt.subplots(figsize=(5,4), dpi=150)
            sns.heatmap(data=cm_normalized, annot=True, annot_kws={"size":22}, fmt='.2f', ax=ax, xticklabels=self.class_labels, yticklabels=self.class_labels)
            fig.suptitle(t=f'\n {self.configs.experiment_run} epoch_{save_plots_mlflow}, test \n loss: {round(loss,2)}, accuracy: {round(acc,2)}', x=0.5, y=1.01)
            plt.ylabel('Actual labels')
            plt.xlabel('Predicted labels')
            fig.savefig(fname='temp.png', bbox_inches='tight', pad_inches=0)
            # mlflow.log_figure(fig, os.path.join('plots', self.configs.experiment_run, 'epoch_'+save_plots_mlflow+'_cm.png'))
            cm_plot_test = fig
            plt.close()
            self.model.train()
            return loss, acc, cm_plot_test
        else:
            self.model.train()
            return loss, acc
        # by placebeyondtheclouds

    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))




    # by placebeyondtheclouds
    def validate(self, resume_model=None, save_plots_mlflow=None):
        if self.val_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model
        
        preds_prob = []
        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label, input_lens) in enumerate(tqdm(self.test_loader)):
                
                features = features.to(self.device)
                label = label.to(self.device).long()
                output = eval_model(features)
                los = self.loss(output, label)
                for one_output in output:
                    result = torch.nn.functional.softmax(one_output, dim=-1)
                    result = result.data.cpu().numpy()
                    preds_prob.append(result)
                # 计算准确率
                acc = accuracy(output, label)
                accuracies.append(acc)
                # 模型预测标签
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1
        # print(f'{labels[:5]=}')
        # print(f'{preds[:5]=}')
        result_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
        result_acc = accuracy_score(y_true=labels, y_pred=preds)            # sanity check, must be the same with acc (the original code calcucations) 
        # print(f'{result_f1=}')
        # print(f'{result_acc=}')
        
        preds_prob = np.array(preds_prob)[:,1]
        # print(f'{preds_prob[:5]=}')
        fpr, tpr, threshold = roc_curve(labels, preds_prob, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER_sanity = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        precision, recall, threshold = precision_recall_curve(labels, preds_prob)

        result_eer_fpr = EER
        resut_eer_thr = eer_threshold
        result_eer_fnr = EER_sanity
        result_roc_auc_score = roc_auc_score(labels, preds_prob)
        result_pr_auc = auc(recall, precision)

        if save_plots_mlflow:
            # cm
            confusion_matrix = metrics.confusion_matrix(labels, preds)
            # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            # fig, ax = plt.subplots(figsize=(4,4))
            # cm_display.plot(ax=ax)
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_normalized, display_labels = self.class_labels)
            fig, ax = plt.subplots(figsize=(5,4), dpi=150)
            sns.heatmap(data=cm_normalized, annot=True, annot_kws={"size":22}, fmt='.2f', ax=ax, xticklabels=self.class_labels, yticklabels=self.class_labels)
            fig.suptitle(t=f'\n {self.configs.experiment_run} epoch_{save_plots_mlflow}, validation \n EER: {round(result_eer_fpr,2)}, F1: {round(result_f1,2)}, Accuracy: {round(result_acc,2)}', x=0.5, y=1.01)
            plt.ylabel('Actual labels')
            plt.xlabel('Predicted labels')
            fig.savefig(fname='temp.png', bbox_inches='tight', pad_inches=0)
            cm_plot = fig
            plt.close()

            # ROC curve
            fig, ax = plt.subplots(figsize=(4,4), dpi=150)
            ns_probs = [0 for _ in range(len(labels))] #no skill data
            ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs) #no skill data
            # fpr, tpr, threshold = roc_curve(labels, preds_prob)
            ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            ax.plot(fpr, tpr, marker='.', label=f'epoch_{save_plots_mlflow}')
            fig.suptitle(t=f'\n {self.configs.experiment_run} epoch_{save_plots_mlflow}, validation \n EER: {round(result_eer_fpr,2)}, F1: {round(result_f1,2)}, Accuracy: {round(result_acc,2)}', x=0.5, y=1.01)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            fig.savefig(fname='temp.png', bbox_inches='tight', pad_inches=0)
            roc_curve_plot = fig
            plt.close()
        
        self.model.train()
        return loss, acc, result_f1, result_acc, result_eer_fpr, resut_eer_thr, result_eer_fnr, result_roc_auc_score, result_pr_auc, cm_plot, roc_curve_plot
    # by placebeyondtheclouds