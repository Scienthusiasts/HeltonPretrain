# coding=utf-8
import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# 自定义模块
from utils.utils import *
from utils.metricsUtils import *
from utils.runnerUtils import *
from utils.exportUtils import torchExportOnnx, onnxInferenceSingleImg, onnxInferenceBatchImgs






class Runner():
    '''训练/验证/推理时的流程'''
    def __init__(self, 
                 seed:int, 
                 mode:str, 
                 class_names:list, 
                 img_size:list, 
                 epoch:int, 
                 resume:str, 
                 log_dir:str, 
                 log_interval:int, 
                 eval_interval:int, 
                 dataset:dict, 
                 model:dict, 
                 optimizer:dict):
        '''Runner初始化
        Args:
            - mode:            当前模式是训练/验证/推理
            - backbone_name:   骨干网络名称
            - froze_backbone:  训练时是否冻结Backbone
            - img_size:        统一图像尺寸的大小
            - class_names:     数据集类别名称
            - train_json_path: 训练集json文件路径
            - val_json_path:   验证集json文件路径
            - train_img_dir:   训练集图像路径
            - val_img_dir:     验证集图像路径
            - eopch:           训练批次
            - bs:              训练batch size
            - lr:              学习率
            - log_dir:         日志文件保存目录
            - log_interval:    训练或验证时隔多少bs打印一次日志
            - optim_type:      优化器类型
            - load_ckpt:
            - resume:          是否恢复断点训练
            - seed:            固定随机种子
            - map:             数据集类别id映射字典(COCO)(json文件里的id->按顺序的id)
            - reverse_map:     数据集类别id逆映射字典(COCO)(按顺序的id->json文件里的id)

        Returns:
            None
        '''
        self.seed = seed
        self.mode = mode
        self.resume = resume
        # 设置全局种子
        seed_everything(self.seed)
        self.class_names = class_names
        self.cat_nums = len(class_names)
        self.img_size = img_size
        self.epoch = epoch
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        '''GPU/CPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''日志模块'''
        if mode in ['train', 'eval']:
            self.logger, self.log_dir, self.log_save_path = myLogger(self.mode, self.log_dir)
            '''训练/验证时参数记录模块'''
            json_save_dir, _ = os.path.split(self.log_save_path)
            self.argsHistory = ArgsHistory(json_save_dir)
        '''导入数据集'''
        if self.mode not in ['test', 'export']:
            self.train_data, self.train_data_loader, self.val_data, self.val_data_loader = loadDatasets(mode=self.mode, **dataset)
        '''导入网络'''
        # 根据模型名称动态导入模块
        self.model = dynamic_import_class(model.pop('path'), 'Model')(**model).to(self.device)
        # torch.save(self.model.half().state_dict(), "./last_fp16.pt")
        cudnn.benchmark = True
            

        '''定义优化器(自适应学习率的带动量梯度下降方法)'''
        if mode == 'train':
            self.optimizer, self.scheduler = optimSheduler(**optimizer, 
                                                           model=self.model, 
                                                           total_epoch=self.epoch, 
                                                           train_data_loader=self.train_data_loader)
        '''当恢复断点训练'''
        self.start_epoch = 0
        if self.resume and self.mode=='train':
            trainResume(self.resume, self.model, self.optimizer, self.logger, self.argsHistory)
        '''打印训练参数'''
        if self.mode not in ['test', 'export']:
            val_data_len = self.val_data.__len__()
            train_data_len = self.train_data.__len__() if self.mode=='train' else 0
            printRunnerArgs(
                backbone_name=model['backbone_name'], 
                mode=self.mode, 
                logger=self.logger, 
                device=self.device, 
                seed=self.seed, 
                bs=dataset['bs'], 
                img_size=self.img_size, 
                train_data_len=train_data_len, 
                val_data_len=val_data_len, 
                cat_nums=self.cat_nums, 
                optimizer=optimizer)
            # torch.save(self.model.state_dict(), "./tmp.pt")








    def fitBatch(self, step, train_batch_num, epoch, batch_datas):
        '''一个batch的训练流程(前向+反向)

        Args:

        Returns:
            - losses:      所有损失组成的列表
            - total_loss:  所有损失之和
        '''
        # 一个batch的前向传播+计算损失
        losses = self.model.batchLoss(self.device, batch_datas)
        # 将上一次迭代计算的梯度清零
        self.optimizer.zero_grad()
        # 反向传播
        losses['total_loss'].backward()
        # 更新权重
        self.optimizer.step()
        # 更新学习率
        self.scheduler.step(epoch * train_batch_num + step) 

        return losses



    def fitEpoch(self, epoch):
        '''一个epoch的训练
        '''
        self.model.train()
        train_batch_num = len(self.train_data_loader)
        for step, batch_datas in enumerate(self.train_data_loader):
            '''一个batch的训练, 并得到损失'''
            losses = self.fitBatch(step, train_batch_num, epoch, batch_datas)
            '''打印日志'''
            printLog(
                mode='train', 
                log_interval=self.log_interval, 
                logger=self.logger, 
                optimizer=self.optimizer, 
                step=step, 
                epoch=epoch, 
                batch_num=train_batch_num, 
                losses=losses)
            '''记录变量(loss, lr等, 每个iter都记录)'''
            recoardArgs(mode='train', optimizer=self.optimizer, argsHistory=self.argsHistory, loss=losses)




    def valEpoch(self, ckpt_path=False, half=False):
        '''一个epoch的评估(基于验证集)
        '''

        '''是否导入权重'''
        if ckpt_path:
            print(f'ckpt({ckpt_path}) has loaded in val phase!')
            # self.model.load_state_dict(torch.load(ckpt_path))
            self.model = loadWeightsBySizeMatching(self.model, ckpt_path)
        if half:
            self.model.half()
        self.model.eval()
        # 记录真实标签和预测标签
        pred_list, true_list, soft_list = [], [], []
        # 验证时无需计算梯度
        with torch.no_grad():
            for batch_datas in tqdm(self.val_data_loader):
                '''推理一个batch'''
                pred, true, soft = self.valBatch(batch_datas, half)
                # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
                pred_list += pred
                true_list += true
                soft_list += soft
        
        pred_list = np.array(pred_list)
        true_list = np.array(true_list)
        soft_list = np.array(soft_list)
        
        '''评估'''
        # 准确率
        acc = sum(pred_list==true_list) / pred_list.shape[0]
        # # 可视化混淆矩阵
        showComMatrix(true_list, pred_list, self.class_names, self.log_dir)
        # 绘制PR曲线
        PRs = drawPRCurve(self.class_names, true_list, soft_list, self.log_dir)
        # 计算每个类别的 AP, F1Score
        mAP, mF1Score, form = clacAP(PRs, self.class_names)
        self.logger.info(f'\n{form}')

        return acc, mAP, mF1Score



    def valBatch(self, batch_datas, half=False):
        '''一个batch的评估

        '''
        # 半精度推理时数据也得是半精度
        if half: batch_datas = [datas.half() for datas in batch_datas]
        '''一个batch的评估, 并得到batch的评估指标'''
        pred, true, soft = self.model.batchVal(self.device, batch_datas)
        return pred, true, soft



    def trainer(self):
        '''所有epoch的训练流程(训练+验证)
        '''
        for epoch in range(self.start_epoch, self.epoch):
            '''一个epoch的训练'''
            self.fitEpoch(epoch)
            '''以json格式保存args'''
            self.argsHistory.saveRecord()
            '''一个epoch的验证'''
            self.evaler(epoch)
            if epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1):
                '''打印日志(一个epoch结束)'''
                printLog(mode='epoch', logger=self.logger, argsHistory=self.argsHistory, step=0, epoch=epoch)
                '''保存网络权重(一个epoch结束)'''
                saveCkpt(epoch, self.model, self.optimizer, self.scheduler, self.log_dir, self.argsHistory, self.logger)





    def evaler(self, epoch=0, ckpt_path=False, half=False):
        '''一个epoch的验证(验证集)
        '''
        if (epoch % self.eval_interval == 0 and (epoch!=0 or self.eval_interval==1)) or self.mode=='eval':
            '''在验证集上评估并计算AP'''
            # 采用一张图一张图遍历的方式,并生成评估结果json文件
            acc, mAP, mF1Score = self.valEpoch(ckpt_path, half)
            '''记录变量'''
            recoardArgs(mode='epoch', argsHistory=self.argsHistory, acc=acc, mAP=mAP, mF1Score=mF1Score)
            if self.mode == 'eval':         
                printLog(mode='epoch', logger=self.logger, argsHistory=self.argsHistory, step=0, epoch=0)
        else:
            if epoch < self.eval_interval -1:
                acc, mAP, mF1Score = 0,0,0
            else:
                acc = self.argsHistory.args_history_dict['val_acc'][-1]
                mAP = self.argsHistory.args_history_dict['val_mAP'][-1]
                mF1Score = self.argsHistory.args_history_dict['val_mF1Score'][-1]
            '''记录变量'''
            recoardArgs(mode='epoch', argsHistory=self.argsHistory, acc=acc, mAP=mAP, mF1Score=mF1Score)






    def tester(self, test_mode:str, img_path:str, save_vis_path:str, ckpt_path:str, onnx_path=False, img_dir=False, id_img_dir=False, img_pair_paths=False, half=False, tta=False):
        '''把pytorch测试代码独自分装成一个函数

        Args:
            - img_path:     测试图像路径
            - save_res_dir: 推理结果保存目录

        Returns:
            None
        '''
        from datasets.preprocess import Transforms
        tf = Transforms(imgSize = self.img_size)
        # 是否导入预训练权重:
        if ckpt_path:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()
        # 是否半精度推理:
        if half: self.model.half()

        if test_mode == 'clssify_single':
            if tta:
                inferenceSingleImgTTA(self.model, self.device, tf, img_path, save_vis_path, half)
            else:
                inferenceSingleImg(self.model, self.device, tf, img_path, save_vis_path, half)
        if test_mode == 'clssify_batch':
            inferenceBatchImgs(self.model, self.device, tf, img_dir, self.class_names, half, tta)
        elif test_mode == 'identify_all':
            Identify(self.model, self.device, tf, id_img_dir, half)
        elif test_mode == 'identify_pair':
            identifyPair(self.model, self.device, tf, img_pair_paths, half)
        elif test_mode == 'identify_all_by_dynamic_T':
            IdentifyByDynamicT(self.model, self.device, tf, id_img_dir, half)
        elif test_mode == 'onnx_classify_single':
            onnxInferenceSingleImg(model=self.model, onnx_path=onnx_path, device=self.device, tf=tf, img_path=img_path, save_vis_path=save_vis_path)
        elif test_mode == 'onnx_classify_batch':
            onnxInferenceBatchImgs(model=self.model, onnx_path=onnx_path, device=self.device, tf=tf, img_dir=img_dir)





    def exporter(self, export_dir, export_name, ckpt_path):
        '''导出为onnx格式
        '''
        torchExportOnnx(self.model, self.device, self.img_size, export_dir, export_name, ckpt_path)








if __name__ == '__main__':
    args = getArgs()
    config_path = args.config
    # 使用动态导入的模块
    config_file = dynamic_import_class(config_path, get_class=False)
    # 调用参数
    runner_config = config_file.runner
    eval_config = config_file.eval
    test_config = config_file.test
    export_config = config_file.export

    runner = Runner(**runner_config)
    # 训练模式
    if runner_config['mode'] == 'train':
        # 拷贝一份当前训练对应的config文件(方便之后查看细节)
        shutil.copy(config_path, os.path.join(runner.log_dir, 'config.py'))
        runner.trainer()
    # 验证模式
    elif runner_config['mode'] == 'eval':
        runner.evaler(epoch=0, **eval_config)
    # 测试模式
    elif runner_config['mode'] == 'test':
        runner.tester(**test_config)
    elif runner_config['mode'] == 'export':
        runner.exporter(**export_config)
    else:
        print("mode not valid. it must be 'train', 'eval', 'test' or 'export'.")