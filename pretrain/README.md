<div align='center'>
    <h1>HeltonX♾️</h1>
    <img src="https://github.com/Scienthusiasts/heltonx/blob/main/demo/logo.png"/>
    <p><em>致力于 0~1 实现通用深度学习框架，基于 Pytorch，支持各类下游任务，不断完善中 ~</em></p>
</div>


## ✒️`pretrain` 设计逻辑

```
pretrain:
├─configs  (自定义模型配置参数文件)
├─datasets (自定义Dataset和数据增强)
├─losses   (自定义损失函数)
├─models         (自定义网络组件)
│  ├─backbones   (自定义骨干网络)
│  ├─classifiers (自定义分类器)
│  ├─heads       (自定义任务头)
│  └─vfms        (视觉基础模型)
├─utils             (实现任务特定的相关功能)
│  ├─eval_utils.py  (评估pipeline, 和具体任务有关)
│  └─metrics.py     (评估指标计算, 和具体任务有关)
└─tools    
   ├─train.py            (训练pipeline)
   ├─train_accelerate.py (基于Accelerator库的训练pipeline)
   ├─eval.py             (评估pipeline)
   ├─test.py             (测试相关逻辑, 完善中)
   ├─run.sh              (DDP训练脚本)
   └─run_accelerate.sh   (基于Accelerator库的DDP训练脚本)
```
