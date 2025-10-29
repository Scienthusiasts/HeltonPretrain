<div align='center'>
    <h1>HeltonX♾️</h1>
    <img src="https://github.com/Scienthusiasts/heltonx/blob/main/demo/logo.png"/>
    <p><em>致力于 0~1 实现通用深度学习框架，基于 Pytorch，支持各类下游任务，不断完善中 ~</em></p>
</div>


## ✒️`detection` 设计逻辑

```
detection:
├─configs  (自定义模型配置参数文件)
├─datasets (自定义Dataset和数据增强)
├─losses   (自定义损失函数)
├─models         (自定义网络组件)
│  ├─backbones   (自定义骨干网络)
│  ├─necks       (各种fpn)
│  ├─dense_heads (一阶段检测器的检测头或二阶段检测器的RPN)
│  ├─roi_heads   (二阶段检测器的检测头)
│  └─detectors   (完整的检测器)
├─assigners.py      (标签分配策略)
├─bbox_coders.py    (回归框编码解码策略)
├─utils             (实现任务特定的相关功能)
│  ├─eval_utils.py  (检测任务评估pipeline)
│  ├─metrics.py     (检测任务评估指标计算)
│  ├─nms.py         (非极大值抑制)
│  └─utils.py       (检测任务可能用到的功能)
├─demos    (测试用图像)
└─tools    (同pretrain)
```
