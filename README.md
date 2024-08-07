# HeltonPretrain

从0实现图像分类/预训练框架，正在完善中~



**update 24/6/16**:

- 支持经典图像分类预训练、CLIP蒸馏+对比学习等视觉预训练任务
- 兼容`timm`支持的Backbone，可任意更换Backbone
- 支持`Albentation`图像增强方法
- 支持warmup+cos decay scheduler
- 自动日志记录
- 支持混淆矩阵，accuracy， f1score，mAP等评估指标
- 支持onnx格式导出(完善中)

**update 24/7/30**:

- 添加了pytorch DDP逻辑

**update 24/8/7**:
- 更新华为杯相关内容
- 创建pretrain++分支, 支持个体识别离线数据集训练逻辑:https://github.com/Scienthusiasts/HeltonPretrain/tree/pretrain%2B%2B
- 更新了消融实验 `exp.md`