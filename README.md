# HeltonPretrain 🚀

从0实现图像分类/预训练框架，正在完善中~


## 设计逻辑
简洁版mm框架

**框架设计逻辑 通用(换一个任务, 当前的设计逻辑可以完全套用) / 专用(仅适用于当前任务)**
- `configs/*` (专用, 但其组织形式通用)
- `modules/*` (专用, 但其组织形式通用)
- `utils/log_utils.py` (通用)
- `utils/utils.py` (通用)
- `utils/eval_utils.py` (专用)
- `utils/metrics.py` (专用)
- `utils/hooks.py` (目前是通用)
- `nregister.py` (通用)
- `runner.py` (通用)
- `infer.py` (专用)


## 训练











## 更新日志
**update 25/9/24**
- 代码重构 ✅
- **支持注册机制(动态配置自定义参数文件), hook机制** ✨
- **支持Pytorch DDP 分布式训练** ✨
- 自动日志记录, 支持 `tensorboard`
- resume 中断恢复训练
- 支持 scheduler (学习率decay)
- 兼容 `timm` 支持的 Backbone模型，可灵活更换 Backbone
- 兼容 `Albentation` 图像增强方法
- 支持混淆矩阵, accuracy, f1score, mAP 等评估指标
- 深度学习 Hello Word -> 支持图像分类任务 ✅
- 各种表征学习自监督方法 (完善中...) 

## TODO
框架迁移 (将适配更多下游任务, 检测, 分割, 生成...)


