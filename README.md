<div align='center'>
    <h1>HeltonX♾️</h1>
    <img src="https://github.com/Scienthusiasts/heltonx/blob/main/demo/logo.png"/>
    <p><em>致力于 0~1 实现通用深度学习框架，基于 Pytorch，支持各类下游任务，不断完善中 ~</em></p>
</div>



## ✒️设计逻辑
设计理念: 简洁版mm框架

**框架设计逻辑： 通用(换一个任务, 当前的设计逻辑可以完全套用) / 专用(仅适用于当前任务)**

```
HeltonX:
├─demo    (README展示相关)
├─configs
│  └─accelerate_yamls
│     ├─accelerate_ddp.yaml         (通用, Accelerator库ddp训练配置文件)
│     └─accelerate_single_gpu.yaml  (通用, Accelerator库单卡训练配置文件)
├─tools
│  ├─train.py  (通用, 训练pipeline)
│  └─eval.py   (通用, 评估pipeline)
├─utils            
│  ├─utils.py       (通用, 一些可能用到的方法)
│  ├─ckpts_utils.py (通用, 权重load/save相关逻辑)
│  ├─log_utils.py   (通用, 日志记录逻辑)
│  ├─hooks.py       (通用, 钩子机制, 实现训练,评估时必用的方法)
│  └─register.py    (通用, 注册机制)
├─optimization
│  ├─optimizers.py  (通用, 优化器)
│  └─schedulers.py  (通用, 学习率decay)
├─setup.py (安装脚本)
│------------------------------------------------------------- 
├─pretrain    (专用, 但其组织形式通用)
│  ├─configs  (自定义模型配置参数文件)
│  ├─datasets (自定义Dataset和数据增强)
│  ├─losses   (自定义损失函数)
│  ├─models         (自定义网络组件)
│  │  ├─backbones   (自定义骨干网络)
│  │  ├─classifiers (自定义分类器)
│  │  ├─heads       (自定义任务头)
│  │  └─vfms        (视觉基础模型)
│  ├─utils             (实现任务特定的相关功能)
│  │  ├─eval_utils.py  (评估pipeline, 和具体任务有关)
│  │  └─metrics.py     (评估指标计算, 和具体任务有关)
│  └─tools             
│     ├─test.py            (测试相关逻辑, 完善中)
│     ├─run.sh             (DDP训练脚本)
│     └─run_accelerate.sh  (基于Accelerator库的DDP训练脚本)
└─generation    (同pretrain)
   └─... ...
```

###  `utils/register.py`

- `MODELS`：注册nn.Module子类
- `DATASETS`：注册dataset.Dataset子类
- `OPTIMIZERS`：注册torch.optim.Optimizer子类
- `SCHEDULERS`：注册torch.optim.lr_scheduler子类
- `EVALPIPELINES`：注册任务特定的评估pipelines

### `utils/hooks.py`

- `NecessaryHook.hook_after_batch`：记录/打印日志
- `NecessaryHook.hook_after_epoch`：**评估**+记录/打印日志+保存权重
  - 注：当使用DDP训练时, 这里只在主节点的模型上评估,其余结点什么也不做
- `NecessaryHook.hook_after_epoch`：**评估**+记录/打印日志

### `xxx/utils/eval_utils.py`

- `XxxEvalPipeline`：用于评估流程(Trainner中一个epoch后的评估 / Evaler中的评估)，传入runner实例用于获取评估相关参数

### `xxx/configs/*.py`  (配置文件)

- model_cfgs：模型相关配置参数
- dataset_cfgs：数据集相关配置参数
- optimizer_cfg：优化器相关配置参数
- scheduler_cfgs：学习率decay相关配置参数
- eval_pipeline_cfgs：任务特定的评估pipeline配置参数



## 🔧安装

```bash
conda create -n hx python=3.10
cd HeltonXNet
pip install -r requirements.txt
pip install -e .
```




## 🔥训练

单卡训练 example

```bash
# 根据具体需求修改config文件里相关配置参数, mode="train"
python tools/train.py --config pretrain/configs/xxx.py

# 或，使用accelerate库封装过的训练pipeline
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file tools/accelerate_single_gpu.yaml tools/train_accelerate.py --config pretrain/configs/xxx.py
```

DDP 多卡训练 example

```bash
# 根据具体需求修改config文件里相关配置参数, config文件下mode="train_ddp"
sh pretrain/tools/run.sh

# 或，使用accelerate库封装过的训练pipeline
sh pretrain/tools/run_accelerate.sh
```



## 🔥评估

```bash
# 注意修改config文件里对应参数, mode="eval"
python tools/eval.py --config pretrain/configs/xxx.py
```



## 🔥推理 (完善中...)

```bash
python pretrain/tools/test.py 
```





## 📃更新日志
|     时间     |                             详细                             |
| :----------: | :----------------------------------------------------------: |
| `2025/09/24` | ✅ 代码重构  ✨ **支持注册机制(动态配置自定义参数文件), hook机制**  ✨ **支持Pytorch DDP 分布式训练**  ✅ 自动日志记录, 支持 `tensorboard`  ✅ resume 中断恢复训练  ✅ 支持 scheduler (学习率decay)  ✅ 兼容 `timm` 支持的 Backbone模型，可灵活更换 Backbone  ✅ 兼容 `Albentation` 图像增强方法  ✅ 支持混淆矩阵, accuracy, f1score, mAP 等评估指标  ✅ 深度学习 Hello Word -> 支持图像分类任务  ➡️ 各种表征学习自监督方法 (完善中...) |
| `2025/09/25` | ✅ 添加 openai-clip，qihoo360-fgclip VFM、 clip-zero-shot 分类模型 |
| `2025/09/26` |                    ✅ 添加 timm DINOv3 VFM                    |
| `2025/09/29` |                     ✅ 添加 VFMs 蒸馏损失                     |
| `2025/10/4`  |             ✅ 添加基于 CLIP 蒸馏的多任务分类模型             |
| `2025/10/11` | ➡️ 添加生成任务`./generation`, 支持 DDPM/DDIM (开发中，目前代码逻辑还不够清晰) |
| `2025/10/12` |       ✅ **支持 `Accelerate`(一键 DDP、混合精度训练)**        |



## ➡️TODO
任务拓展 (将支持更多下游任务, 检测, 分割, 生成...)

