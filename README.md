<div align='center'>
    <h1>HeltonPretrain 🚀</h1>
    <p><em>0~1实现预训练框架，基于Pytorch，不断完善中 ~</em></p>
</div>





## ✒️设计逻辑
设计理念: 简洁版mm框架

**框架设计逻辑： 通用(换一个任务, 当前的设计逻辑可以完全套用) / 专用(仅适用于当前任务)**

- `configs/*` (专用, 但其组织形式通用)
- `modules/*` (专用, 但其组织形式通用)
- `utils/log_utils.py` (通用)
- `utils/utils.py` (通用)
- `utils/eval_utils.py` (专用)
- `utils/metrics.py` (专用)
- `utils/hooks.py` (目前是通用)
- `nregister.py` (通用)
- `tools/train.py` (通用)
- `tools/eval.py` (通用)
- `tools/test.py` (专用)



## 🔧安装

```
conda create -n hp python=3.10
cd HeltonPretrain
pip install -r requirements.txt
pip install -e .
```




## 🔥训练

单卡训练 example

```
 # 注意修改config文件里对应参数, mode="train"
 python tools/train.py --config configs/fcnet.py
```

DDP 多卡训练 example

```
 # 注意修改config文件里对应参数, mode="train_ddp"
sh run.sh
```



## 🔥评估

```
 # 注意修改config文件里对应参数, mode="eval"
 python tools/eval.py --config configs/fcnet.py
```



## 🔥推理

```
python tools/test.py 
```





## 📃更新日志
|     时间     |                             详细                             |
| :----------: | :----------------------------------------------------------: |
| `2025/09/24` | ✅ 代码重构  ✨ **支持注册机制(动态配置自定义参数文件), hook机制**  ✨ **支持Pytorch DDP 分布式训练**  ✅ 自动日志记录, 支持 `tensorboard`  ✅ resume 中断恢复训练  ✅ 支持 scheduler (学习率decay)  ✅ 兼容 `timm` 支持的 Backbone模型，可灵活更换 Backbone  ✅ 兼容 `Albentation` 图像增强方法  ✅ 支持混淆矩阵, accuracy, f1score, mAP 等评估指标  ✅ 深度学习 Hello Word -> 支持图像分类任务  ➡️ 各种表征学习自监督方法 (完善中...) |
| `2025/09/25` |    ✅ 添加 openai-clip，qihoo360-fgclip 基础模型、 clip-zero-shot 分类模型     |



## ➡️TODO
框架迁移 (将适配更多下游任务, 检测, 分割, 生成...)

