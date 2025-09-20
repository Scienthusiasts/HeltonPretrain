import torch

from models import *
from register import MODELS
# print("registered keys:", list(MODELS.keys()))


if __name__ == '__main__':
    # 配置字典
    cfg = {
        "type": "MLFNet",
        "backbone":{
            "type": "TIMMBackbone",
            "model_name": "mobilenetv4_conv_small",
            "pretrained": True,
            "out_indices": [1, 2, 3, 4],
            "froze_backbone": True,
            "load_ckpt": None

        },
        "head":{
            "type": "MLPHead",
            "layers_dim":[32+64+96+960, 256, 10]
        },
        "load_ckpt": None
    }
    mlfnet = MODELS.build_from_cfg(cfg)
    print(mlfnet)

    x = torch.randn(4, 3, 224, 224)
    features = mlfnet(x)
    print(features.shape)
