## food-101

**Model : mobilenetv3_large_100(timm)**

**bs:128**

**optimizer:adamw**

|        model         | epoch |  lr  | CLIP-teacher | distill_ratio | drop_block |  acc.(%)   |
| :------------------: | :---: | :--: | :----------: | :-----------: | :--------: | :--------: |
|      mbv3_large      |  121  | 1e-3 |      /       |       /       |     /      |   80.816   |
|      mbv3_large      |  49   | 2e-3 |      /       |       /       |     /      |   80.978   |
|      mbv3_large      |  49   | 2e-3 |      /       |       /       |   p=0.5    |   83.236   |
|      mbv3_large      |  121  | 2e-3 |      /       |       /       |   p=0.5    |   83.461   |
|      mbv3_large      |  49   | 2e-3 |    c=512     |       1       |   p=0.5    |   84.044   |
| mbv3_large_head_k111 |  49   | 2e-3 |    c=512     |       1       |   p=0.5    |   83.806   |
|      mbv3_large      |  49   | 2e-3 |    c=512     |      10       |   p=0.5    |   84.673   |
|      mbv3_large      |  49   | 2e-3 |    c=512     |      50       |   p=0.5    |   86.044   |
|      mbv3_large      |  49   | 2e-3 |    c=512     |      100      |   p=0.5    | **86.812** |



## cats_&_dogs (Kaggle)

**Model : mobilenetv3_large_100(timm)**

**epoch:49**

**optimizer:adamw**

**bs:128**

**lr:2e-3**

**drop_block:p=0.5**

|           model            | CLIP-teacher |  distill_ratio   |  acc.(%)   |
| :------------------------: | :----------: | :--------------: | :--------: |
|         mbv3_large         |    c=512     |        1         |   99.050   |
|         mbv3_large         |    c=512     |       100        |   99.250   |
| mbv3_large_decoupled_heads |    c=512     |       100        |   99.225   |
|   mbv3_large_three_heads   |    c=512     |       100        |   99.225   |
|         mbv3_large         |    c=512     |       200        |   99.300   |
|         mbv3_large         |    c=768     |       100        | **99.325** |
|         mbv3_large         |    c=768     | 100(contrast=10) |   98.925   |

**Model : mobilenetv4_hybrid_medium.e500_r224_in1k(timm)**

| model  | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block | acc.(%) |
| :----: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :-----: |
| mbv4_m |  49   | 128  | 2e-3 | adamw |    c=768     |      100      |   p=0.5    | 99.350  |

**Model : tf_efficientnet_b4.ns_jft_in1k(timm)**

| model  | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block | acc.(%) |
| :----: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :-----: |
| mbv4_m |  49   | 48   | 2e-3 | adamw |    c=768     |      100      |   p=0.5    | 99.500  |

**Model : resnetaa50d.sw_in12k_ft_in1k(timm)**

|        model         | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block |  acc.(%)   |
| :------------------: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :--------: |
|       resnet50       |  49   | 96   | 2e-3 | adamw |    c=768     |      100      |   p=0.5    | **99.575** |
| resnet50-**midc512** |  49   | 96   | 2e-3 | adamw |    c=768     |      100      |   p=0.5    |   99.500   |



## cats_&_dogs_merge

**Model : resnetaa50d.sw_in12k_ft_in1k(timm)**

**distill_ratio:100**

**CLIP-teacher:c=768**

bs:96*4 (ddp)

|  model   | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block |  acc.(%)   |
| :------: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :--------: |
| resnet50 |  49   | 384  | 2e-3 | adamw |    c=768     |      100      |   p=0.5    | **99.725** |
|          |       |      |      |       |              |               |            |            |



## The_Oxford_IIIT_Pet_Dataset

**Model : mobilenetv3_large_100(timm)**

**bs:128**

**drop_block:p=0.5**

CLIP-teacher:c=768

|        model        | epoch | optim |  lr  | distill_ratio | triplet_ratio |  acc.(%)   |
| :-----------------: | :---: | :---: | :--: | :-----------: | :-----------: | :--------: |
|     mbv3_large      |  49   | adamw | 2e-3 |      100      |       /       |   87.844   |
| mbv3_large_pretrain |  49   | adamw | 2e-3 |      100      |       /       |   89.425   |
| mbv3_large_pretrain |  49   |  sgd  | 2e-3 |      100      |       /       |   86.972   |
| mbv3_large_pretrain |  145  |  sgd  | 4e-3 |      100      |       /       | **90.243** |
| mbv3_large_pretrain |  145  |  sgd  | 4e-3 |      200      |       /       |   90.215   |
| mbv3_large_pretrain |  145  |  sgd  | 4e-3 |      100      |       1       |   90.052   |
| mbv3_large_pretrain |  145  |  sgd  | 4e-3 |      100      |      10       |   89.970   |

**Model : mobilenetv4_hybrid_medium.e500_r224_in1k(timm)**

| model  | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block | acc.(%) |
| :----: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :-----: |
| mbv4_m |  145  | 128  | 4e-3 |  sgd  |    c=768     |      100      |   p=0.5    | 91.060  |

**Model : tf_efficientnet_b4.ns_jft_in1k(timm)**

| model  | epoch | bs   |  lr  | optim | CLIP-teacher | distill_ratio | drop_block | acc.(%) |
| :----: | :---: | ---- | :--: | :---: | :----------: | :-----------: | :--------: | :-----: |
| mbv4_m |  145  | 40   | 4e-3 |  sgd  |    c=768     |      100      |   p=0.5    | 92.123  |

**Model : resnetaa50d.sw_in12k_ft_in1k(timm)**

**distill_ratio:100**

**CLIP-teacher:c=768**

**aug+:**

`A.CoarseDropout(max_holes=10, max_height=40, max_width=40, min_holes=5, min_height=10, min_width=10, fill_value=128, p=0.5)`

`A.RandomResizedCrop(imgSize[0], imgSize[1], scale=(0.4, 1), ratio=(0.75, 1.33), p=0.5)`

|               model                | epoch | bs   |  lr  | optim | drop_block |  acc.(%)   |
| :--------------------------------: | :---: | ---- | :--: | :---: | :--------: | :--------: |
|         resnet50_pretrain          |  145  | 96   | 4e-3 |  sgd  |   p=0.5    |   92.150   |
| resnet50_pretrain (froze_backbone) |  145  | 96   | 4e-3 |  sgd  |   p=0.5    |   92.232   |
|     resnet50_pretrain-midc512      |  145  | 96   | 4e-3 |  sgd  |   p=0.5    |   92.314   |
|   resnet50_pretrain-midc512-aug+   |  145  | 96   | 4e-3 |  sgd  |   p=0.5    | **92.723** |
| resnet50_pretrain-midc512-aug+-lp  |  145  | 96   | 4e-3 |  sgd  |   p=0.5    |   92.723   |



## FlickrBreeds+The_Oxford_IIIT_Pet_Dataset

**Model : resnetaa50d.sw_in12k_ft_in1k(timm)-midc512**

**optimizer:sgd**

**CLIP-teacher:c=768**

**pretrain：cats_&_dogs (Kaggle)**

**aug+:**

`A.CoarseDropout(max_holes=10, max_height=40, max_width=40, min_holes=5, min_height=10, min_width=10, fill_value=128, p=0.5)`

`A.RandomResizedCrop(imgSize[0], imgSize[1], scale=(0.4, 1), ratio=(0.75, 1.33), p=0.5)`

|               model               | epoch | bs   |  lr  | distill_ratio | drop_block |  acc.(%)   |
| :-------------------------------: | :---: | ---- | :--: | :-----------: | :--------: | :--------: |
|     resnet50_pretrain(kaggle)     |  49   | 96   | 4e-3 |      100      |   p=0.5    |   94.631   |
|     resnet50_pretrain(kaggle)     |  145  | 96   | 4e-3 |      100      |   p=0.5    |   95.557   |
|  resnet50_aug+_pretrain(kaggle)   |  145  | 96   | 4e-3 |      100      |   p=0.5    |   95.884   |
| resnet50_aug+_pretrain(kaggle)-lp |  145  | 256  | 4e-3 |      100      |   p=0.5    | **95.912** |

