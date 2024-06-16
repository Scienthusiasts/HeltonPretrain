
import os
# train eval test
MODE = 'train'
# mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
FROZEBACKBONE = False
TESTCKPT = 'log/mobilenetv3_large_100_food101/bs_128_lr_2e-3_epoch_121_aug/best_acc.pt'
# fficientvit_m5:
# BACKBONEPATH = f'ckpt/efficientvit_m5.r224_in1k.pt'; MIDC = [256, 256, 256]; KERNELS = [3,1,1]; BACKBONENAME = 'efficientvit_m5.r224_in1k'
# mobilenetv3_large_100:
BACKBONEPATH = f'ckpt/mobilenetv3_large_100.ra_in1k.pt'; MIDC = [512, 512, 512]; KERNELS = [1, 1, 1]; BACKBONENAME = 'mobilenetv3_large_100.ra_in1k'
LOADCKPT = False
RESUME = False
IMGSIZE = [224, 224]

'''food-101'''
img_dir = 'E:/datasets/Classification/food-101/images'
img_cat_dir = 'E:/datasets/Classification/food-101/images/train'
cat_names = [cat_name for cat_name in os.listdir(img_cat_dir)]
cat_names.sort()
cls_num = len(cat_names)







runner = dict(
    seed = 22,
    mode = MODE,
    resume = RESUME,
    img_size = IMGSIZE,
    epoch = 12*10,
    log_dir = './log/mobilenetv3_large_100_food101',
    log_interval = 50,
    eval_interval = 2,
    class_names = cat_names, 

    dataset = dict(
        bs = 8*16,
        num_workers = 0,
        # 自定义的Dataset:
        my_dataset = dict(
            path = 'datasets/ClassifyDataset.py',
            train_dataset = dict(
                mode='train',
                contrast=False,
                imgSize = IMGSIZE,
                dir = img_dir,
            ),
            val_dataset = dict(
                mode='valid',
                contrast=False,
                imgSize = IMGSIZE,
                dir = img_dir,                
            ),
        ),
    ),


    model = dict(
        path = 'models/ClsNet/ClsNet.py',
        cls_num = cls_num, 
        cls_name = cat_names,
        loadckpt = LOADCKPT,           
        backbone_name = BACKBONENAME,
        backbone = dict(
            loadckpt=BACKBONEPATH, 
            pretrain=False, 
            froze=FROZEBACKBONE,
        ),
        head = dict(
            add_share_head = True,
            kernel_s = KERNELS,
            mid_c = MIDC
        )
    ),
    optimizer = dict(
        optim_type = 'adamw',
        lr = 5e-4,
        lr_min_ratio = 0.01,
        warmup_lr_init_ratio = 0.01,
    ),
)


eval = dict(
    half=False,
    ckpt_path=TESTCKPT,    
)


test = dict(
    # classify, identify, onnx_classify
    test_mode = 'classify_single', 
    img_path = "E:/datasets/Classification/food-101/images/valid/apple_pie/3676725.jpg",
    save_vis_path = './res1.jpg',
    ckpt_path = TESTCKPT,
    half=False,
)

