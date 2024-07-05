
import os
# train eval test export
MODE = 'test'
CLIPWEIGHT = "F:/DeskTop/git/HeltonPretrain/ckpt/CLIP_ViT-L-14.pt"
FROZEBACKBONE = True
RESUME = False
IMGSIZE = [224, 224]

'''mobilenetv3_large_100:'''
# BACKBONEPATH = f'ckpt/mobilenetv3_large_100.ra_in1k.pt'; MIDC = [768, 768, 768]; KERNELS = [1, 1, 1]; BACKBONENAME = 'mobilenetv3_large_100.ra_in1k'
'''resnetaa50d.sw_in12k_ft_in1k:'''
BACKBONEPATH = f'ckpt/resnetaa50d.sw_in12k_ft_in1k.pt'; MIDC = [768, 768, 768]; KERNELS = [1, 1, 1]; BACKBONENAME = 'resnetaa50d.sw_in12k_ft_in1k'
'''efficientvit_m5.r224_in1kk:'''
# BACKBONEPATH = f'ckpt/tf_efficientnet_b4.ns_jft_in1k.pt'; MIDC = [768, 768, 768]; KERNELS = [1, 1, 1]; BACKBONENAME = 'tf_efficientnet_b4.ns_jft_in1k'
'''vit_small_patch16_224.augreg_in21k_ft_in1k.pt:'''
# BACKBONEPATH = f'ckpt/vit_small_patch16_224.augreg_in21k_ft_in1k.pt'; MIDC = [768, 768, 768]; KERNELS = [1, 1, 1]; BACKBONENAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
'''mobilenetv4_hybrid_medium.e500_r224.pt:'''
# BACKBONEPATH = f'ckpt/mobilenetv4_hybrid_medium.e500_r224_in1k.pt'; MIDC = [768, 768, 768]; KERNELS = [1, 1, 1]; BACKBONENAME = 'mobilenetv4_hybrid_medium.e500_r224_in1k'

TESTCKPT = "log/clip_resnetaa50d.sw_in12k_ft_in1k_cats_dogs_37/bs_128_lr_4e-3_epoch_145_aug_clip_distillloss_100_pretrain_sgd/last.pt"
LOADCKPT = "log/clip_resnetaa50d.sw_in12k_ft_in1k_cats_dogs_37/bs_128_lr_4e-3_epoch_145_aug_clip_distillloss_100_pretrain_sgd/last.pt"

onnx_export_dir = os.path.join('onnx_ckpt', TESTCKPT.split('/')[1])
onnx_export_name = f"{TESTCKPT.split('/')[-2]}.onnx"






'''food-101'''
# img_dir = 'E:/datasets/Classification/food-101/images'
# img_cat_dir = 'E:/datasets/Classification/food-101/images/train'
# cat_names = [cat_name for cat_name in os.listdir(img_cat_dir)]
# cat_names.sort()
# cls_num = len(cat_names)
# prompts_template_train = [
#     "a picture of /=/, a kind of food.",
#     "a picture of one /=/ in the scene, a sort of food that seems delicious.",
#     "a photo showing the /=/ in the center, a kind of food.",
#     "there is a /=/ in the scene, a kind of food.",
#     "a color picture of a /=/, a kind of food.",
#     "a photograph of a nice /=/, a sort of food.",
#     "a photograph of a nice /=/ I took recently, delicious!",
#     "a cropped photo of a /=/, a kind of food.",
#     "I made a /=/ and I really like it.",
#     "a picture of a /=/ taken long time ago, a sort of food.",
#     "A picture of /=/ that seems delicious.",
#     "A picture of food, the category of which is /=/.",
#     "a food in the scence, it is /=/.",
#     "I made this /=/, a kind of food, which is tasty.",
# ]
# prompts_template_val = [
#     "a picture of one /=/ in the scene, a sort of food that seems delicious."
# ]


'''cats_dogs'''
# img_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_vs_dogs/classification'
# train_img_cat_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_vs_dogs/classification/train'
# valid_img_cat_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_vs_dogs/classification/train'
# cat_names = [cat_name for cat_name in os.listdir(train_img_cat_dir)]
# cat_names.sort()
# cls_num = len(cat_names)
# prompts_template_train = [
#     "A /=/ with a happy expression.",
#     "A photo of a friendly /=/.",
#     "A lovely little /=/.",
#     "a photo showing a /=/ in the scene",
#     "a color picture of a /=/, it is cute",
#     "a photograph of a nice /=/.",
#     "a cropped photo of a /=/, it is playful.",
#     "I own a /=/ and I really like it.",
#     "a picture of a /=/ taken long time ago.",
#     "the picture showing a /=/ in the center.",
#     "a picture of one /=/ in the scene.", # 
#     "I adopted this /=/ several years ago.",
#     "I took a picture of my /=/.",
#     "I love my /=/ and he loves me too.",
#     "The /=/ in the picture is my friend's.",
#     "This /=/ was a birthday present from my best friend.",
#     "I accidentally snapped a picture of this /=/.",
#     "I petted my /=/ and she seemed to enjoy it.",
#     "I called out to the /=/ and he rushed to me.",
#     "My /=/ looking at the camera. It's the best memory ever.",
#     "this /=/ used to be my best mate. Now he's gone.",
#     "You're the best, my good /=/.",
#     "the /=/ is staring at me, want something to eat.",
#     "My neighbour's /=/, it looks funny.",

# ]
# prompts_template_val = [
#     "a picture of one /=/ in the scene."
# ]



'''cats & dogs 37'''
img_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/The_Oxford_IIIT_Pet_Dataset/images'
train_img_cat_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/The_Oxford_IIIT_Pet_Dataset/images/train'
valid_img_cat_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/The_Oxford_IIIT_Pet_Dataset/images/valid'
cat_names = [cat_name for cat_name in os.listdir(train_img_cat_dir)]
cat_names.sort()
cls_num = len(cat_names)
prompts_template_train = [
    "A /=/ with a happy expression.",
    "A photo of a friendly /=/.",
    "A lovely little /=/.",
    "a photo showing a /=/ in the scene",
    "a color picture of a /=/, it is cute",
    "a photograph of a nice /=/.",
    "a cropped photo of a /=/, it is playful.",
    "I own a /=/ and I really like it.",
    "a picture of a /=/ taken long time ago.",
    "the picture showing a /=/ in the center.",
    "a picture of one /=/ in the scene.", # 
    "I adopted this /=/ several years ago.",
    "I took a picture of my /=/.",
    "I love my /=/ and he loves me too.",
    "The /=/ in the picture is my friend's.",
    "This /=/ was a birthday present from my best friend.",
    "I accidentally snapped a picture of this /=/.",
    "I petted my /=/ and she seemed to enjoy it.",
    "I called out to the /=/ and he rushed to me.",
    "My /=/ looking at the camera. It's the best memory ever.",
    "this /=/ used to be my best mate. Now he's gone.",
    "You're the best, my good /=/.",
    "the /=/ is staring at me, want something to eat.",
    "My neighbour's /=/, it looks funny.",
]
prompts_template_val = [
    "a picture of one /=/ in the scene."
]





runner = dict(
    seed = 22,
    mode = MODE,
    resume = RESUME,
    img_size = IMGSIZE,
    epoch = 12*4+1,
    log_dir = './log/mobilenetv3_large_100_food101',
    log_interval = 50,
    eval_interval = 1,
    class_names = cat_names, 

    dataset = dict(
        bs = 8*16,
        num_workers = 0,
        # 自定义的Dataset:
        my_dataset = dict(
            path = 'datasets/ClassifyDataset.py',
            train_dataset = dict(
                mode='train',
                contrast=True,
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
        path = 'models/CLIPDistillClsNet/CLIPDistillClsNet.py',
        # cls clip ensemble
        infer_mode= 'ensemble',
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
            mid_c = MIDC,
            clip_embedding_c = 768,
        ),
        clip = dict(
            cls_names = cat_names,
            prompts_template_train = prompts_template_train,
            prompts_template_val = prompts_template_val,
            weight_path = CLIPWEIGHT,
        ),
    ),
    optimizer = dict(
        optim_type = 'adamw',
        lr = 2e-3,
        lr_min_ratio = 0.01,
        warmup_lr_init_ratio = 0.01,
    ),
)


eval = dict(
    half=False,
    ckpt_path=TESTCKPT,    
)


test = dict(
    # clssify_single, clssify_batch, identify, onnx_classify_single, onnx_classify_batch
    test_mode = 'onnx_classify_batch', 
    ckpt_path = TESTCKPT,
    half=False,
    tta=False,
    # clssify_single
    img_path = "E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/The_Oxford_IIIT_Pet_Dataset/images/miniature_pinscher/miniature_pinscher_49.jpg",
    save_vis_path = './res1.jpg',
    # clssify_batch:
    img_dir = valid_img_cat_dir,  
    # identify:
    id_img_dir = "E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/DogFace/after_4_bis",
    # onnx_classify_single:
    onnx_path = os.path.join(onnx_export_dir, onnx_export_name)
)

export = dict(
    export_dir = onnx_export_dir,
    export_name = onnx_export_name,
    ckpt_path = TESTCKPT,
)