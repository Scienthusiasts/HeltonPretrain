import os
# train eval test
MODE = 'eval'
BACKBONENAME = "ViT-L-14"
FROZEBACKBONE = False
WEIGHT = "F:/DeskTop/git/HeltonPretrain/ckpt/CLIP_ViT-L-14.pt"
RESUME = False
IMGSIZE = [224, 224]

'''food-101'''
# img_dir = 'E:/datasets/Classification/food-101/images'
# img_cat_dir = 'E:/datasets/Classification/food-101/images/train'
# cls_names = [cat_name for cat_name in os.listdir(img_cat_dir)]
# cls_names.sort()
# cls_num = len(cls_names)
# prompts_template_train = [
#     "a picture of /=/, a kind of food.",
#     "a picture of one /=/ in the scene, a sort of food that seems delicious.", #
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


'''cats & dogs'''
# img_dir = 'E:/datasets/Classification/cats_dogs/classification'
# img_cat_dir = 'E:/datasets/Classification/cats_dogs/classification/train'
# cls_names = [cat_name for cat_name in os.listdir(img_cat_dir)]
# cls_names.sort()
# cls_num = len(cls_names)
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
img_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_dogs_detail/images'
img_cat_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_dogs_detail/images/train'
cls_names = [cat_name for cat_name in os.listdir(img_cat_dir)]
cls_names.sort()
cls_num = len(cls_names)
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
    epoch = 12*4,
    log_dir = './log',
    log_interval = 50,
    eval_interval = 1,
    class_names = cls_names, 

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
        # backbone_name no used, but must kept
        backbone_name = BACKBONENAME,
        path = 'models/CLIP.py',
        cls_names = cls_names,
        prompts_template_train = prompts_template_train,
        prompts_template_val = prompts_template_val,
        weight_path = WEIGHT,
    ),
    # no used, but must kept
    optimizer = dict(
        optim_type = 'adamw',
        lr = 1e-3,
        lr_min_ratio = 0.1,
        warmup_lr_init_ratio = 0.1,
    ),
)


eval = dict(
    half=False,
    ckpt_path=False,    
)


test = dict(
    # classify_single, identify
    test_mode = 'classify_single', 
    img_path = "E:/datasets/Classification/food-101/images/valid/apple_pie/3676725.jpg",
    save_vis_path = './res1.jpg',
    ckpt_path=False, 
    half=False,
    # identify:
    img_dir = "E:/datasets/Classification/DogFace/after_4_bis",
)

# no used, but must kept
export = dict(
)