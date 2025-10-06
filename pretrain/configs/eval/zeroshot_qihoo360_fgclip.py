# this config file for eval only
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
seed = 42
log_dir = r'./log/zeroshot_qihoo360_fgclip_eval'
# base
# img_size = [224, 224]
# pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-base/snapshots/f30d2b82ba939fd54ca732426f99f4d6c3c92387'
# large
img_size = [336, 336]
pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-large/snapshots/19c2df7667052518ade09341652562b89b1332da'

template_prompt = [
        "a {} in the scene."
        "A {} with a happy expression.",
        "A photo of a friendly {}.",
        "A lovely little {}.",
        "a photo showing a {} in the scene",
        "a color picture of a {}, it is cute",
        "a photograph of a nice {}.",
        "a cropped photo of a {}, it is playful.",
        "I own a {} and I really like it.",
        "a picture of a {} taken long time ago.",
        "the picture showing a {} in the center.",
        "a picture of one {} in the scene.", # 
        "I adopted this {} several years ago.",
        "I took a picture of my {}.",
        "I love my {} and it loves me too.",
        "The {} in the picture is my friend's.",
        "This {} was a birthday present from my best friend.",
        "I accidentally snapped a picture of this {}.",
        "I petted my {} and she seemed to enjoy it.",
        "I called out to the {} and it rushed to me.",
        "My {} looking at the camera. It's the best memory ever.",
        "this {} used to be my best mate. Now it's gone.",
        "You're the best, my good {}.",
        "the {} is staring at me, want something to eat.",
        "My neighbour's {}, it looks funny.",
        "An elegant {} posing gracefully for the camera.",
        "This {} always greets me when I come home from work.",
        "The {} I rescued from the shelter last month.",
        "My {} waiting patiently for dinner time.",
        "The {} that helped me through difficult times.",
        "A {} communi{}ing through its expressive body language.",
        "this {}, A loyal companion with eyes full of devotion.",
        "A mischievous little {}, always getting into something new.",
        "Caught in the act! My {} staring guiltily at the mess it just made.",
        "my {}, The best welcome home committee, always waiting at the door.", 
        "My shadow, my {}, following me from room to room throughout the day.",
        "A blur of energy, the {} racing across the yard at full speed.",
        "A candid moment, this {} completely unaware of the camera.",
        "A calming influence, this {} knowing how to soothe a bad day.",
        "An unexpected friendship that grew into an unbreakable bond, my {}.",
        "The simple pleasure of watching my {} enjoy the sunshine.",
        "The first picture I took before we brought the {} home." 
    ]
# template_prompt = "a {} in the scene."
cat_names = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 
             'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 
             'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 
             'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 
             'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

'''模型配置参数'''
model_cfgs = dict(
    type="ZeroShotCLIP",
    cat_names=cat_names, 
    template_prompt=template_prompt,
    clip_model=dict(
        type="Qihoo360FGCLIP",
        pretrain_path=pretrain_path
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    valid_dataset_cfg=dict(
        type="INDataset",
        img_dir=validset_path,
        mode="valid",
        img_size=img_size,
        drop_block=False
    ),
    valid_bs=1,
    num_workers=8,
    valid_shuffle=False
)
