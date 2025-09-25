# this config file for eval only
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
seed = 42
log_dir = r'./log/zeroshot_openai_clip_eval'
img_size = [224, 224]
# clip_vitb32_512 clip_vitl14_768
name = 'clip_vitl14_768' 
pretrain_path = fr'/mnt/yht/code/HeltonPretrain/ckpts/{name}.pt'

template_prompt = "a picture of one {} in the scene."
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
        type="OpenAICLIP",
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
