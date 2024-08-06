import os
import numpy as np
from PIL import Image, ImageFile







if __name__ == '__main__':
    dir = r'E:\datasets\Classification\HUAWEI_cats_dogs_fine_grained\DogFace\after_4_bis'
    id_imgpath_dict = {}
    imgpath_list = []
    for dir1_name in os.listdir(dir):
        dir1 = os.path.join(dir, dir1_name)
        for id_dir_name in os.listdir(dir1):
            id_dir = os.path.join(dir1, id_dir_name)
            id_imgpath_dict[id_dir] = []
            for img_name in os.listdir(id_dir):
                img_path = os.path.join(id_dir, img_name)
                imgpath_list.append(img_path)
                id_imgpath_dict[id_dir].append(img_path)


    anchor_path = imgpath_list[24]
    anchor = Image.open(anchor_path).convert('RGB')     
    anchor = np.array(anchor)
    anchor_id_path = os.path.split(anchor_path)[0]   
    pos_list = id_imgpath_dict[anchor_id_path]
    # anchor样本在对应id文件夹中的位置索引   
    anchor_idx = pos_list.index(anchor_path)
    # 正样本随机在id文件夹中选择
    pos_idx = np.random.randint(len(pos_list))
    # 如果正样本取到anchor样本的位置, 则正样本直接取anchor样本的下一个元素
    if pos_idx==anchor_idx: pos_idx = ( anchor_idx + 1) % len(pos_list)
    neg_item = np.random.randint(len(imgpath_list))
    print(anchor_path)
    print(anchor_id_path)
    print(pos_list)
    print(anchor_idx)
    print(pos_idx)
    print(neg_item)
