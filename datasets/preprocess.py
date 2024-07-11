import albumentations as A
import cv2
import torchvision.transforms as tf







class Transforms():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, imgSize):
        self.CoarseDropout = A.Compose([
                # 随机掩码
                A.CoarseDropout(max_holes=10, max_height=40, max_width=40, min_holes=5, min_height=10, min_width=10, fill_value=128, p=0.5),
                # A.CoarseDropout(max_holes=40, max_height=10, max_width=10, min_holes=30, min_height=5, min_width=5, fill_value=128, p=0.5),
        ])
        # 训练时增强
        self.trainTF = A.ReplayCompose([
                # 随机旋转
                A.Rotate(limit=15, p=0.5),
                # 先看下有没效果:
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # 随机尺寸裁剪并缩放到固定尺寸
                A.RandomResizedCrop(imgSize[0], imgSize[1], scale=(0.4, 1), ratio=(0.75, 1.33), p=0.5),
                # 随机镜像
                A.HorizontalFlip(p=0.5),
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机明亮对比度
                A.RandomBrightnessContrast(p=0.2),   
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                # 随机转为灰度图
                A.ToGray(p=0.01),
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=imgSize[0]),
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
            ])
        # 验证时增强
        self.validTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=imgSize[0]),
                # 较短的边做padding
                A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])
        

