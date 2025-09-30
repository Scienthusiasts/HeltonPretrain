import torch
import torch.nn as nn
import torch.nn.functional as F
from register import MODELS



@MODELS.register
class TeacherDistillLoss1D(nn.Module):
    """利用教师模型(冻结教师)的输出表征计算蒸馏损失
    """
    def __init__(self, s_dim:int, t_dim:int, teacher:nn.Module, distill_loss:nn.Module):
        """
        """
        super(TeacherDistillLoss1D, self).__init__()
        self.teacher = teacher.eval()
        # self.proj用于对齐学生和教师输出特征的维度
        self._proj = nn.Linear(s_dim, t_dim) if s_dim!=t_dim else nn.Identity()
        self.distill_loss = distill_loss


    def forward(self, device, img, students_embs):
        """
        Args:
            img: [B, C, H, W]
            students_embs: [B, dims]
        """
        # 维度对齐
        students_embs = self._proj(students_embs)
        teacher_embs = self.teacher(img, type='image', device=device)
        loss = self.distill_loss(students_embs, teacher_embs)
        return loss


    def state_dict(self, *args, **kwargs):
        # 过滤掉带 _no_state_dict 标记的层
        orig = super().state_dict(*args, **kwargs)
        return {k: v for k, v in orig.items() if "_proj" not in k and "teacher" not in k}





@MODELS.register
class TeacherDistillLoss2D(nn.Module):
    """利用教师模型(冻结教师)的输出表征计算蒸馏损失
    """
    def __init__(self, s_dim:int, t_dim:int, teacher:nn.Module, distill_loss:nn.Module):
        """
        """
        super(TeacherDistillLoss2D, self).__init__()
        self._teacher = teacher.eval()
        # self.proj用于对齐学生和教师输出特征的维度
        self._proj = nn.Conv2d(s_dim, t_dim, 1) if s_dim!=t_dim else nn.Identity()
        self.distill_loss = distill_loss


    def forward(self, device, img, students_embs):
        """
        Args:
            img: [B, C, H, W]
            students_embs: [B, c, h, w]
        """
        # 通道维度对齐
        students_embs = self._proj(students_embs)
        teacher_embs = self._teacher(img, type='image_dense', device=device)
        sh, sw = students_embs.shape[2:]
        th, tw = teacher_embs.shape[2:]
        # 空间尺寸对齐
        if sh!=th or sw!=tw:
            students_embs = F.interpolate(students_embs, size=(th, tw), mode="bilinear", align_corners=False)
        loss = self.distill_loss(students_embs, teacher_embs)
        return loss


    def state_dict(self, *args, **kwargs):
        # 过滤掉带 _no_state_dict 标记的层
        orig = super().state_dict(*args, **kwargs)
        return {k: v for k, v in orig.items() if "_proj" not in k and "_teacher" not in k}
















if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from modules.datasets.preprocess import Transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfgs = dict(
        type='TeacherDistillLoss2D',
        s_dim=256,
        t_dim=1280,
        teacher=dict(
            type='DINOv3',
            model_name='vit_huge_plus_patch16_dinov3.lvd1689m',
            pretrained=False, 
            load_ckpt='ckpts/vit_huge_plus_patch16_dinov3.lvd1689m.pt'
        ),
        distill_loss=dict(
            type="SmoothL1Loss"
        )
    )
    dloss = MODELS.build_from_cfg(cfgs).to(device)

    img_size = [224, 224]
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
    img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_41.jpg"
    image = np.array(Image.open(img_path).convert('RGB'))

    # 图像预处理
    transform = Transforms(img_size)
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)

    # student_embs = torch.rand((1, 1080)).to(device)
    student_embs = torch.rand((1, 256, 7, 7)).to(device)
    loss = dloss(device, tensor_img, student_embs)
    print(dloss)
    print(loss)
    

