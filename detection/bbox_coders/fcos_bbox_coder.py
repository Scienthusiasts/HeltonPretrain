import torch
import torch.nn as nn
from detection.utils.fcos_utils import *
from utils.register import MODELS





@MODELS.register
class FCOSBBoxCoder(nn.Module):
    """FCOS bbox编解码操作. 
       encode: 将gt编码成能直接和网络预测结果计算损失的特征
       decode: 对网络预测的结果解码成真实的基于原图尺寸的结果     

    """
    def __init__(self, strides=[8, 16, 32, 64, 128]):
        """
            Args:
                strides: 每一个尺度特征图相对于原图的下采样率
        """
        super(FCOSBBoxCoder, self).__init__()
        self.strides = strides


    def decode(self, cls_logits, cnt_logits, reg_preds):
        """将预测结果解码为真实结果(xyxy)
            Args:
                cls_logits:  网络预测的类别logits     [[bs, cls_num, h1, w1], ..., [[bs, cls_num, hn, wn]]]
                cnt_logits:  网络预测的中心度logits   [[bs, 1, h1, w1], ..., [[bs, 1, hn, wn]]]
                reg_preds:   网络预测的回归值         [[bs, 4, h1, w1], ..., [[bs, 4=(l,t,r,b), hn, wn]]]
            Returns:
                boxes_score_classes: 解码后的预测结果 [bs, total_anchor_num, 4+1+1=(x0, y0, x1, y1, score, label)]
        """
        '''调整形状'''
        # [[bs, cls_num, 80, 80],...,[[bs, cls_num, 5, 5]]] -> [bs, 8525, cls_num]
        cls_preds = reshape_cat_out(cls_logits)
        # [[bs, 1, 80, 80],...,[[bs, 1, 5, 5]]] -> [bs, total_anchor_num, 1]
        cnt_preds = reshape_cat_out(cnt_logits)
        # [[bs, 4, 80, 80],...,[[bs, 4, 5, 5]]] -> [bs, total_anchor_num, 4]
        reg_preds = reshape_cat_out(reg_preds)
        # 对分类结果归一化到0,1之间
        cls_preds = torch.sigmoid(cls_preds)
        cnt_preds = torch.sigmoid(cnt_preds)
        # 获得得分最高对应的类别得分和类别 [bs, total_anchor_num, 1]
        cls_scores, cls_classes = torch.max(cls_preds, dim=-1, keepdim=True)

        '''生成网格(类似anchor point, 每个网格的中心点)'''
        lvl_size = [feat.shape[2:4] for feat in cnt_logits]
        # [total_anchor_num, 2=(cx, cy)]
        grids = self.gen_grid(lvl_size).to(cnt_preds.device)
        
        '''置信度是类别得分和centerness的乘积'''
        cls_scores = cls_scores * cnt_preds
        # 通过中心点坐标和网络预测的ltrb获得box的左上角右下角点xyxy(原图的未归一化坐标)
        # grids多一个bs维度
        left_top = grids[None, :, :] - reg_preds[..., :2]      # (cx, cy) - (l, t) = (x0, y0)
        right_bottom = grids[None, :, :] + reg_preds[..., 2:]  # (cx, cy) + (r, b) = (x1, y1)
        # boxes.shape = [bs, total_anchor_num, 2+2=4]
        boxes = torch.cat([left_top, right_bottom], dim=-1)
        # 将预测的坐标, 类别置信度, 类别拼在一起 predictions.shape = [bs, total_anchor_num, 4+1+1]
        predictions = torch.cat([boxes, cls_scores, cls_classes], dim=-1)
        return predictions


    def gen_grid(self, lvl_size):
        """因为FCOS是anchor free, 因此需要生成网格, 解码时基于每个网格的位置进行解码
            Args:
                lvl_size: list, 每一个多尺度特征图的尺寸
            Returns: 
                grids: 所有尺度的网格中心坐标 [total_anchor_num(所有尺度concat), 2=(cx, cy)]

        """
        # 遍历所有尺度
        grids = []
        for size, stride in zip(lvl_size, self.strides):
            h, w = size
            shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
            shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
            # indexing="ij"符合 NumPy 的默认行为
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            # 左上角点->网格的中心点
            grid = torch.stack([shift_x, shift_y], -1) + stride // 2
            # grid.shape = [w*h, 2]
            grids.append(grid)

        grids = torch.cat(grids, dim=0)
        return grids