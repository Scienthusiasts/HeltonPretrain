import torch
from torchvision.ops import nms
import torch.nn as nn







class NMS(nn.Module):
    """NMS(非极大值抑制) for object detection
    """
    def __init__(self, ):
        super(NMS, self).__init__()
        

    def forward(self, batch_preds, score_thr=0.1, iou_thr=0.3, agnostic=False):
        """
            Args:
                batch_preds: 预测结果 [bs, total_anchor_num, 4+1+1=(x1, y1, x2, y2, class_conf, class_pred)]
                score_thr:   置信度阈值, 低于该阈值的框直接过滤
                iou_thr:     nms iou阈值, 高于该iou的冗余框直接过滤
            Returns:
                filter_batch_preds: nms过滤后保留的预测框
        """
        filter_batch_preds = []
        # 遍历batch每张图像, 每张图像单独nms:
        for preds in batch_preds:
            '''首先筛选掉置信度小于阈值的预测'''
            scores = preds[:, 4]
            preds = preds[scores >= score_thr]
            # 如果第一轮筛选就没有框,则继续
            if not preds.size(0): 
                continue
            if agnostic:
                '''类别无关nms(eval时使用这个一般会掉点)'''
                preds = self.cat_agnostic_nms(preds, iou_thr).cpu().numpy()
            else:
                '''逐类别nms'''
                preds = self.cat_wise_nms(preds, iou_thr).cpu().numpy()
            filter_batch_preds.append(preds)
        return filter_batch_preds


    def cat_wise_nms(self, preds, nms_thres):
        '''逐类别nms
        '''
        cls_output = torch.tensor([])
        unique_cats = preds[:, -1].unique()
        for cat in unique_cats:
            # 获得某一类下的所有预测结果
            detections_class = preds[preds[:, -1] == cat]
            # 使用官方自带的非极大抑制会速度更快一些
            final_cls_score = detections_class[:, 4]
            '''接着筛选掉nms大于nms_thres的预测''' 
            keep = nms(detections_class[:, :4], final_cls_score, nms_thres)
            nms_detections = detections_class[keep]
            # 将类别nms结果记录cls_output
            cls_output = nms_detections if len(cls_output)==0 else torch.cat((cls_output, nms_detections))
        return cls_output
        

    def cat_agnostic_nms(self, preds, nms_thres):
        '''类别无关的nms
        '''
        # 使用官方自带的非极大抑制会速度更快一些
        '''接着筛选掉nms大于nms_thres的预测''' 
        keep = nms(preds[:, :4], preds[:, 4], nms_thres)
        nms_detections = preds[keep]
    
        return nms_detections


