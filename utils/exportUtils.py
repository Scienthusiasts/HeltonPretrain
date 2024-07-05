import onnx
import torch
import os
from torch import nn
import onnxruntime

from utils.runnerUtils import *
from utils.utils import visInferResult


def torchExportOnnx(model:nn.Module, device:str, input_size:list[int], export_dir:str, export_name:str, ckpt_path=False, ):
    if not os.path.isdir(export_dir):os.makedirs(export_dir)
    export_path = os.path.join(export_dir, export_name)
    model = model.to(device)
    # 导入预训练权重
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 在调用torch.onnx.export之前，需要先创建输入数据x
    # 基于追踪（trace）的模型转换方法：给定一组输入，实际执行一遍模型，把这组输入对应的计算图记录下来，保存为 ONNX 格式(静态图)
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    with torch.no_grad():
        torch.onnx.export(
            model,                   # 要转换的模型
            x,                       # 模型的任意一组输入
            export_path,             # 导出的 ONNX 文件名
            opset_version=11,        # ONNX 算子集版本: https://onnx.ai/onnx/operators/
            input_names=['input'],   # 输入 Tensor 的名称, 如果不指定，会使用默认名字
            output_names=['cls_head', 'clip_head'],  # 输出 Tensor 的名称, 如果不指定，会使用默认名字
            # 动态输入输出设置:
            dynamic_axes = {
                # 哪个维度动态字典里索引就设置在哪个维度:
                'input':     {0: 'batch_size'},
                'cls_head':  {0: 'batch_size'},
                'clip_head': {0: 'batch_size'}
            }
        ) 

    # 读取 ONNX 模型
    onnx_model = onnx.load(export_path)
    # 检查模型格式是否正确
    onnx.checker.check_model(onnx_model)
    print('无报错, onnx模型导出成功')
    # 以可读的形式打印计算图
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # NETRON在线平台可视化模型结构 https://netron.app/





def onnxInferenceSingleImg(model, device:str, tf, img_path:str, onnx_path=False, ort_session=False, save_vis_path=False):
    '''基于onnx格式推理一张图像
    '''
    if onnx_path:
        # print(onnxruntime.get_device()) # GPU
        '''载入 onnx 模型，获取 ONNX Runtime 推理器'''
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    # 加载图像
    image = Image.open(img_path).convert('RGB')
    # Image 转numpy
    image = np.array(image)
    logits, sorted_id = model.onnxInfer(onnx_model=ort_session, device=device, image=image, tf=tf)
    '''是否可视化推理结果'''
    if save_vis_path:
        visInferResult(image, logits, sorted_id, model.cls_name, save_vis_path)
    return logits, sorted_id








def onnxInferenceBatchImgs(model:nn.Module, device:str, tf, img_dir:str, onnx_path:str):
    '''基于onnx格式推理图像s
    '''
    # print(onnxruntime.get_device()) # GPU
    '''载入 onnx 模型，获取 ONNX Runtime 推理器'''
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    cat_names = model.cls_name
    cat_names_dict = dict(zip(cat_names, [i for i in range(len(cat_names))]))
    # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
    true_list, pred_list, soft_list = [], [], []
    # 遍历每个类别下的所有图像:
    for cls_img_dir_name in tqdm(os.listdir(img_dir)):
        cls_img_dir = os.path.join(img_dir, cls_img_dir_name)
        for img_name in os.listdir(cls_img_dir):
            img_path = os.path.join(cls_img_dir, img_name)
            logits, sorted_id = onnxInferenceSingleImg(model, device, tf, img_path, save_vis_path=False, onnx_path=False, ort_session=ort_session)
            soft_list.append(logits)
            pred_list.append(sorted_id[0])
            true_list.append(cat_names_dict[cls_img_dir_name])
            
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)
    soft_list = np.array(soft_list)

    '''评估'''
    # 准确率
    acc = sum(pred_list==true_list) / pred_list.shape[0]
    # # 可视化混淆矩阵
    showComMatrix(true_list, pred_list, cat_names, './')
    # 绘制PR曲线
    PRs = drawPRCurve(cat_names, true_list, soft_list, './')
    # 计算每个类别的 AP, F1Score
    mAP, mF1Score, form = clacAP(PRs, cat_names)
    print('='*100)
    print(form)
    print('='*100)
    print(f"acc.: {acc} | mAP: {mAP} | mF1Score: {mF1Score}")
    print('='*100)
