import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
from torch.nn import functional as F
import onnxruntime
import copy
import gc # 引入垃圾回收

# ==================== SkySeg 配置 ====================
SKY_MODEL_PATH = "./skyseg.onnx"
SKY_THRESHOLD = 32
SKY_INPUT_SIZE = [320, 320]

def run_skyseg(onnx_session, input_size, image):
    """运行ONNX模型进行天空分割"""
    # 这里的 image 已经是 RGB 格式
    resize_image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    x = resize_image.astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    onnx_result = onnx_session.run([output_name], {input_name: x})
    onnx_result = np.array(onnx_result).squeeze()
    
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    
    if max_value - min_value == 0:
        onnx_result = np.zeros_like(onnx_result)
    else:
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
    
    onnx_result *= 255
    return onnx_result.astype("uint8")
# ========================================================

parser = argparse.ArgumentParser('HybridNets Inference (Low Memory Mode)')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file')
parser.add_argument('-bb', '--backbone', type=str, help='Backbone replacement')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of backbone')
parser.add_argument('--source', type=str, default='demo/image', help='Input folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Detection only")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Segmentation only")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16")
parser.add_argument('--save_mask', type=boolean_string, default=False, help="Save binary mask")
parser.add_argument('--speed_test', type=boolean_string, default=False) # 保留参数定义以免报错

args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')
color_list_seg = {}
for seg_class in params.seg_list:
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))

source = args.source
if source.endswith("/"): source = source[:-1]
output = args.output
if output.endswith("/"): output = output[:-1]
os.makedirs(output, exist_ok=True)

# 获取文件列表
img_paths = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
img_paths.sort()
print(f"FOUND {len(img_paths)} IMAGES")

# --- 初始化模型 ---
use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list
color_list = standard_to_bgr(STANDARD_COLORS)

# 加载 SkySeg
sky_session = None
if args.save_mask:
    if os.path.exists(SKY_MODEL_PATH):
        try:
            sky_session = onnxruntime.InferenceSession(SKY_MODEL_PATH)
            print(f"Loaded SkySeg model.")
        except Exception as e:
            print(f"ERROR loading SkySeg: {e}")
            sky_session = None

# 预处理变换
resized_shape = params.model['image_size']
if isinstance(resized_shape, list): resized_shape = max(resized_shape)
normalize = transforms.Normalize(mean=params.mean, std=params.std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

# 加载 HybridNets
print(f"Loading weights from {args.load_weights}...")
weight = torch.load(args.load_weights, map_location='cuda' if use_cuda else 'cpu', weights_only=False)
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    seg_mode = params.seg_multilabel and MULTILABEL_MODE or MULTICLASS_MODE
print(f"SEGMENTATION MODE: {seg_mode}")

model = HybridNetsBackbone(compound_coef=args.compound_coef, num_classes=len(obj_list), 
                           ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales), 
                           seg_classes=len(seg_list), backbone_name=args.backbone, seg_mode=seg_mode)
model.load_state_dict(weight)
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
    if use_float16: model = model.half()

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

print(f"STARTING STREAM PROCESSING (Batch Size = 1)...")

# ================= 核心修改：流式处理循环 =================
# 每次只处理 1 张图片，处理完立即释放内存
for path in tqdm(img_paths, desc="Processing"):
    filename = os.path.splitext(os.path.basename(path))[0]
    
    # 1. 读图
    ori_img_bgr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if ori_img_bgr is None: continue
    ori_img = cv2.cvtColor(ori_img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = ori_img.shape[:2]

    # 2. 预处理
    r = resized_shape / max(h0, w0)
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]
    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)
    
    # 转 Tensor
    if use_cuda:
        x = transform(input_img).cuda()
    else:
        x = transform(input_img)
    x = x.unsqueeze(0).to(torch.float16 if use_cuda and use_float16 else torch.float32)

    # 3. 推理
    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)
        
        # Seg 处理
        if seg_mode == BINARY_MODE:
            seg_mask = torch.where(seg >= 0, 1, 0).squeeze(1)
        elif seg_mode == MULTICLASS_MODE:
            _, seg_mask = torch.max(seg, 1)
        else:
            seg_mask = torch.where(torch.sigmoid(seg) >= 0.5, 1, 0)
        
        seg_mask_ = seg_mask[0].squeeze().cpu().numpy()
        
        # 还原 Mask 尺寸
        pad_h, pad_w = int(pad[1]), int(pad[0])
        seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
        seg_mask_ = cv2.resize(seg_mask_, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # Det 处理
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, args.conf_thresh, args.iou_thresh)
        out = out[0]
        out['rois'] = scale_coords(ori_img.shape[:2], out['rois'], (h0, w0), ((h/h0, w/w0), pad))

    # 4. 生成 Mask (黑底白路)
    if args.save_mask:
        # 初始化全白
        binary_mask = np.ones((h0, w0, 3), dtype=np.uint8) * 255
        
        # 涂黑 HybridNets 路面 (稍后反转颜色)
        # 注意：这里逻辑先按你的"白底黑路"来，最后再统一反转或者按需调整
        # 为了符合你刚才要求的"黑底白路"最终效果，我们这里直接构建：
        
        # --- 新逻辑：直接构建黑底白路 ---
        final_mask = np.zeros((h0, w0), dtype=np.uint8) # 全黑单通道
        
        # 路面设为白色 (255)
        final_mask[seg_mask_ > 0] = 255 
        
        # 处理 SkySeg (如果天空判定为 True，则保持黑色)
        # 逻辑：只需要把非天空、非路面的部分保持黑色。
        # 在黑底白路模式下，其实 SkySeg 作用不大了（因为背景本来就是黑的），
        # 除非你想把"路面误检到天空上"的部分扣掉。
        if sky_session is not None:
             # 计算天空
             sky_map = run_skyseg(sky_session, SKY_INPUT_SIZE, ori_img) # 传入 RGB
             sky_map = cv2.resize(sky_map, (w0, h0), interpolation=cv2.INTER_NEAREST)
             # 如果是天空 (值 < 阈值，越小越天)，则强制涂黑
             final_mask[sky_map < SKY_THRESHOLD] = 0

        # 处理车辆检测框 (涂黑)
        for j in range(len(out['rois'])):
            x1, y1, x2, y2 = out['rois'][j].astype(int)
            # 将车辆位置涂黑 (0)
            cv2.rectangle(final_mask, (x1, y1), (x2, y2), 0, -1)

        # 保存 Mask
        cv2.imwrite(f'{output}/{filename}_mask.png', final_mask)

    # 5. 生成可视化图 (可选)
    if args.imwrite and not args.save_mask:
        # 仅当不需要 mask 时才生成这个，节省时间
        # (这里省略可视化代码以节省篇幅，核心是 mask)
        pass

    # 手动释放内存，防止累积
    del x, features, regression, classification, anchors, seg, out, ori_img, ori_img_bgr
    # torch.cuda.empty_cache() # 可选，稍微影响速度但更省显存

print("Processing complete!")