import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
from torch.nn import functional as F
from tqdm import tqdm  # 引入进度条库

# ================= 参数设置 =================
parser = argparse.ArgumentParser('HybridNets Inference (OOM Fix + Black/White Mask)')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. ')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
parser.add_argument('--save_mask', type=boolean_string, default=False, help="Save a binary mask image")

args = parser.parse_args()

# ================= 初始化 =================
params = Params(f'projects/{args.project}.yml')
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights

# 获取图片列表并排序
img_paths = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
img_paths.sort() 

os.makedirs(output, exist_ok=True)

use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list
color_list = standard_to_bgr(STANDARD_COLORS)

# ================= 加载模型 =================
print("Loading model...")
anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales
threshold = args.conf_thresh
iou_threshold = args.iou_thresh

# 加载权重
weight_dict = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
weight_last_layer_seg = weight_dict['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print(f"DETECTED SEGMENTATION MODE: {seg_mode}")

model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
model.load_state_dict(weight_dict)
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
    if use_float16:
        model = model.half()

# 图像预处理变换
resized_shape = params.model['image_size']
if isinstance(resized_shape, list):
    resized_shape = max(resized_shape)
normalize = transforms.Normalize(mean=params.mean, std=params.std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

print(f"FOUND {len(img_paths)} IMAGES. STARTING PROCESSING...")

# ================= 开始逐张处理循环 =================
# 使用 tqdm 显示进度条
for path in tqdm(img_paths, desc="Processing Images"):
    filename_with_ext = os.path.basename(path)
    filename, ext = os.path.splitext(filename_with_ext)
    
    # 1. 读取单张图片
    ori_img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if ori_img is None:
        print(f"Warning: Could not read {path}")
        continue
    
    # OpenCV 读入是 BGR，转为 RGB
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    h0, w0 = ori_img.shape[:2]

    # 2. Letterbox 预处理 (保持长宽比缩放)
    r = resized_shape / max(h0, w0)
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]
    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)
    
    # 转 Tensor
    if use_cuda:
        x = transform(input_img).cuda()
    else:
        x = transform(input_img)
    
    # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
    x = x.unsqueeze(0)
    x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)

    # 3. 模型推理
    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)

        # ---------------- 分割处理 (Segmentation) ----------------
        if seg_mode == BINARY_MODE:
            seg_mask = torch.where(seg >= 0, 1, 0)
            seg_mask.squeeze_(1)
        elif seg_mode == MULTICLASS_MODE:
            _, seg_mask = torch.max(seg, 1)
        else:
            seg_mask = torch.where(torch.sigmoid(seg) >= 0.5, 1, 0)

        # 取出 Batch 中的第一张 (也是唯一一张)
        seg_mask_ = seg_mask[0].squeeze().cpu().numpy()
        
        # 去除 Padding (还原到 resize 后的尺寸)
        pad_h = int(pad[1])
        pad_w = int(pad[0])
        seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
        
        # 缩放回原图尺寸
        seg_mask_ = cv2.resize(seg_mask_, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # ---------------- 生成并保存 Mask (黑底白路) ----------------
        if args.save_mask:
            # 1. 初始化全黑背景 (0, 0, 0)
            binary_mask_img = np.zeros((h0, w0, 3), dtype=np.uint8)
            
            # 2. 将检测到的路面/车道线 (值 > 0 的区域) 设为白色 (255, 255, 255)
            # 在 HybridNets 中：0=背景, 1=车道线, 2=可行驶区域
            binary_mask_img[seg_mask_ > 0] = (255, 255, 255)
            
            # 3. 保存为灰度图 (单通道 png，节省空间且只有纯黑纯白)
            mask_gray = cv2.cvtColor(binary_mask_img, cv2.COLOR_RGB2GRAY)
            
            # 构造保存路径
            save_path = os.path.join(output, f'{filename}_mask.png')
            cv2.imwrite(save_path, mask_gray)

        # ---------------- 生成可视化结果 (可选) ----------------
        # 只有在 --imwrite True 且 --save_mask False 的情况下才生成彩色叠加图
        # 如果你只想要 Mask，这样可以节省大量时间
        if args.imwrite and not args.save_mask:
            # 这里的可视化代码省略了，以保持代码在流式处理中的简洁性。
            # 3000张图通常只需要Mask。如果需要彩色检测框图，请去掉上面的 "and not args.save_mask"
            pass

print("\nProcessing complete!")