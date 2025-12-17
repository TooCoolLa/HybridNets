import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm  # 确保已安装: pip install tqdm
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

# ==================== SkySeg 配置和函数定义 ====================
SKY_MODEL_PATH = "./skyseg.onnx" # 确保该文件位于脚本同级目录
SKY_THRESHOLD = 32
SKY_INPUT_SIZE = [320, 320]

def run_skyseg(onnx_session, input_size, image):
    """运行ONNX模型进行天空分割并返回结果图"""
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
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

parser = argparse.ArgumentParser('HybridNets + SkySeg Inference')
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
parser.add_argument('--speed_test', type=boolean_string, default=False, help='Measure inference latency')
parser.add_argument('--save_mask', type=boolean_string, default=False, help="Save a binary mask image (White bg, Black objects/Sky)")
parser.add_argument('-ps','--patch_size', type=int, default=2, help='Patch size for Handle')
args = parser.parse_args()
    
params = Params(f'projects/{args.project}.yml')
color_list_seg = {}
for seg_class in params.seg_list:
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights
img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
input_imgs = []
shapes = []
det_only_imgs = []

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales
threshold = args.conf_thresh
iou_threshold = args.iou_thresh
imshow = args.imshow
imwrite = args.imwrite
show_det = args.show_det
show_seg = args.show_seg
os.makedirs(output, exist_ok=True)
patch_size = args.patch_size
use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list

color_list = standard_to_bgr(STANDARD_COLORS)

# compute number of chunks using ceiling division to avoid an extra empty chunk
tqdm_chunk_size_count = (len(img_path) + patch_size - 1) // patch_size

# --- 加载 SkySeg 模型 ---
print('正在加载 SkySeg 模型...')
sky_session = None
if args.save_mask:
    if os.path.exists(SKY_MODEL_PATH):
        print(f"Loading SkySeg model from {SKY_MODEL_PATH}...")
        try:
            sky_session = onnxruntime.InferenceSession(SKY_MODEL_PATH)
        except Exception as e:
            print(f"ERROR loading SkySeg model: {e}")
            sky_session = None
    else:
        print(f"Warning: Sky model not found at {SKY_MODEL_PATH}, sky segmentation will be skipped for mask generation.")

# [修改点] 增加模型加载日志
print(f"Loading HybridNets weights from {weight}...")
weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE:", seg_mode)

model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
model.load_state_dict(weight)
model.requires_grad_(False)
model.eval()

resized_shape = params.model['image_size']
if isinstance(resized_shape, list):
    resized_shape = max(resized_shape)
normalize = transforms.Normalize(
    mean=params.mean, std=params.std
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
for chunk_idx in tqdm(range(tqdm_chunk_size_count), desc="Processing Chunks", unit="chunk"):
    chunk_start = chunk_idx * patch_size
    chunk_end = min((chunk_idx + 1) * patch_size, len(img_path))
    current_chunk_paths = img_path[chunk_start:chunk_end]
    print(f"Detected {len(current_chunk_paths)} images. Loading from disk...")
    handle_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in current_chunk_paths]
    print("Converting BGR to RGB...")
    handle_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in handle_imgs]
    print(f"FOUND {len(handle_imgs)} IMAGES")
    # reset per-chunk buffers to avoid accumulation across chunks
    input_imgs = []
    shapes = []
    det_only_imgs = []
    # --- 读取并预处理图像 ---
    # [修改点] 确保预处理进度条显示
    print("Preprocessing images (Resize & Letterbox)...")
    for ori_img in tqdm(handle_imgs, desc="Preprocessing", unit="img"):
        h0, w0 = ori_img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0) 
        input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]

        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)

        input_imgs.append(input_img)
        shapes.append(((h0, w0), ((h / h0, w / w0), pad)))
    

    ori_imgs = handle_imgs
    # [修改点] 增加 Stack 日志，因为这一步如果是3000张图会很慢且占内存
    print("Stacking tensors and moving to GPU/CPU...")
    if use_cuda:
        x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
    else:
        x = torch.stack([transform(fi) for fi in input_imgs], 0)

    x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)

    if use_cuda:
        model = model.cuda()
        if use_float16:
            model = model.half()

    with torch.no_grad():
        print("Running HybridNets Inference (Batch Mode)...")
        features, regression, classification, anchors, seg = model(x)

        # --- HybridNets Segmentation 处理 ---
        print("Processing segmentation masks...")
        seg_mask_list = []
        if seg_mode == BINARY_MODE:
            seg_mask = torch.where(seg >= 0, 1, 0)
            seg_mask.squeeze_(1)
            seg_mask_list.append(seg_mask)
        elif seg_mode == MULTICLASS_MODE:
            _, seg_mask = torch.max(seg, 1)
            seg_mask_list.append(seg_mask)
        else:
            seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
            seg_mask_list.pop(0)

        # 预处理 HybridNets 分割结果
        processed_seg_masks = [] 
        # [修改点] 确保分割处理进度条显示
        for i in tqdm(range(seg.size(0)), desc="Resizing Masks", unit="img"):
            current_img_seg_mask = None
            # append the original (for det-only visualization) once per image
            det_only_imgs.append(ori_imgs[i].copy())
            for seg_class_index, seg_mask in enumerate(seg_mask_list):
                seg_mask_ = seg_mask[i].squeeze().cpu().numpy()

                # 1. 裁剪 Letterbox Padding
                pad_h = int(shapes[i][1][1][1])
                pad_w = int(shapes[i][1][1][0])
                seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]

                # 2. 缩放回原图尺寸
                seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)

                current_img_seg_mask = seg_mask_

                # 可视化分割（彩色）
                color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
                color_seg = color_seg[..., ::-1] # RGB -> BGR
                color_mask = np.mean(color_seg, 2)
                seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]
                seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                seg_img = seg_img.astype(np.uint8)

                filename_with_ext = os.path.basename(current_chunk_paths[i])
                filename, _ = os.path.splitext(filename_with_ext)
                seg_filename = f'{output}/{filename}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                            f'{output}/{filename}_seg.jpg'
                if show_seg or seg_mode == MULTILABEL_MODE:
                    cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
            processed_seg_masks.append(current_img_seg_mask)

        # --- HybridNets Detection 处理 ---
        print("Post-processing bounding boxes...")
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

        # --- 最终循环：坐标映射 + 生成 Result 和 Mask ---
        print("Generating final outputs (Masks & Overlays)...")
        for i in tqdm(range(len(ori_imgs)), desc="Saving Results", unit="img"):
            # use the chunk-local path list to get the correct filename
            filename_with_ext = os.path.basename(current_chunk_paths[i])
            filename, ext = os.path.splitext(filename_with_ext)
            h0, w0 = ori_imgs[i].shape[:2]
            
            # 关键步骤：将检测框坐标从 模型尺寸 映射回 原图尺寸
            out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])

            # =========================================================
            # Mask 生成逻辑 (White Background, Black Objects/Sky)
            # =========================================================
            binary_mask_img = None
            if args.save_mask:
                # 1. 初始化全白背景 (255)
                binary_mask_img = np.ones((h0, w0, 3), dtype=np.uint8) * 255
                
                # 2. 绘制 HybridNets 道路分割 (黑色)
                if processed_seg_masks[i] is not None:
                    # 只需检查是否有非零像素 (即检测到的道路/车道线等)
                    binary_mask_img[processed_seg_masks[i] > 0] = (0, 0, 0)
                
                # 3. 计算并绘制 SkySeg 天空分割 (黑色)
                if sky_session is not None:
                    try:
                        # 注意: ori_imgs[i] 是 RGB 格式, run_skyseg 函数内部会处理成 BGR 然后再转 RGB 归一化
                        sky_map = run_skyseg(sky_session, SKY_INPUT_SIZE, cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))
                        sky_map_resized = cv2.resize(sky_map, (w0, h0), interpolation=cv2.INTER_NEAREST)
                        
                        # SkySeg 的输出值越低，越可能是天空。根据阈值设为黑色。
                        # [保留原有逻辑] 原始逻辑中：binary_mask_img[sky_map_resized > SKY_THRESHOLD] = (0, 0, 0)
                        # 意味着：如果天空置信度高(>32)，则设为黑色
                        binary_mask_img[sky_map_resized > SKY_THRESHOLD] = (0, 0, 0)
                    except Exception as e:
                        print(f"Sky segmentation failed for {filename}: {e}")

            # 4. 绘制 HybridNets 车辆检测框 (黑色)
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])
                
                # 绘制彩色框到原图
                plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                            color=color_list[get_index_label(obj, obj_list)])

                # 绘制黑色框到 Mask
                if args.save_mask and binary_mask_img is not None:
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w0, x2), min(h0, y2)
                    cv2.rectangle(binary_mask_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # 保存各种结果
            if show_det:
                cv2.imwrite(f'{output}/{filename}_det.jpg', cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))
            if imshow:
                cv2.imshow('img', ori_imgs[i])
                cv2.waitKey(0)
            if imwrite:
                cv2.imwrite(f'{output}/{filename_with_ext}', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))
            
            # 保存最终叠加的 Mask
            if args.save_mask and binary_mask_img is not None:
                # 转换为单通道灰度图保存，文件更小，且确保只有黑白两色
                mask_gray = cv2.cvtColor(binary_mask_img, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(f'{output}/{filename}.png', mask_gray)

if not args.speed_test:
    print("All tasks completed successfully.")
    exit(0)
print('running speed test...')
# Speed test code omitted...