import json
import os
import random
import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

# --- Model related imports ---
import utils.config as config
from model import build_segmenter
from utils.dataset import tokenize

# ==============================================================================
# Part 1: Logic from `prepare_custom_data.py` (Unchanged)
# ==============================================================================
# ... (LABEL_ATTRIBUTES, get_grid_location, generate_sentence functions remain the same) ...
LABEL_ATTRIBUTES = {
    "flower": {"class_name": "yellow wildflower", "generic": ["flowering weed", "plant"], "feature": ["with small yellow petals", "that has yellow blossoms"], "context_singular": "a single yellow wildflower", "context_plural": "a patch of yellow wildflowers"},
    "jjc": {"class_name": "tussock grass", "generic": ["grass", "weed", "vegetation"], "feature": ["with long, thin blades", "that looks dry and straw-colored"], "context_singular": "one clump of grass", "context_plural": "an area of grass"},
    "zmc": {"class_name": "poisonous weed", "generic": ["plant", "vegetation", "weed"], "feature": ["with green leaves", "growing in a scattered patch"], "context_singular": "a single plant", "context_plural": "a patch of vegetation"},
    "pzy": {"class_name": "leafy plant", "generic": ["broadleaf weed", "leafy vegetation"], "feature": ["with silvery-green leaves", "that has wide, oval-shaped leaves"], "context_singular": "a single leafy plant", "context_plural": "a cluster of leafy plants"}
}
LABEL_ATTRIBUTES["Thermopsis lanceolate"] = LABEL_ATTRIBUTES["pzy"]
AREA_THRESHOLD_RATIO = 0.01
def get_grid_location(x, y, width, height):
    row = int(y / (height / 3)); col = int(x / (width / 3))
    locations = [["top left", "top center", "top right"], ["middle left", "center", "middle right"], ["bottom left", "bottom center", "bottom right"]]
    return locations[row][col]
def generate_sentence(obj):
    label = obj['label']; attributes = LABEL_ATTRIBUTES.get(label)
    if not attributes: return f"the object in the {obj['location']}"
    img_h, img_w = obj['img_h'], obj['img_w']
    total_area = img_h * img_w
    polygon = np.array(obj['polygon'], dtype=np.int32)
    object_area = cv2.contourArea(polygon)
    is_plural = (object_area / total_area) > AREA_THRESHOLD_RATIO
    general_desc = attributes['context_plural'] if is_plural else attributes['context_singular']
    specific_name = attributes['class_name']
    sentence_structures = [
        f"The {specific_name} which looks like {general_desc} in the {obj['location']}.",
        f"A photo of {general_desc}, which is a type of {specific_name}, located in the {obj['location']} area.",
        f"The segmentation mask for the {specific_name} at the {obj['location']}.",
    ]
    return " ".join(random.choice(sentence_structures).split())


# ==============================================================================
# Part 2: Visualization Function (Unchanged)
# ==============================================================================
# ... (visualize_with_prediction and generate_mask_from_polygon functions remain the same) ...
def visualize_with_prediction(image_path, gt_mask, pred_mask, output_path):
    """
    Generates a three-panel figure without a text banner:
    (a) Image, (b) Ground Truth Overlay, (c) Prediction Overlay
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 设定统一的面板尺寸，例如 480x360
    target_h, target_w = 360, 480
    
    # Panel A: Original Image
    panel_a = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # 将标题改为更通用的 "Image"
    cv2.putText(panel_a, "(a) Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Panel B: Ground Truth Overlay
    # 创建一个新的底图副本，而不是复用 panel_a，以避免标题重叠
    panel_b = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gt_mask_resized = cv2.resize(gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    gt_color_overlay = np.zeros_like(panel_b)
    gt_mask_color = [0, 255, 0]  # GREEN for Ground Truth
    gt_color_overlay[gt_mask_resized > 0] = gt_mask_color
    cv2.addWeighted(panel_b, 1, gt_color_overlay, 0.6, 0, panel_b)
    cv2.putText(panel_b, "(b) Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Panel C: Prediction Overlay
    # 同样创建新的底图副本
    panel_c = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pred_mask_resized = cv2.resize(pred_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    pred_color_overlay = np.zeros_like(panel_c)
    pred_mask_color = [255, 0, 255] # MAGENTA for Prediction
    pred_color_overlay[pred_mask_resized > 0] = pred_mask_color
    cv2.addWeighted(panel_c, 1, pred_color_overlay, 0.6, 0, panel_c)
    cv2.putText(panel_c, "(c) Our Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # <<< 直接横向拼接三个面板，不再有文字横幅 >>>
    final_figure = cv2.hconcat([panel_a, panel_b, panel_c])
    
    cv2.imwrite(output_path, final_figure)
    # 更新打印信息
    print(f"Figure (without text) saved successfully to {output_path}")

def generate_mask_from_polygon(height, width, polygon):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

# ==============================================================================
# <<< Part 3: 新的配置区域，用于指定图片 >>>
# ==============================================================================

# 在这里列出您想要处理的每一张图片。
# 每一项都需要提供：
# 'image_path': 图片的完整路径。
# 'annotation_path': 对应标注json文件的完整路径。
# 'object_index': 在json文件中，您想分割第几个物体（从0开始）。
#                 如果json中只有一个物体，就写0。
# 'custom_text': (可选) 如果您不想用脚本自动生成的句子，可以在这里提供一个自定义的句子。
SPECIFIC_IMAGES_TO_PROCESS = [
    {
        'image_path': '/home/featurize/work/Project/project/my_etris/ETRIS-main/img/218_1080_1080.jpg',
        'annotation_path': '/home/featurize/work/data/flower/flower1/218_1080_1080.json',
        'object_index': 0,
        # 'custom_text': 'The yellow flower at the bottom' # <-- 可选的自定义文本
    },
    {
        'image_path': '/home/featurize/work/Project/project/my_etris/ETRIS-main/img/269.jpg',
        'annotation_path': '/home/featurize/work/data/pzy/pzy3/269.json',
        'object_index': 1, # 假设这张图的json里有多个物体，我们选择第二个
    },
    {
        'image_path': '/home/featurize/work/Project/project/my_etris/ETRIS-main/img/JJC_dj619_1080_2160.jpg',
        'annotation_path': '/home/featurize/work/data/jjc/JJC_dj619_1080_2160.json',
        'object_index': 0,
    },
    {
        'image_path': '/home/featurize/work/Project/project/my_etris/ETRIS-main/img/dj175_0_2760.jpg',
        'annotation_path': '/home/featurize/work/data/zmc/zmc_label/dj175_0_2760.json',
        'object_index': 0,
    },
    # --- 在这里继续添加更多您想处理的图片 ---
]


# ==============================================================================
# <<< Part 4: 修改后的主工作流，处理指定的图片列表 >>>
# ==============================================================================

@torch.no_grad()
def generate_figures_for_specific_images(cfg, model, device, output_dir, image_list):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Pre-process setup (from your dataset class)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    input_size = (cfg.input_size, cfg.input_size)

    # <<< 不再遍历文件夹，而是遍历您指定的列表 >>>
    for i, item in enumerate(tqdm(image_list, desc="Processing specified images")):
        img_path = item['image_path']
        ann_path = item['annotation_path']
        obj_idx = item['object_index']
        custom_text = item.get('custom_text') # 获取自定义文本，如果没有则为None

        # --- 检查文件是否存在 ---
        if not os.path.exists(img_path):
            print(f"\n错误: 图片文件未找到，跳过此项: {img_path}")
            continue
        if not os.path.exists(ann_path):
            print(f"\n错误: 标注文件未找到，跳过此项: {ann_path}")
            continue

        try:
            # --- 读取图片和标注 ---
            image = cv2.imread(img_path)
            img_h, img_w, _ = image.shape
            with open(ann_path, 'r') as f:
                raw_ann = json.load(f)

            # --- 定位到指定的物体 ---
            polygons = [shape for shape in raw_ann.get('shapes', []) if shape.get('shape_type') == 'polygon']
            if not polygons or obj_idx >= len(polygons):
                print(f"\n错误: 在 {ann_path} 中找不到索引为 {obj_idx} 的多边形物体，跳过。")
                continue
            
            target_shape = polygons[obj_idx]
            
            # 1. 准备 Ground Truth 和 Text
            polygon = target_shape.get('points')
            gt_mask = generate_mask_from_polygon(img_h, img_w, polygon)
            
            if custom_text:
                text = custom_text
            else:
                # 如果没有自定义文本，则自动生成
                label = target_shape.get('label')
                points_arr = np.array(polygon, dtype=np.float32)
                cx, cy = np.mean(points_arr, axis=0)
                location = get_grid_location(cx, cy, img_w, img_w)
                obj_for_sentence = {"label": label, "location": location, "polygon": polygon, "img_h": img_h, "img_w": img_w}
                text = generate_sentence(obj_for_sentence)
            
            # 2. 准备模型输入
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_CUBIC)
            img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float()
            img_tensor = (img_tensor / 255.0 - mean) / std
            img_tensor = img_tensor.unsqueeze(0).to(device)
            text_tensor = tokenize(text, cfg.word_len, True).to(device)
            
            # 3. 运行推理
            pred_logit = model(img_tensor, text_tensor)
            pred_prob = torch.sigmoid(pred_logit)
            
            # 4. 后处理预测结果
            pred_prob_resized = F.interpolate(pred_prob, size=(img_h, img_w), mode='bilinear', align_corners=False)
            pred_prob_np = pred_prob_resized.squeeze().cpu().numpy()
            pred_mask = (pred_prob_np > 0.35).astype(np.uint8) * 255 # 使用您的阈值

            # 5. 生成可视化结果
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_file_path = os.path.join(output_dir, f"figure_pred_{base_name}_obj{obj_idx}.jpg")
            print(f"\n处理样本: {base_name}.jpg")
            print(f"  使用文本: {text}")
            visualize_with_prediction(img_path, gt_mask, pred_mask, text, output_file_path)

        except Exception as e:
            print(f"\n处理文件 {img_path} 时发生未知错误: {e}")
            continue

    print(f"\n所有指定的图片已处理完毕，结果保存在 '{output_dir}' 文件夹中。")

# ==============================================================================
# <<< Part 5: 修改后的主程序入口 >>>
# ==============================================================================
if __name__ == '__main__':
    # --- 配置 ---
    CONFIG_FILE = 'config/refcoco/bridge_r101.yaml'
    MODEL_PATH = 'exp/refcoco/Lar-Net_CLIPResNet-101_6.94M/best_model.pth'
    OUTPUT_DIR = './paper_figures_from_specific_images'

    # --- 设置模型 ---
    print("--- 正在设置模型 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.load_cfg_from_cfg_file(CONFIG_FILE)
    model, _ = build_segmenter(cfg)
    
    if os.path.exists(MODEL_PATH):
        print(f"正在从 {MODEL_PATH} 加载模型权重")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"警告: 模型路径 {MODEL_PATH} 未找到。将使用随机初始化的模型进行可视化。")
        
    model.to(device)
    print("模型设置完成。")
    
    # --- 运行新的工作流 ---
    generate_figures_for_specific_images(
        cfg=cfg,
        model=model,
        device=device,
        output_dir=OUTPUT_DIR,
        image_list=SPECIFIC_IMAGES_TO_PROCESS  # <<< 传入我们新定义的列表
    )