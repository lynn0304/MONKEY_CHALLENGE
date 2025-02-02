"""
It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import os
import cv2
import json
import shutil
import openslide
import numpy as np
import torch
from glob import glob
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import time
from torchvision import transforms, models
from PIL import Image 
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open(os.path.join(MODEL_PATH, "config.json"), "r") as file:
    data = json.load(file)

CONFIG_MODE = data.get("mode")
CONFIG_max_workers = data.get("max_workers")
CONFIG_yolo_imgsz = data.get("yolo_imgsz")
CONFIG_yolo_weight_name = data.get("yolo_weight_name")
CONFIG_cls_0 = os.path.join(MODEL_PATH, data.get("cls_weight_name_0"))
CONFIG_cls_1 = os.path.join(MODEL_PATH, data.get("cls_weight_name_1"))
CLS_MODEL = data.get("cls_model_type")
CONFIG_cls_figsize = data.get("cls_figsize")

def px_to_mm(px: int, spacing: float):
    """Convert pixels to millimeters."""
    return px * spacing / 1000

def infer_cls(model, image, return_pred='max'):
    if isinstance(image, np.ndarray):  # 若圖片為 numpy.ndarray
        image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((CONFIG_cls_figsize, CONFIG_cls_figsize)),  # 調整圖片大小
        transforms.ToTensor(),  # 轉換為 Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        if return_pred=='max':
            _, predicted = torch.max(outputs, 1)
        elif return_pred=='second':
            _, predictions=torch.topk(outputs, 2, dim=1)
            predicted = predictions[:, 1]
        probabilities = F.softmax(outputs, dim=1)
         

    return predicted.item(), probabilities.detach().cpu().numpy()[0]

def get_cls_image(rgb, x, y, w, h):
    x, y, w, h = int(x), int(y), int(w), int(h)
    height, width, _ = rgb.shape
    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(width, x + w // 2)
    y_max = min(height, y + h // 2)

    return rgb[y_min:y_max, x_min:x_max]

def write_json_file(location, content):
    """Writes a JSON file."""
    with open(location, 'w') as f:
        json.dump(content, f, indent=4)

def process_patch(x, y, yolo_imgsz, model_path, org_img, org_mask, spacing):
    """Process a single patch by applying mask and running YOLO inference."""
    cropped_mask = np.asarray(
        org_mask.read_region((x, y), 0, (yolo_imgsz, yolo_imgsz)))
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_RGBA2BGR)

    if not np.any(cropped_mask):
        return None  # Skip empty mask

    cropped_image = np.asarray(
        org_img.read_region((x, y), 0, (yolo_imgsz, yolo_imgsz)))
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGR)
    masked_img = cropped_image * cropped_mask
    model = YOLO(model_path)
    results = model.predict(masked_img, device=device, verbose=False)
    return results, x, y, spacing, cropped_image


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetV2, self).__init__()
        
        # 載入預訓練的 MobileNetV2 模型，並去掉最後的分類層
        mobilenetv2 = models.mobilenet_v2(pretrained=False)
        self.features = mobilenetv2.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(mobilenetv2.last_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.classifier(x)
        return x

def call_clssfiy(cls_model_path_0, cls_model_path_1, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    MODEL_efficientnet_b0_lympho_other_0 = cls_model_path_0
    MODEL_efficientnet_b0_lympho_other_1 = cls_model_path_1

    if model_type == 'efficientnet_b4':
        model_cls_0 = models.efficientnet_b4(pretrained=False)  # 不載入預訓練權重
        model_cls_0.classifier[1] = torch.nn.Linear(
            model_cls_0.classifier[1].in_features, num_classes)
        model_cls_1 = models.efficientnet_b4(pretrained=False)  # 不載入預訓練權重
        model_cls_1.classifier[1] = torch.nn.Linear(
            model_cls_1.classifier[1].in_features, num_classes)
    elif model_type == 'resnet':
        model_cls_0 = models.resnet50(pretrained=False)    # restnet-50 model
        model_cls_0.fc = nn.Linear(model_cls_0.fc.in_features, num_classes)
        model_cls_1 = models.resnet50(pretrained=False)    # restnet-50 model
        model_cls_1.fc = nn.Linear(model_cls_1.fc.in_features, num_classes)
    elif model_type == 'vistran':
        model_cls_0 = models.vit_b_16(pretrained=False)
        model_cls_0.heads.head = nn.Linear(model_cls_0.heads.head.in_features, num_classes)
        model_cls_1 = models.vit_b_16(pretrained=False)
        model_cls_1.heads.head = nn.Linear(model_cls_1.heads.head.in_features, num_classes)
    elif model_type == 'mobilenet':
        model_cls_0 = MobileNetV2(num_classes=3)
        model_cls_1 = MobileNetV2(num_classes=2)
    model_cls_0.load_state_dict(
        torch.load(MODEL_efficientnet_b0_lympho_other_0, map_location=device))
    model_cls_0 = model_cls_0.to(device)
    model_cls_0.eval()

    model_cls_1.load_state_dict(
        torch.load(MODEL_efficientnet_b0_lympho_other_1, map_location=device))
    model_cls_1 = model_cls_1.to(device)
    model_cls_1.eval()
    return model_cls_0, model_cls_1

def update_json_data(result, data_monocytes, data_lymphocytes,
                     data_inflammatory, counters):
    """Update JSON data dictionaries with prediction results."""
    results, x_offset, y_offset, spacing = result[:4]
    list_xywh = results[0].boxes.xywh.tolist()
    list_conf = results[0].boxes.conf.tolist()
    list_cls = results[0].boxes.cls.tolist()

    for i in range(len(list_xywh)):
        adjust_x = px_to_mm(list_xywh[i][0] + x_offset, spacing)
        adjust_y = px_to_mm(list_xywh[i][1] + y_offset, spacing)
        class_id = int(list_cls[i])
        prob = float(list_conf[i])
        if class_id == 0:  # Lymphocytes
            data_lymphocytes['points'].append({
                'name':
                f'Point {counters["lympho"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["lympho"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

        elif class_id == 1:  # Monocytes
            data_monocytes['points'].append({
                'name':
                f'Point {counters["mono"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["mono"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

def update_json_data_cls(result, data_monocytes, data_lymphocytes,
                     data_inflammatory, counters, model_cls_0, model_cls_1):
    """Update JSON data dictionaries with prediction results."""
    results, x_offset, y_offset, spacing, cropped_image_bgr = result
    list_xywh = results[0].boxes.xywh.tolist()
    list_conf = results[0].boxes.conf.tolist()
    list_cls = results[0].boxes.cls.tolist()
    cropped_image_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
    for i in range(len(list_xywh)):
        adjust_x = px_to_mm(list_xywh[i][0] + x_offset, spacing)
        adjust_y = px_to_mm(list_xywh[i][1] + y_offset, spacing)
        class_id = int(list_cls[i])
        prob = float(list_conf[i])
        temp_img = get_cls_image(cropped_image_rgb,
                                             list_xywh[i][0], list_xywh[i][1],
                                             list_xywh[i][2], list_xywh[i][3])
        if class_id == 0:
            cls_pred_0, probabilities_0 = infer_cls(
                model_cls_0, temp_img)
            if cls_pred_0 == 1:
                class_id = 2
        if class_id == 1:
            cls_pred_1, probabilities_1 = infer_cls(
                model_cls_1, temp_img)
            if cls_pred_1 == 0:
                class_id = 2
        if class_id == 0:  # Lymphocytes
            data_lymphocytes['points'].append({
                'name':
                f'Point {counters["lympho"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["lympho"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

        elif class_id == 1:  # Monocytes
            data_monocytes['points'].append({
                'name':
                f'Point {counters["mono"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["mono"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

def update_json_data_single(result, data_monocytes, data_lymphocytes,
                     data_inflammatory, counters, model_cls_0, model_cls_1):
    """Update JSON data dictionaries with prediction results."""
    results, x_offset, y_offset, spacing, cropped_image_bgr = result
    list_xywh = results[0].boxes.xywh.tolist()
    list_conf = results[0].boxes.conf.tolist()
    list_cls = results[0].boxes.cls.tolist()
    cropped_image_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
    for i in range(len(list_xywh)):
        adjust_x = px_to_mm(list_xywh[i][0] + x_offset, spacing)
        adjust_y = px_to_mm(list_xywh[i][1] + y_offset, spacing)
        class_id = int(list_cls[i])
        prob = float(list_conf[i])
        temp_img = get_cls_image(cropped_image_rgb,
                                             list_xywh[i][0], list_xywh[i][1],
                                             list_xywh[i][2], list_xywh[i][3])
        if class_id == 1:
            class_id = 2
        elif class_id == 0:
            cls_pred_0, probabilities_0 = infer_cls(
                model_cls_0, temp_img)
            if cls_pred_0 == 0:
                class_id = 0
                prob_cls = probabilities_0[class_id]
            else:
                cls_pred_1, probabilities_1 = infer_cls(
                model_cls_1, temp_img)
                if cls_pred_1 == 1:
                    class_id = 1
                    prob_cls = probabilities_1[class_id]
                else:
                    class_id = 0
        if class_id == 0:
            cls_pred_0, probabilities_0 = infer_cls(
                model_cls_0, temp_img)
            if cls_pred_0 == 1:
                class_id = 2
        if class_id == 1:
            cls_pred_1, probabilities_1 = infer_cls(
                model_cls_1, temp_img)
            if cls_pred_1 == 0:
                class_id = 2
        if class_id == 0:  # Lymphocytes
            data_lymphocytes['points'].append({
                'name':
                f'Point {counters["lympho"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["lympho"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

        elif class_id == 1:  # Monocytes
            data_monocytes['points'].append({
                'name':
                f'Point {counters["mono"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["mono"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

def prepare_nms(result, bboxes, confs, classes, model_cls_0, model_cls_1):
    results, x_offset, y_offset, spacing, cropped_image_bgr = result
    list_xywh = results[0].boxes.xywh.tolist()
    list_xyxy = results[0].boxes.xyxy.tolist()
    list_conf = results[0].boxes.conf.tolist()
    list_cls = results[0].boxes.cls.tolist()
    if np.any(list_xywh):
        LENGTH = len(list_xywh)
        for i in range(LENGTH):
            bboxes.append([list_xyxy[i][0]+x_offset, list_xyxy[i][1]+y_offset, list_xyxy[i][2]+x_offset, list_xyxy[i][3]+y_offset])
            confs.append(list_conf[i]) 
            cropped_image_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
            temp_img = get_cls_image(cropped_image_rgb,
                                    list_xywh[i][0], list_xywh[i][1],
                                    list_xywh[i][2], list_xywh[i][3])
            class_id = 0
            if list_cls[i] == 1:
                class_id = 2
            elif list_cls[i] == 0:
                cls_pred_0, probabilities_0 = infer_cls(
                    model_cls_0, temp_img)
                if cls_pred_0 == 0:
                    class_id = 0
                    prob_cls = probabilities_0[class_id]
                else:
                    cls_pred_1, probabilities_1 = infer_cls(
                    model_cls_1, temp_img)
                    if cls_pred_1 == 0:
                        class_id = 1
                        prob_cls = probabilities_1[class_id]
                    else:
                        class_id = 2
            classes.append(class_id)

def update_nms(data_monocytes, data_lymphocytes,
                data_inflammatory, counters, spacing,
                bboxes, confs, classes):
    print('Start NMS')
    bboxes = torch.tensor(bboxes)
    scores = torch.tensor(confs)
    labels = torch.tensor(classes)
    iou_threshold = 0.5
    unique_labels = torch.unique(labels)

    final_boxes = []
    final_scores = []
    final_labels = []

    for label in unique_labels:
        mask = (labels == label)
        class_boxes = bboxes[mask]
        class_scores = scores[mask]
        
        # Apply NMS to the current class's bounding boxes
        keep = ops.nms(class_boxes, class_scores, iou_threshold)
        
        # Collect the final boxes, scores, and labels for the current class
        final_boxes.append(class_boxes[keep])
        final_scores.append(class_scores[keep])
        final_labels.append(labels[mask][keep])
    final_boxes = torch.cat(final_boxes)
    final_scores = torch.cat(final_scores)
    final_labels = torch.cat(final_labels)

    final_boxes_list = final_boxes.tolist()
    final_scores_list = final_scores.tolist()
    final_labels_list = final_labels.tolist()
    for i in range(len(final_boxes_list)):
        adjust_x = px_to_mm((final_boxes_list[i][0]+final_boxes_list[i][2])*0.5, spacing)
        adjust_y = px_to_mm((final_boxes_list[i][1]+final_boxes_list[i][3])*0.5, spacing)
        class_id = int(final_labels_list[i])
        prob = float(final_scores_list[i])
        if class_id == 0:  # Lymphocytes
            data_lymphocytes['points'].append({
                'name':
                f'Point {counters["lympho"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["lympho"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

        elif class_id == 1:  # Monocytes
            data_monocytes['points'].append({
                'name':
                f'Point {counters["mono"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["mono"] += 1

            data_inflammatory['points'].append({
                'name':
                f'Point {counters["inflamm"]}',
                'point': [adjust_x, adjust_y, spacing],
                'probability':
                prob
            })
            counters["inflamm"] += 1

def save_json(output_path, data_monocytes, data_lymphocytes,
              data_inflammatory):
    """Save the JSON data dictionaries to files."""
    os.makedirs(output_path, exist_ok=True)

    write_json_file(os.path.join(output_path, "detected-monocytes.json"),
                    data_monocytes)
    write_json_file(os.path.join(output_path, "detected-lymphocytes.json"),
                    data_lymphocytes)
    write_json_file(
        os.path.join(output_path, "detected-inflammatory-cells.json"),
        data_inflammatory)

def run_yolo_multiclass(image_path, mask_path, output_path, yolo_imgsz):
    """Run YOLO multiclass detection on a whole-slide image."""
    start_time = time.time()

    data_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "points": [],
        "version": {
            "major": 1,
            "minor": 0
        }
    }
    data_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "points": [],
        "version": {
            "major": 1,
            "minor": 0
        }
    }
    data_inflammatory = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "points": [],
        "version": {
            "major": 1,
            "minor": 0
        }
    }

    counters = {"mono": 1, "lympho": 1, "inflamm": 1}

    spacing_min = 0.24199951445730394
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)

    org_img = openslide.OpenSlide(image_path)
    org_mask = openslide.OpenSlide(mask_path)
    org_img_w, org_img_h = org_img.level_dimensions[0]
    bboxes = []
    confs = []
    classes = []

    model_path = os.path.join(MODEL_PATH, CONFIG_yolo_weight_name)
    assert os.path.exists(model_path), "YOLO weight file does not exist!"
    # model = YOLO(model_path)
    if CONFIG_MODE == 'clssify' or CONFIG_MODE == 'singleclass' or CONFIG_MODE=='NMS':
        model_cls_0, model_cls_1 = call_clssfiy(CONFIG_cls_0, CONFIG_cls_1, CLS_MODEL)
    if CONFIG_MODE == 'NMS':
        ratio = 0.95
    else:
        ratio = 1
    with ThreadPoolExecutor(max_workers=CONFIG_max_workers) as executor:
        futures = [
            executor.submit(process_patch, x, y, yolo_imgsz, model_path, org_img,
                            org_mask, spacing)
            for y in range(0, org_img_h, int(yolo_imgsz*ratio))
            for x in range(0, org_img_w, int(yolo_imgsz*ratio))
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                if CONFIG_MODE == 'multiclass':
                    update_json_data(result, data_monocytes, data_lymphocytes,
                                    data_inflammatory, counters)
                elif CONFIG_MODE == 'clssify':
                    update_json_data_cls(result, data_monocytes, data_lymphocytes,
                                    data_inflammatory, counters, model_cls_0, model_cls_1)
                elif CONFIG_MODE == 'singleclass':
                    update_json_data_single(result, data_monocytes, data_lymphocytes,
                                    data_inflammatory, counters, model_cls_0, model_cls_1)
                elif CONFIG_MODE == 'NMS':
                    prepare_nms(result, bboxes, confs, classes, 
                                model_cls_0, model_cls_1)
    if CONFIG_MODE == 'NMS':
        update_nms(data_monocytes, data_lymphocytes, data_inflammatory, 
        counters, spacing, bboxes, confs, classes)
    save_json(output_path, data_monocytes, data_lymphocytes, data_inflammatory)
    print(f"Parallel processing time: {time.time() - start_time:.2f} seconds")


def run():
    """Main run function to execute the inference pipeline."""
    image_paths = glob(
        os.path.join(INPUT_PATH,
                     "images/kidney-transplant-biopsy-wsi-pas/*.tif"))
    mask_paths = glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif"))

    assert image_paths and mask_paths, "No input files found!"

    image_path = image_paths[0]

    mask_path = mask_paths[0]

    run_yolo_multiclass(image_path, mask_path, OUTPUT_PATH,
                        CONFIG_yolo_imgsz)

    detected_jsons = glob(os.path.join(OUTPUT_PATH, "*.json"))
    print(f"Detected output JSON files: {detected_jsons}")
    print("Done!")


if __name__ == "__main__":
    raise SystemExit(run())
