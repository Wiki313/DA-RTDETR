import os
import json
from PIL import Image

img_dir = "C:/confmix_data/real_yolo/kitti/source/images"
label_dir = "C:/confmix_data/real_yolo/kitti/source/labels"
output_json = "C:/confmix_data/coco_json/kitti_train.json"

# KITTI only has car (class 0) → map to car in cityscapes (class 2)
# Cityscapes categories: person=0,rider=1,car=2,truck=3,bus=4,train=5,motorcycle=6,bicycle=7
categories = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "rider"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
    {"id": 6, "name": "train"},
    {"id": 7, "name": "motorcycle"},
    {"id": 8, "name": "bicycle"},
    {"id": 9, "name": "other"},
]

images = []
annotations = []
ann_id = 1

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

for img_id, img_file in enumerate(img_files, 1):
    img_path = os.path.join(img_dir, img_file)
    img = Image.open(img_path)
    w, h = img.size
    
    images.append({
        "id": img_id,
        "file_name": img_file,
        "width": w,
        "height": h
    })
    
    label_file = os.path.join(label_dir, img_file.replace('.png', '.txt'))
    if not os.path.exists(label_file):
        continue
    
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            # Convert YOLO to COCO bbox
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            bw_abs = bw * w
            bh_abs = bh * h
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 3,  # car
                "bbox": [x1, y1, bw_abs, bh_abs],
                "area": bw_abs * bh_abs,
                "iscrowd": 0
            })
            ann_id += 1

coco = {"images": images, "annotations": annotations, "categories": categories}
with open(output_json, 'w') as f:
    json.dump(coco, f)

print(f"Done! {len(images)} images, {len(annotations)} annotations")