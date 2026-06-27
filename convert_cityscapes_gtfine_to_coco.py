import json
import os
import glob
from PIL import Image

# Cityscapes category mapping (same as DA-RTDETR paper)
CATEGORIES = [
    {'id': 1, 'name': 'person'},
    {'id': 2, 'name': 'rider'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'truck'},
    {'id': 5, 'name': 'bus'},
    {'id': 6, 'name': 'train'},
    {'id': 7, 'name': 'motorcycle'},
    {'id': 8, 'name': 'bicycle'},
]

LABEL_MAP = {
    'person': 1, 'rider': 2, 'car': 3, 'truck': 4,
    'bus': 5, 'train': 6, 'motorcycle': 7, 'bicycle': 8
}

gtfine_dir = 'C:/confmix_data/real/gtFine/val'
img_dir = 'C:/confmix_data/real_yolo/cityscapes/val/images'

images = []
annotations = []
img_id = 1
ann_id = 1

json_files = glob.glob(os.path.join(gtfine_dir, '*', '*_polygons.json'))
print(f'Found {len(json_files)} annotation files')

for jf in sorted(json_files):
    with open(jf) as f:
        data = json.load(f)
    
    h, w = data['imgHeight'], data['imgWidth']
    fname = os.path.basename(jf).replace('_gtFine_polygons.json', '_leftImg8bit.png')
    
    images.append({'id': img_id, 'file_name': fname, 'height': h, 'width': w})
    
    for obj in data['objects']:
        label = obj['label']
        if label not in LABEL_MAP:
            continue
        
        poly = obj['polygon']
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))
        bw, bh = x2-x1, y2-y1
        
        if bw < 1 or bh < 1:
            continue
        
        annotations.append({
            'id': ann_id, 'image_id': img_id,
            'category_id': LABEL_MAP[label],
            'bbox': [x1, y1, bw, bh],
            'area': bw * bh,
            'iscrowd': 0,
            'segmentation': []
        })
        ann_id += 1
    
    img_id += 1

coco = {'images': images, 'annotations': annotations, 'categories': CATEGORIES}
out = 'C:/confmix_data/coco_json/cityscapes_val.json'
with open(out, 'w') as f:
    json.dump(coco, f)

car_anns = [a for a in annotations if a['category_id']==3]
print(f'Done! {len(images)} images, {len(annotations)} anns, {len(car_anns)} car anns')