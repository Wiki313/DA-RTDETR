"""
Convert YOLO format to COCO format JSON
Works for Cityscapes and Foggy datasets
"""
import os
import json
import argparse
from pathlib import Path
from PIL import Image


CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def convert(img_dir, label_dir, output_json):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    images = []
    annotations = []
    ann_id = 1
    
    img_paths = sorted(list(img_dir.rglob('*.jpg')) + 
                       list(img_dir.rglob('*.png')))
    
    print(f'Found {len(img_paths)} images in {img_dir}')
    
    for img_id, img_path in enumerate(img_paths):
        # Get image size
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            w, h = 640, 640
        
        images.append({
            'id': img_id,
            'file_name': str(img_path),
            'width': w,
            'height': h,
        })
        
        # Find label file
        lbl_path = label_dir / img_path.relative_to(img_dir)
        lbl_path = lbl_path.with_suffix('.txt')
        
        # Also try replacing images with labels in path
        if not lbl_path.exists():
            lbl_str = str(img_path).replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
            lbl_str = lbl_str.replace('.jpg', '.txt').replace('.png', '.txt')
            lbl_path = Path(lbl_str)
        
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # Convert to COCO format (x1, y1, w, h)
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    bw_px = bw * w
                    bh_px = bh * h
                    
                    if bw_px <= 0 or bh_px <= 0:
                        continue
                    
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cls + 1,  # COCO is 1-indexed
                        'bbox': [x1, y1, bw_px, bh_px],
                        'area': bw_px * bh_px,
                        'iscrowd': 0,
                    })
                    ann_id += 1
    
    categories = [{'id': i+1, 'name': name} for i, name in enumerate(CLASSES)]
    
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    
    print(f'Saved {len(images)} images, {len(annotations)} annotations to {output_json}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--label-dir', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    label_dir = args.label_dir or args.img_dir.replace('images', 'labels')
    convert(args.img_dir, label_dir, args.output)


if __name__ == '__main__':
    main()
