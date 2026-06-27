import json
import random

random.seed(42)

with open('C:/confmix_data/coco_json/kitti_train.json') as f:
    data = json.load(f)

random.shuffle(data['images'])
images_subset = data['images'][:3000]
img_ids = {img['id'] for img in images_subset}
anns_subset = [a for a in data['annotations'] if a['image_id'] in img_ids]

data_subset = {
    'images': images_subset,
    'annotations': anns_subset,
    'categories': data['categories']
}

with open('C:/confmix_data/coco_json/kitti_train_3000.json', 'w') as f:
    json.dump(data_subset, f)

print(f'Done! {len(images_subset)} images, {len(anns_subset)} annotations')