import torch
import json

# Check best checkpoint
ckpt = torch.load('output/rtdetrx_kitti2city/checkpoint.pth', map_location='cpu')
print('Keys in checkpoint:', list(ckpt.keys()))

# Show all AP50 from log
data = [json.loads(l) for l in open('output/rtdetrx_kitti2city/log.txt') if 'test_coco_eval_bbox' in l]
data = [d for d in data if d['test_coco_eval_bbox'][1] > 0.01]
print('\nEpoch | AP50')
for d in data:
    print(f'Epoch {d["epoch"]:2d}: AP50={d["test_coco_eval_bbox"][1]*100:.1f}%')