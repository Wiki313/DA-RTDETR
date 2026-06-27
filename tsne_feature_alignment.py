"""
Generate a t-SNE feature-alignment figure (source vs. target backbone
features), in the style of Fig. 1 of Geng et al. 2026 -- but using YOUR
checkpoints and YOUR data, so it's a legitimate, original figure for your
own paper.

Usage:
    python tsne_feature_alignment.py \
        --config configs/rtdetr/city2foggy_rtdetrx.yml \
        --checkpoints output/rtdetrx_city2foggy/checkpoint0000.pth output/rtdetrx_city2foggy/checkpoint0047.pth \
        --labels "Source only" "Ours (epoch 47)" \
        --source-dir data/cityscapes/leftImg8bit/val \
        --target-dir data/foggy_cityscapes/leftImg8bit/val \
        --n-images 200 \
        --out tsne_alignment.png

Notes:
- "checkpoint0000.pth" (or your source-only/no-adaptation checkpoint) gives
  the "before adaptation" panel. Your best checkpoint (e.g. epoch 47/48 at
  56.3% mAP) gives the "after adaptation" panel.
- Adjust `extract_backbone_feature` if your model wrapper exposes backbone
  features differently (e.g. model.backbone(x) vs model.model.backbone(x)).
- This produces ONE PNG with one t-SNE panel per checkpoint passed in,
  source points in one color and target points in another -- directly
  analogous to Geng et al. Fig. 1, but from your own runs.
"""

import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_image_paths(directory, n, seed=0):
    directory = Path(directory)
    all_paths = sorted(directory.rglob("*.png")) + sorted(directory.rglob("*.jpg"))
    random.Random(seed).shuffle(all_paths)
    return all_paths[:n]


def preprocess(path, size=736):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    arr = arr.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(arr).float()


@torch.no_grad()
def extract_backbone_feature(model, img_tensor, device):
    """Returns a single pooled feature vector for one image.

    EDIT THIS if your model's backbone is accessed differently. For the
    RT-DETR-X / ResNet-101 setup, the backbone typically returns a list of
    multi-scale feature maps [S3, S4, S5]; we global-average-pool the
    deepest one (S5) to get a single vector per image.
    """
    x = img_tensor.unsqueeze(0).to(device)
    feats = model.backbone(x)          # adjust attribute path if needed
    deepest = feats[-1]                # S5, shape [1, C, H, W]
    pooled = deepest.mean(dim=[2, 3])  # global average pool -> [1, C]
    return pooled.squeeze(0).cpu().numpy()


def load_model(config_path, checkpoint_path, device):
    """EDIT THIS to match how your training script builds + loads the model.

    This is a placeholder showing the general shape -- it assumes your repo
    exposes a `build_model(cfg)` function and that checkpoints store a
    'model' (or 'ema') state dict, as is typical for RT-DETR-style repos.
    """
    import yaml
    from src.core import YAMLConfig  # adjust import to your repo layout

    cfg = YAMLConfig(config_path)
    model = cfg.model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", None) or ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--out", default="tsne_alignment.png")
    args = ap.parse_args()

    assert len(args.checkpoints) == len(args.labels), \
        "Need one --labels entry per --checkpoints entry"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_paths = load_image_paths(args.source_dir, args.n_images)
    target_paths = load_image_paths(args.target_dir, args.n_images)

    fig, axes = plt.subplots(1, len(args.checkpoints), figsize=(5 * len(args.checkpoints), 5))
    if len(args.checkpoints) == 1:
        axes = [axes]

    for ax, ckpt_path, label in zip(axes, args.checkpoints, args.labels):
        print(f"Loading {ckpt_path} ...")
        model = load_model(args.config, ckpt_path, device)

        feats, domains = [], []
        for p in source_paths:
            feats.append(extract_backbone_feature(model, preprocess(p), device))
            domains.append(0)
        for p in target_paths:
            feats.append(extract_backbone_feature(model, preprocess(p), device))
            domains.append(1)

        feats = np.stack(feats)
        domains = np.array(domains)

        print(f"Running t-SNE on {feats.shape[0]} points ({feats.shape[1]}-D) ...")
        proj = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(feats)

        ax.scatter(proj[domains == 0, 0], proj[domains == 0, 1],
                   c="#D85A30", s=8, alpha=0.6, label="Source")
        ax.scatter(proj[domains == 1, 0], proj[domains == 1, 1],
                   c="#378ADD", s=8, alpha=0.6, label="Target")
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    axes[0].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
