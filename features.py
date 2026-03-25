"""CLIP feature extraction utilities."""

import cv2
import numpy as np
import torch
from PIL import Image

from models import DEVICE


def extract_clip_features(img_crop, clip_model, clip_processor):
    """Extract normalised CLIP image features from a BGR/RGB crop."""
    pil_img = Image.fromarray(cv2.resize(img_crop, (224, 224)))
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        features = clip_model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            vec = features[0].detach().cpu().numpy()
        else:
            vec = features.pooler_output[0].detach().cpu().numpy()

    return vec / (np.linalg.norm(vec) + 1e-8)
