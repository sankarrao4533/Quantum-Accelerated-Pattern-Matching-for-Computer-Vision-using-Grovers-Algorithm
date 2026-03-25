"""Similarity metrics: AE-QIP, edge-structure, and colour histogram."""

import cv2
import numpy as np


def ae_qip_similarity(v1, v2):
    """Cosine-based quantum-inspired probability similarity in [0, 1]."""
    cosine = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
    probability = (1 + cosine) / 2
    return probability


def edge_structure_similarity(img1, img2, size=128):
    """Edge-structure similarity in [0, 1]."""
    g1 = cv2.cvtColor(cv2.resize(img1, (size, size)), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (size, size)), cv2.COLOR_RGB2GRAY)

    e1 = cv2.Canny(g1, 80, 180)
    e2 = cv2.Canny(g2, 80, 180)

    if np.count_nonzero(e1) < 20 or np.count_nonzero(e2) < 20:
        corr = np.corrcoef(g1.flatten(), g2.flatten())[0, 1]
        if np.isnan(corr):
            return 0.5
        return float(np.clip((corr + 1) / 2, 0.0, 1.0))

    v1 = e1.astype(np.float32).flatten()
    v2 = e2.astype(np.float32).flatten()
    cos = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
    return float(np.clip(cos, 0.0, 1.0))


def color_hist_similarity(img1, img2, size=128):
    """HSV histogram similarity in [0, 1]."""
    h1 = cv2.cvtColor(cv2.resize(img1, (size, size)), cv2.COLOR_RGB2HSV)
    h2 = cv2.cvtColor(cv2.resize(img2, (size, size)), cv2.COLOR_RGB2HSV)

    hist1 = cv2.calcHist([h1], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([h2], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1) / 2, 0.0, 1.0))
