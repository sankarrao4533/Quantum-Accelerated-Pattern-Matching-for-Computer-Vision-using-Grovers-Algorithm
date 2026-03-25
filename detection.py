"""Scene region detection: contour-based grid tiles and uniform grid split."""

import cv2
import numpy as np


def detect_grid_tiles(scene_img, min_tile_area_ratio=0.005, max_tiles=32):
    """
    Detect individual tiles/cells in a grid image using contour detection.
    Works for pattern grids, mosaics, zentangle sheets, etc.
    """
    gray = cv2.cvtColor(scene_img, cv2.COLOR_RGB2GRAY)
    img_area = scene_img.shape[0] * scene_img.shape[1]
    min_area = int(img_area * min_tile_area_ratio)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        if aspect < 0.3 or aspect > 3.0:
            continue
        pad = max(2, int(min(w, h) * 0.03))
        x1, y1 = x + pad, y + pad
        x2, y2 = x + w - pad, y + h - pad
        if x2 <= x1 or y2 <= y1:
            continue
        crop = scene_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tiles.append((x1, y1, x2, y2, crop))

    tiles.sort(key=lambda t: (t[1] // 50, t[0]))
    return tiles[:max_tiles]


def uniform_grid_split(scene_img, rows, cols):
    """Split image into a uniform grid of rows x cols tiles."""
    h, w = scene_img.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            x1 = c * tile_w
            y2 = min((r + 1) * tile_h, h)
            x2 = min((c + 1) * tile_w, w)
            pad = max(2, int(min(tile_w, tile_h) * 0.05))
            cy1, cx1 = y1 + pad, x1 + pad
            cy2, cx2 = y2 - pad, x2 - pad
            if cy2 <= cy1 or cx2 <= cx1:
                continue
            crop = scene_img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            tiles.append((cx1, cy1, cx2, cy2, crop))
    return tiles
