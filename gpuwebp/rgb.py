"""RGB(A) -> YUV420 conversion (libwebp BT.601 fixed point)."""
import numpy as np


def rgb_to_yuv420(rgba):
    """rgba (H,W,4) uint8, H/W even -> y (H,W), u/v (H/2,W/2) uint8."""
    r = rgba[..., 0].astype(np.int64)
    g = rgba[..., 1].astype(np.int64)
    b = rgba[..., 2].astype(np.int64)
    YFIX = 16
    HALF = 1 << (YFIX - 1)

    def clip_uv(v):
        x = (v + (HALF << 2) + (128 << YFIX << 2)) >> (YFIX + 2)
        return np.clip(x, 0, 255)

    y = (16839 * r + 33059 * g + 6420 * b + HALF + (16 << YFIX)) >> YFIX
    H, W = y.shape
    y2 = y.reshape(H // 2, 2, W // 2, 2)
    r2 = r.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
    g2 = g.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
    b2 = b.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
    u = clip_uv(-9719 * r2 - 19081 * g2 + 28800 * b2)
    v = clip_uv(28800 * r2 - 24116 * g2 - 4684 * b2)
    return (np.clip(y, 0, 255).reshape(H, W).astype(np.uint8),
            u.astype(np.uint8), v.astype(np.uint8))
