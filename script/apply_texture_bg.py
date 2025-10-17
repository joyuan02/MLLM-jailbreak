# apply_texture_bg.py
# 用随机纹理替换合成图中的“白色背景”。支持保持原有子目录结构输出。

import argparse
import glob
import os
import random
from typing import List

from PIL import Image, ImageOps, ImageFilter
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return files


def choose_random_texture(texture_dir: str) -> str:
    all_tex = []
    for root, _, files in os.walk(texture_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                all_tex.append(os.path.join(root, f))
    if not all_tex:
        raise RuntimeError(f"[ERR] 找不到纹理：{texture_dir}")
    all_tex.sort() 
    return random.choice(all_tex)


def augment_texture(img: Image.Image) -> Image.Image:
    # 随机旋转(0/90/180/270) + 随机水平/垂直翻转
    rot_k = random.choice([0, 1, 2, 3])
    if rot_k:
        img = img.rotate(90 * rot_k, expand=True)
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.3:
        img = ImageOps.flip(img)
    return img


def tile_texture_to_size(tex: Image.Image, size_w: int, size_h: int) -> Image.Image:
    tw, th = tex.size
    if tw == 0 or th == 0:
        tex = Image.new("RGB", (32, 32), (200, 200, 200))
        tw, th = tex.size

    # 基于目标尺寸的比例缩放
    scale = max(size_w / max(tw, 1), size_h / max(th, 1)) * random.uniform(0.4, 1.2)
    new_w = max(1, int(tw * scale))
    new_h = max(1, int(th * scale))
    tex = tex.resize((new_w, new_h), Image.BICUBIC)

    # 平铺到目标大小
    out = Image.new("RGB", (size_w, size_h))
    for y in range(0, size_h, new_h):
        for x in range(0, size_w, new_w):
            out.paste(tex, (x, y))
    return out


def build_white_mask(img: Image.Image, white_thresh: int) -> Image.Image:
    """
    生成替换区域mask：像素(R,G,B)都 >= white_thresh 视为背景(=255)，否则为前景(=0)。
    对mask做轻微膨胀+模糊，边缘更自然。
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    mask_bool = (arr[:, :, 0] >= white_thresh) & (arr[:, :, 1] >= white_thresh) & (arr[:, :, 2] >= white_thresh)
    mask = Image.fromarray((mask_bool.astype(np.uint8) * 255))

    # 边缘处理：膨胀一点再高斯模糊，避免“白边”
    if mask.width >= 16 and mask.height >= 16:
        mask = mask.filter(ImageFilter.MaxFilter(size=3))   # 轻微膨胀
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    return mask


def replace_white_with_texture(img: Image.Image, texture_dir: str, white_thresh: int) -> Image.Image:
    tex_path = choose_random_texture(texture_dir)
    tex = Image.open(tex_path).convert("RGB")
    tex = augment_texture(tex)

    w, h = img.size
    bg = tile_texture_to_size(tex, w, h)  # 生成与目标同尺寸的平铺纹理
    mask = build_white_mask(img, white_thresh)  # 255=替换为纹理, 0=保留原图

    # 用 mask 选择纹理或原图
    out = Image.composite(bg, img, mask)
    return out


def ensure_ext(path: str, fallback_ext: str = ".jpg") -> str:
    root, ext = os.path.splitext(path)
    if ext == "":
        return root + fallback_ext
    return path


def process_one(in_path: str, in_root: str, out_root: str, texture_dir: str, white_thresh: int):
    img = Image.open(in_path).convert("RGB")
    out_img = replace_white_with_texture(img, texture_dir, white_thresh)

    # 复刻子目录结构
    rel = os.path.relpath(in_path, start=in_root)  # 例如 "Animal\\18.jpg"
    out_path = os.path.join(out_root, rel)
    out_path = ensure_ext(out_path, ".jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_img.save(out_path, quality=95)
    print(f"[OK] {in_path} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", required=True, help="输入图像glob，如 .\\distraction_images\\CS-DJ_best_method\\*\\*.jpg")
    ap.add_argument("--in_root", required=True, help="输入图像根目录，用于复刻子目录结构")
    ap.add_argument("--texture_dir", required=True, help="纹理图片目录")
    ap.add_argument("--out_root", required=True, help="输出根目录")
    ap.add_argument("--white_thresh", type=int, default=245, help="白色阈值，RGB都>=此值判定为背景")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（影响纹理选择与增广）")
    args = ap.parse_args()

    random.seed(args.seed)

    # 收集输入文件
    files = glob.glob(args.in_glob, recursive=True)
    files = [fp for fp in files if os.path.splitext(fp.lower())[1] in IMG_EXTS]
    if not files:
        raise SystemExit(f"[ERR] 没有匹配到输入图片：{args.in_glob}")

    # 预检查纹理
    _ = choose_random_texture(args.texture_dir)

    for fp in files:
        try:
            process_one(fp, args.in_root, args.out_root, args.texture_dir, args.white_thresh)
        except Exception as e:
            print(f"[WARN] 处理失败：{fp}  — {repr(e)}")


if __name__ == "__main__":
    main()
