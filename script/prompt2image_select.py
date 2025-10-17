# -*- coding: utf-8 -*-
"""
Given an input image, find the farthest (least similar) images in the dataset
based on precomputed CLIP image embeddings (image_embeding/map_seed_*.json).

Usage (PowerShell):
  python .\script\image2image_select.py `
    --query_image .\some\query.jpg `
    --emb_file .\image_embeding\map_seed_0_num_500.json `
    --topk 9 `
    --diversify `
    --out_json .\combine_image_data\image_query_farthest.json

Notes:
- emb_file JSON items look like: {"img_path": "...", "img_emb": [...]}
- Model: clip-ViT-L-14 (same as你的embedding脚本)
"""

import os
import json
import argparse
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def load_emb_file(emb_file: str):
    with open(emb_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_paths = []
    img_embs = []
    for it in data:
        p = it["img_path"]
        e = it["img_emb"]
        img_paths.append(p)
        img_embs.append(e)
    img_embs = torch.tensor(img_embs, dtype=torch.float32)
    return img_paths, img_embs


def encode_image(model: SentenceTransformer, image_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        emb = model.encode(img, convert_to_tensor=True, device=device, normalize_embeddings=False)
    return emb


def simple_farthest(query_emb: torch.Tensor, img_embs: torch.Tensor, k: int) -> List[int]:
    """
    Return indices of k farthest images to query_emb (cosine similarity smallest).
    """
    # [1, D] x [N, D] -> [N]
    sims = util.cos_sim(query_emb, img_embs)[0]  # higher = closer
    vals, idx = torch.sort(sims, descending=False)  # ascending -> smallest first
    k = min(k, img_embs.size(0))
    return idx[:k].tolist(), vals[:k].tolist()


def farthest_first_diversified(query_emb: torch.Tensor, img_embs: torch.Tensor, k: int) -> List[int]:
    """
    Greedy diversified selection: start from query_emb, each step pick the image
    with the smallest mean similarity to the already selected set.
    """
    device = query_emb.device
    selected_vecs = [query_emb]  # seeds
    chosen = []
    sims_cache = None

    for _ in range(min(k, img_embs.size(0))):
        # mean cos sim to selected set (越小越“不相似”)
        M = torch.vstack(selected_vecs)               # [m, D]
        sims = util.cos_sim(M, img_embs).mean(dim=0) # [N]
        if chosen:
            sims[torch.tensor(chosen, device=device)] = 1e9  # mask already chosen
        i = int(torch.argmin(sims).item())
        chosen.append(i)
        selected_vecs.append(img_embs[i])
    # 也返回与当前“已选集合”的均值相似度（最后一次）用于展示
    final_sims = util.cos_sim(torch.vstack(selected_vecs), img_embs).mean(dim=0)
    return chosen, final_sims[chosen].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_image", required=True, help="要检索的查询图片路径")
    ap.add_argument("--emb_file", required=True, help="预计算的图片embedding文件（JSON）")
    ap.add_argument("--topk", type=int, default=9, help="返回最不相似Top-K")
    ap.add_argument("--diversify", action="store_true",
                    help="开启贪心“最远优先”多样化（默认关闭，单点Top-K）")
    ap.add_argument("--out_json", default="./combine_image_data/image_query_farthest.json",
                    help="把选出的结果写入的JSON路径")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) 加载预计算的dataset图片向量
    print(f"[Load] {args.emb_file}")
    img_paths, img_embs = load_emb_file(args.emb_file)
    img_embs = img_embs.to(device)

    # 2) 加载CLIP并编码查询图片
    print("[Model] SentenceTransformer('clip-ViT-L-14')")
    model = SentenceTransformer("clip-ViT-L-14").to(device)
    query_emb = encode_image(model, args.query_image, device)

    # 3) 选择最不相似的Top-K
    if args.diversify:
        idxs, scores = farthest_first_diversified(query_emb, img_embs, args.topk)
        mode = "diversified"
    else:
        idxs, scores = simple_farthest(query_emb, img_embs, args.topk)
        mode = "simple"

    # 4) 打印并保存
    print("\n[RESULT] Farthest (cosine similarity smallest) — mode:", mode)
    out_items = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        p = img_paths[i]
        print(f"#{rank:02d}  sim={float(s):.6f}  path={p}")
        out_items.append({"rank": rank, "path": p, "cos_sim": float(s)})

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "query_image": args.query_image,
            "emb_file": args.emb_file,
            "mode": mode,
            "topk": args.topk,
            "results": out_items
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote selection to: {args.out_json}")


if __name__ == "__main__":
    main()
