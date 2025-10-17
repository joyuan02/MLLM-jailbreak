
"""
Quick checks:
- .env loaded (if present), API key presence
- Image folder exists and has images
"""
import os, sys, glob, random, json

def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k,v = line.split("=",1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

import config
try:
    config.assert_config()
except Exception as e:
    print("CONFIG ERROR:", e)
    sys.exit(1)

img_dir = config.IMAGE_DIR
paths = []
for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
    paths.extend(glob.glob(os.path.join(img_dir, "**", ext), recursive=True))

print(json.dumps({
    "object_model": config.OBJECT_MODEL,
    "has_api_key": bool(config.OPENAI_API_KEY),
    "image_dir": os.path.abspath(img_dir),
    "num_images_found": len(paths),
    "sample_images": sorted(random.sample(paths, min(3, len(paths)))) if paths else []
}, indent=2, ensure_ascii=False))
