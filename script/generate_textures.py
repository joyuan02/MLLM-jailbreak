# script/generate_textures.py
import os, argparse, math, random
import numpy as np
from PIL import Image, ImageFilter

# --------------------------
# Utils
# --------------------------
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def to_uint8(img, clip=True):
    if clip:
        img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def save_tex(arr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(to_uint8(arr), mode="L").save(out_path, quality=95)

# --------------------------
# Perlin / fBM
# --------------------------
def perlin(width, height, scale=8.0, seed=None):
    """
    简洁 Perlin：基于栅格梯度插值，适合做纸张/云雾基础
    """
    rng = np.random.default_rng(seed)
    gx = int(max(1, math.ceil(width / scale))) + 1
    gy = int(max(1, math.ceil(height / scale))) + 1
    # 随机栅格梯度方向（单位向量）
    grads = rng.normal(size=(gy, gx, 2))
    norms = np.linalg.norm(grads, axis=2, keepdims=True) + 1e-8
    grads = grads / norms

    # 网格坐标
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = xs / scale
    y = ys / scale

    # 左上角整数格坐标
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # 相对向量
    dx0 = x - x0; dy0 = y - y0
    dx1 = x - x1; dy1 = y - y1

    # 栅格梯度点积
    def g(ix, iy, dx, dy):
        gxy = grads[np.clip(iy, 0, gy-1), np.clip(ix, 0, gx-1)]
        return gxy[...,0]*dx + gxy[...,1]*dy

    n00 = g(x0, y0, dx0, dy0)
    n10 = g(x1, y0, dx1, dy0)
    n01 = g(x0, y1, dx0, dy1)
    n11 = g(x1, y1, dx1, dy1)

    # 平滑插值（fade）
    def fade(t): return t*t*t*(t*(t*6 - 15) + 10)
    u = fade(dx0); v = fade(dy0)

    nx0 = n00*(1-u) + n10*u
    nx1 = n01*(1-u) + n11*u
    nxy = nx0*(1-v) + nx1*v

    # 标准化到 [0,1]
    nxy = (nxy - nxy.min()) / (nxy.max() - nxy.min() + 1e-8)
    return nxy

def fbm(width, height, base_scale=12.0, octaves=5, persistence=0.5, lacunarity=2.0, seed=None):
    """
    fractal Brownian motion：多频率 Perlin 融合，更自然的纸张/云雾
    """
    val = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for o in range(octaves):
        n = perlin(width, height, scale=base_scale/freq, seed=(None if seed is None else seed+o*97))
        val += amp * n
        amp *= persistence
        freq *= lacunarity
    val -= val.min()
    val /= (val.max() + 1e-8)
    return val

# --------------------------
# Worley (Cellular) Noise
# --------------------------
def worley(width, height, num_points=64, seed=None, metric='euclidean'):
    """
    Worley/Cellular：像鹅卵石、皮革孔隙
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform([0,0],[width, height], size=(num_points,2))
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = xs[...,None]; y = ys[...,None]
    dx = x - pts[:,0]; dy = y - pts[:,1]
    if metric == 'manhattan':
        d = np.abs(dx) + np.abs(dy)
    else:
        d = np.sqrt(dx*dx + dy*dy)
    # 取最近点距离
    dmin = d.min(axis=-1)
    # 归一化
    dmin -= dmin.min()
    dmin /= (dmin.max() + 1e-8)
    return dmin

# --------------------------
# Marble / Stripes
# --------------------------
def marble(width, height, freq=0.015, turb_power=2.5, fbm_scale=24.0, seed=None):
    """
    Marble：sin(基底 + 扰动)，配合 fBM 制造流纹
    """
    t = fbm(width, height, base_scale=fbm_scale, octaves=5, seed=seed)
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    base = xs * freq + t * turb_power
    m = 0.5 * (1 + np.sin(2*np.pi*base))
    return m

def stripes(width, height, angle_deg=30, freq=0.02, fbm_scale=16.0, seed=None):
    """
    带噪声的斜条纹（布纹/纸纹）
    """
    ang = math.radians(angle_deg)
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    proj = xs*math.cos(ang) + ys*math.sin(ang)
    t = fbm(width, height, base_scale=fbm_scale, octaves=4, seed=seed)
    s = 0.5*(1 + np.sin(2*np.pi*(proj*freq + t*0.5)))
    return s

# --------------------------
# Weave（简单织布）
# --------------------------
def weave(width, height, freq=0.02, contrast=0.7, seed=None):
    """
    简化织布：两组正交条纹叠加
    """
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    wx = 0.5*(1+np.sin(2*np.pi*xs*freq))
    wy = 0.5*(1+np.sin(2*np.pi*ys*freq))
    w = (wx*wy)
    # 加点 fBM 粗糙度
    t = fbm(width, height, base_scale=24.0, octaves=3, seed=seed)
    out = (1-contrast)*w + contrast*t
    out -= out.min()
    out /= (out.max()+1e-8)
    return out

# --------------------------
# Speckle / Paper grain
# --------------------------
def speckle(width, height, density=0.01, blur=1.0, seed=None):
    """
    随机颗粒+轻模糊，做成粗糙纸
    """
    rng = np.random.default_rng(seed)
    img = rng.random((height, width))
    mask = (img < density).astype(np.float32)
    base = fbm(width, height, base_scale=18.0, octaves=4, seed=(None if seed is None else seed+123))
    out = 0.7*base + 0.3*mask
    out -= out.min()
    out /= (out.max()+1e-8)
    pil = Image.fromarray(to_uint8(out), 'L')
    if blur > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.array(pil, dtype=np.float32)/255.0
    return arr

# --------------------------
# Checker
# --------------------------
def checker(width, height, cell=32):
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    cx = (xs // cell) % 2
    cy = (ys // cell) % 2
    out = ((cx ^ cy).astype(np.float32))
    return out

# --------------------------
# Dispatcher
# --------------------------
KIND_FUNCS = {
    "perlin": lambda w,h,s,seed,**kw: perlin(w,h,scale=s, seed=seed),
    "fbm":    lambda w,h,s,seed,octaves=5,**kw: fbm(w,h,base_scale=s, octaves=octaves, seed=seed),
    "worley": lambda w,h,s,seed,num_points=64,metric='euclidean',**kw: worley(w,h,num_points=num_points, seed=seed, metric=metric),
    "marble": lambda w,h,s,seed,**kw: marble(w,h, fbm_scale=max(8.0, s), seed=seed),
    "stripes":lambda w,h,s,seed,angle=30,**kw: stripes(w,h, angle_deg=angle, fbm_scale=max(8.0,s), seed=seed),
    "weave":  lambda w,h,s,seed,**kw: weave(w,h, freq=1.0/max(8.0,s), seed=seed),
    "speckle":lambda w,h,s,seed,**kw: speckle(w,h, density=kw.get('density',0.01), blur=kw.get('blur',1.0), seed=seed),
    "checker":lambda w,h,s,seed,**kw: checker(w,h, cell=int(max(4, s))),
}

def generate_one(kind, W, H, scale, seed, **kwargs):
    arr = KIND_FUNCS[kind](W, H, scale, seed, **kwargs)
    # 少量对比度拉伸/偏移，避免太灰
    arr = np.power(arr, kwargs.get('gamma', 1.0))
    lo, hi = kwargs.get('low', 0.0), kwargs.get('high', 1.0)
    arr = lo + (hi - lo) * arr
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="输出纹理文件夹")
    ap.add_argument("--num", type=int, default=50, help="生成数量")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--kinds", type=str, default="fbm,worley,marble,weave,speckle",
                    help="逗号分隔：perlin,fbm,worley,marble,stripes,weave,speckle,checker")
    ap.add_argument("--scale_min", type=float, default=8.0)
    ap.add_argument("--scale_max", type=float, default=48.0)
    ap.add_argument("--seed", type=int, default=None)
    # 风格细节（可选）
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--low", type=float, default=0.0)
    ap.add_argument("--high", type=float, default=1.0)
    ap.add_argument("--worley_points", type=int, default=64)
    ap.add_argument("--stripes_angle", type=float, default=30.0)
    ap.add_argument("--speckle_density", type=float, default=0.01)
    ap.add_argument("--speckle_blur", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip() in KIND_FUNCS]
    if not kinds:
        raise SystemExit("[ERR] --kinds 为空或非法")

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num):
        kind = random.choice(kinds)
        scale = random.uniform(args.scale_min, args.scale_max)
        seed = None if args.seed is None else args.seed + i*31

        kwargs = dict(
            gamma=args.gamma, low=args.low, high=args.high,
            num_points=args.worley_points,
            angle=args.stripes_angle,
            density=args.speckle_density,
            blur=args.speckle_blur,
        )
        tex = generate_one(kind, args.width, args.height, scale, seed, **kwargs)
        out = os.path.join(args.out_dir, f"{i:04d}_{kind}.jpg")
        save_tex(tex, out)
        print(f"[OK] {out}  kind={kind} scale={scale:.2f}")
    print(f"[DONE] Generated {args.num} textures -> {args.out_dir}")

if __name__ == "__main__":
    main()
