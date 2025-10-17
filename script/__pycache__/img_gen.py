# img_gen.py
import time, torch
from tqdm import tqdm
from diffusers import DDIMScheduler
from diffuser_pipeline import StableDiffusionXLPipelineNegCFG  # 或 StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline 

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 你的 diffusers 版本需要 torch_dtype（dtype 会被忽略）
dtype = torch.float32

# 用你自己的子类；如果没有就用 StableDiffusionXLPipeline
pipe = StableDiffusionXLPipelineNegCFG.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=dtype,
).to(device)

# 更温和的采样器（负CFG更稳）
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 内存/稳定性小帮手（有就开）
if hasattr(pipe, "enable_vae_slicing"): pipe.enable_vae_slicing()
if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()

# 关闭内置进度条，用我们自己的 tqdm
pipe.set_progress_bar_config()

steps = 60

gen = torch.Generator(device=device).manual_seed(1234)

out = pipe(
    prompt="an astronaut playing violin on the moon",
    negative_prompt="",
    guidance_scale=4,
    num_inference_steps=steps,
    height=512, width=512,
    output_type="pil",
    num_images_per_prompt=6,
    generator=gen,                     
)

for i, im in enumerate(out.images):
    im.save(f"img_{i+1}.png")
    print(f"Saved img_{i+1}.png")



# prompt -> image -> dataset sample                       
#                 -> stable diffusion negative guidance -> CLIP to text embedding -> cos_similarity