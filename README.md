## Code
.

├─ script/                      # 所有可执行脚本（仅列出与本 README 相关的）

│  ├─ main.py                   # 生成合成图（classic/texture）；写出 image_map 与结果

│  ├─ eval_with_llamaguard_from_images.py   # 用“已生成合成图+LlamaGuard-3-Vision”做预/后审评测

│

├─ safety/                      # 安全模型适配

│  ├─ llamaguard_vision.py      # Meta Llama-Guard-3-11B-Vision 推理包装（HF Transformers）

│

├─ config/                      # 统一的 argparse 参数解析等（CS_DJ_parser）

│

├─ instructions/                # 各类越狱指令 JSON（按类目分：Animal/Financial/Privacy/...）

│

├─ distraction_images/          # 合成图输出根目录

│  ├─ CS-DJ_best_method/        # 基础实验合成图（single 每条一图）

│  ├─ CS-DJ_best_method_textured/ # Texture 纹理版合成图

│  └─ CS-DJ_best_method_subq3/  # 三子图版本（每条三图）

│

├─ image_embedding/             # 图像嵌入缓存

├─ image_map/                   # 记录使用的图索引与采样映射（保证复现实验）

│

├─ results/                     # 主模型+安全前后审的回答结果（每类一个 JSON）

│  └─ <strategy>/<object_model>/<category>.json

│

├─ eval_results/                # BeaverTails 离线评估输出（布尔或分数）

│  └─ <strategy>/<object_model>/<category>.json

│

├─ beavertails/                 # 本地 BeaverTails 代码（供离线评估）

│  ├─ moderations.py

│  ├─ constants.py  

│  └─ utils.py

│

└─ README.md
