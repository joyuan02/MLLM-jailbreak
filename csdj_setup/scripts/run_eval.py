# 作用：调用你仓库里的 beavertails/our_beravertails_eval.py 做评测
# ---------------------------------------------------------------
import subprocess, os, sys
import config

cmd = [
    sys.executable, "beavertails/our_beravertails_eval.py",
    "--attack_result_dir", config.RESULTS_DIR,
    "--strategy_name", config.STRATEGY_NAME,
    "--eval_output_dir", config.EVAL_OUTPUT_DIR,
]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
