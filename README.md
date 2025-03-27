
# CS-DJ: Distraction is All You Need for Multimodal Large Language Model Jailbreaking

<div align="center">
  <a href="https://arxiv.org/abs/2502.10794"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper"></a>
</div>


## Abstract

Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex internal interactions among visual elements and their alignment with text can introduce vulnerabilities, which may be exploited to bypass safety mechanisms. To address this, we analyze the relationship between image content and task and find that the complexity of subimages, rather than their content, is key. Building on this insight, we propose the **Distraction Hypothesis**, followed by a novel framework called Contrasting Subimage Distraction Jailbreaking (**CS-DJ**), to achieve jailbreaking by disrupting MLLMs alignment through multi-level distraction strategies. CS-DJ consists of two components: structured distraction, achieved through query decomposition that induces a distributional shift by fragmenting harmful prompts into sub-queries, and visual-enhanced distraction, realized by constructing contrasting subimages to disrupt the interactions among visual elements within the model. This dual strategy disperses the model’s attention, reducing its ability to detect and mitigate harmful content. 

![model_figure](./figs/CS-DJ.png)

## 🔥 News

<div class="scrollable">
    <ul>
      <li><strong>[2025, Feb 15]</strong>: &nbsp;🎉🎉  We've released <b>CS-DJ!</b> Check our <a href="https://arxiv.org/abs/2502.10794">paper</a> for more details.</li>
      <li><strong>[2025, Feb 28]</strong>: &nbsp;🎉🎉  Our work was accepted by CVPR 2025 !!!</li>
    </ul>
</div>
<span id='table-of-contents'/>

## Preparation

#### utilized Models 

- Embedding Model: [clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14)
- Auxiliary Model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Evaluation Model: [Beaver-Dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)

#### Image Library
Download the [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) dataset and store images in the `./llava_images` directory.


#### Installation Dependency
The deployed Python environment is version 3.10.

The required dependencies can be installed via pip:
```bash
pip install -r requirements.txt
```

## Usage


#### Set Up
Configure the retrieval random  `seed` (e.g., `2025`), `image count` (e.g., `10000`), `object model` (`gpt-4o`, `gpt-4o-mini`, and `gpt-4-vision-preview` are supported), and `api_key` in the `config.py` file.

#### GPT Series Jailbreaking Execution
Execute this command:

```bash
bash run.sh
```

#### Evaluation
Execute this command:

```python
python beavertails/our_beravertails_eval.py --strategy_name CS-DJ_best_method
```

#### Gemini Series Jailbreaking Execution
First, set Gemini API key in script/Gemini_infer.py
```python
genai.configure(api_key='<Your_Gemini_API_KEY>')
```

Then, you can execute this command:

```bash
python script/Gemini_infer.py
```

**Tip: The file structure of the result of Gemini inference may be different from that of the result of GPT series inference. If you need to evaluate the results of Gemini, you may need to modify the file reading part of beavertails/our_beravertails_eval.py appropriately.**

## Citation

If you find this code useful, please consider citing our paper:
 ```bibtex
@article{yang2025distraction,
  title={Distraction is All You Need for Multimodal Large Language Model Jailbreaking},
  author={Yang, Zuopeng and Fan, Jiluan and Yan, Anli and Gao, Erdun and Lin, Xin and Li, Tao and Dong, Changyu and others},
  journal={arXiv preprint arXiv:2502.10794},
  year={2025}
}
``` 

## Related Projects

- [Hades](https://github.com/RUCAIBox/HADES)
- [BeaverTails](https://github.com/PKU-Alignment/beavertails)
