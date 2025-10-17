# safety/llamaguard_vision.py
# Clean API for Llama-Guard-3-11B-Vision (OneVision) via HF Transformers

import json
from typing import List, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


DEFAULT_SYSTEM_PROMPT = (
    "You are Llama Guard. You will be given an image and a user request. "
    "Classify whether the request is safe according to the given safety policy. "
    "Respond strictly in JSON with keys: verdict (SAFE or UNSAFE), category, rationale."
)


def _ensure_images(
    images: Optional[List[Union[Image.Image, "np.ndarray"]]],
    image_paths: Optional[List[str]],
) -> List[Image.Image]:
    ims: List[Image.Image] = []
    if images:
        for im in images:
            ims.append(im if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB"))
    if image_paths:
        for p in image_paths:
            ims.append(Image.open(p).convert("RGB"))
    return ims


def _extract_json(text: str) -> Dict:
    """
    Try to extract a JSON object from free-form text.
    Expected keys: verdict, category, rationale
    """
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = text[s : e + 1]
        try:
            data = json.loads(raw)
            if "verdict" in data:
                data["verdict"] = str(data["verdict"]).upper()
            return {
                "verdict": data.get("verdict", "ParseError"),
                "category": data.get("category", "NA"),
                "rationale": data.get("rationale", text[:300]),
            }
        except Exception:
            pass
    return {"verdict": "ParseError", "category": "NA", "rationale": text[:300]}


class LlamaGuardVision:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-11B-Vision",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code
        )

        load_kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code)
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **load_kwargs)
        self.system_prompt = system_prompt

    def _format_single_user_turn(self, user_text: str) -> str:
        """
        Use a single 'user' turn that includes:
          - safety policy (system_prompt) + the actual request text
          - an <image> placeholder in the same turn
        This satisfies 'user/assistant' alternation in the chat template.
        """
        merged = f"{self.system_prompt}\n\nUSER REQUEST:\n{user_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": merged},
                    {"type": "image"},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return rendered

    @torch.inference_mode()
    def _generate(self, rendered_text: str, images: List[Image.Image], max_new_tokens: int = 256) -> str:
        inputs = self.processor(text=[rendered_text], images=images, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        out = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return out.strip()

    def check_request(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Image.Image]] = None,
        image_paths: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        # --- backward-compat alias ---
        user_text: Optional[str] = None,
    ) -> Dict:
        """
        Pre-response guard. Returns a dict with keys: verdict, category, rationale.
        Accepts either `prompt=` or legacy `user_text=`.
        """
        query = prompt if prompt is not None else (user_text or "")
        ims = _ensure_images(images, image_paths)
        rendered = self._format_single_user_turn(query)
        raw = self._generate(rendered, ims, max_new_tokens=max_new_tokens)
        return _extract_json(raw)

    def check_response(
        self,
        user_text: str,
        assistant_text: str,
        images: Optional[List[Image.Image]] = None,
        image_paths: Optional[List[str]] = None,
        max_new_tokens: int = 256,
    ) -> Dict:
        """
        Post-response guard: judge the assistant's reply in context.
        We concatenate: policy + USER + ASSISTANT, then render a single user turn with <image>.
        """
        ims = _ensure_images(images, image_paths)
        joined = f"{self.system_prompt}\n\nUSER REQUEST:\n{user_text}\n\nASSISTANT:\n{assistant_text}"
        rendered = self._format_single_user_turn(joined)
        raw = self._generate(rendered, ims, max_new_tokens=max_new_tokens)
        return _extract_json(raw)
