"""
Backend de belleza CONTINUO: Qwen3.5-9B (4-bit) + adapter LoRA (checkpoint-1400) vía
Transformers/PEFT. Emite belleza 1-10 CONTINUA (un decimal, p.ej. "7.3").

Drop-in de `BeautyEstimator(backend=...)`: misma interfaz mínima `is_loaded()` + `generate()`.
Se usa SOLO en el pase separado/diferido de belleza (beauty_phase.py / beauty_pass.py), nunca
junto al backend de los 15 clasificadores (no caben dos modelos en 12 GB).

Por qué Transformers/PEFT y NO Ollama: el modelo continuo iguala al de deciles en Pearson
(SCUT 0.934 / CFD 0.806 / MEBeauty 0.807 / HotOrNot 0.606 / M2B 0.524, checkpoint-1400 epoch 2.04)
y da MEJOR MAE + salida decimal. Servirlo por Ollama (GGUF + mmproj de llama.cpp) degradaría el
Pearson ~0.93→0.85 (el gap es del path de visión de llama.cpp, no de la cuantización). En 4-bit cabe
en la RTX 4070 y corre como pase aparte (VRAM libre: gemma4 ya está parado). Preprocesado de imagen =
el MISMO que en entrenamiento/evaluación (thumbnail 672, attn eager). Ver config.BEAUTY_QWEN35_* y CLAUDE.md.
"""
import os
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image

import config
from ui import info, warn


def _base_model_from_adapter(adapter_path: str, fallback: str) -> str:
    """Lee base_model_name_or_path del adapter_config.json (como evaluate_combined.py)."""
    cfg = Path(adapter_path) / "adapter_config.json"
    if cfg.exists():
        try:
            name = json.loads(cfg.read_text()).get("base_model_name_or_path")
            if name:
                return name
        except Exception:
            pass
    return fallback


class Qwen35BeautyBackend:
    def __init__(self, model_name=None, adapter_path=None, load_4bit=True):
        self._loaded = False
        self.torch = None
        adapter_path = adapter_path or getattr(
            config, "BEAUTY_QWEN35_ADAPTER_PATH", "")
        fallback_base = model_name or getattr(
            config, "BEAUTY_QWEN35_BASE", "Qwen/Qwen3.5-9B")
        base = _base_model_from_adapter(adapter_path, fallback_base)

        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig)
            from peft import PeftModel

            self.torch = torch
            quant = None
            if load_4bit:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                )
            info(f"  Beauty backend: [dim]Qwen3.5-9B {base} + LoRA {adapter_path}")
            # AutoModelForImageTextToText → Qwen3_5ForConditionalGeneration (con vision tower).
            # attn eager: requerido por la atención lineal GatedDeltaNet de qwen3_5 en inferencia.
            self.model = AutoModelForImageTextToText.from_pretrained(
                base, quantization_config=quant, device_map={"": 0},
                dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation="eager",
            )
            if adapter_path and os.path.isdir(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            else:
                warn(f"  Adapter de belleza no encontrado: {adapter_path} (usando modelo base)")
            self.model.eval()
            self.model.config.use_cache = True
            # Processor del BASE (no del adapter dir).
            self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
            self._loaded = True
        except Exception as e:
            warn(f"  No se pudo cargar el backend Qwen3.5 de belleza: {e}")
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, image, prompt, max_new_tokens=None, **kwargs) -> str:
        """image: PIL.Image RGB (o np.ndarray BGR→ya convertido por el estimador). Devuelve el texto."""
        torch = self.torch
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"imagen no soportada: {type(image)}")
        image = image.convert("RGB")
        # Mismo preprocesado que entrenamiento/eval: thumbnail 672 (no amplía caras pequeñas).
        image.thumbnail((672, 672), Image.LANCZOS)
        if max_new_tokens is None:
            max_new_tokens = getattr(config, "BEAUTY_MAX_NEW_TOKENS", 8)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self.processor.tokenizer.decode(
            out[0][in_len:], skip_special_tokens=True).strip()


class SharedQwen35BeautyBackend:
    """Backend de belleza que REUTILIZA el `Qwen35VLMBackend` de clasificación ya cargado
    (base COMPARTIDO): en vez de cargar una 2ª copia del Qwen3.5-9B, delega en su
    `generate_beauty()` (adapter de belleza activo). Drop-in de `BeautyEstimator(backend=...)`:
    expone `is_loaded()` + `generate(image, prompt, max_new_tokens)`. Lo construye el pase
    diferido cuando el clasificador trae el LoRA de belleza adjunto (BEAUTY_SHARE_CLASSIFIER_BASE).
    """

    def __init__(self, vlm_backend):
        self._vlm = vlm_backend

    def is_loaded(self) -> bool:
        return (self._vlm is not None and self._vlm.is_loaded()
                and getattr(self._vlm, "_has_beauty_adapter", False))

    def generate(self, image, prompt, max_new_tokens=None, **kwargs) -> str:
        return self._vlm.generate_beauty(image, prompt, max_new_tokens=max_new_tokens)
