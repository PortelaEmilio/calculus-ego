"""
Backend VLM de CLASIFICACIÓN con Qwen3.5-9B (4-bit NF4) vía Transformers.

Drop-in de `VLMBackend`: implementa `is_loaded()`, `load()` (no-op), `generate()` y
hereda `generate_batch()` (bucle secuencial) + `supports_real_batch = False`. Lo consumen
los clasificadores (person_attributes / scene_context / social_distance) igual que a
`TransformersBackend`, envuelto por `JsonFlatteningBackend` cuando el prompt es
`prompts_gemma4_json` (JSON→líneas).

**PRODUCCIÓN desde 2026-07-02**: es el clasificador general de producción (`config.BEHAVIOUR_MODEL_NAME
= "Qwen/Qwen3.5-9B"` + `VLM_PROMPT_MODULE="prompts_qwen3"`). En el banco sintético FLUX EGC da κ medio
0.924 (8 categorías) — mejor que gemma-4 en actividad/sports. ⚠️ Validado SOLO en el banco sintético; el
κ sobre el sample 500 REAL está PENDIENTE (ver CLAUDE.md "PRODUCCIÓN ACTUAL"). Nació para el benchmark
gemma-4 vs Qwen3.5 (`benchmark_gemma4_vs_qwen35.py`). La carga es el espejo exacto de
`models/beauty_backend_qwen35.py` (Qwen3.5-9B es multimodal → `AutoModelForImageTextToText`;
`attn_implementation="eager"` es REQUERIDO por la atención lineal GatedDeltaNet de qwen3_5;
`trust_remote_code=True`; 4-bit NF4) PERO sin adapter LoRA de belleza. **Requiere el wrapper gcc-15 en
PATH** (kernels TileLang de qwen3_5) — en producción lo mete `venv/bin/activate`
(`$HOME/.local/gcc15_wrapper`); CUDA rechaza el gcc-16 de Fedora 44.

Diferencias con el backend de belleza (que emite un dígito):
  - Qwen3.5 es un modelo "thinking": se desactiva el CoT con `enable_thinking=False`
    (fallback: se elimina `<think>…</think>` del output) para no inflar tokens ni romper el JSON.
  - `max_new_tokens` por defecto = `config.VLM_MAX_TOKENS` (~512, salida JSON).

⚠️ La calidad de Qwen3.5-NF4 como clasificador está medida en el banco sintético (κ 0.924), NO en el
sample 500 real (pendiente). Rollback a gemma-4: restaurar `config.before_qwen35_prod_*.py` o invertir
`BEHAVIOUR_MODEL_NAME` + `VLM_PROMPT_MODULE` en `config.py`.
"""
import os
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image

import config
from ui import info, warn
from .backends.base import VLMBackend

# Lado máximo de la imagen enviada al modelo (thumbnail, solo reduce). Acota los
# tokens visuales de Qwen (VRAM/tiempo en la RTX 4070 de 12 GB con el 9B 4-bit).
# Ajustable si el dry-run OOMea; 896 ≈ el downsize del pipeline para gemma-4.
QWEN35_IMG_MAX = int(os.environ.get("QWEN35_IMG_MAX", "896"))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Qwen35VLMBackend(VLMBackend):
    supports_real_batch = False

    def __init__(self, model_name=None):
        self._loaded = False
        self.torch = None
        # Base COMPARTIDO clasificador↔belleza: si está activo y el backend de belleza es
        # "qwen35_cont" (mismo base Qwen3.5-9B), cargamos el LoRA de belleza sobre esta base
        # y servimos AMBAS tareas con una sola instancia (clasificación con disable_adapter(),
        # belleza con el adapter activo). Verificado byte a byte == base puro. Ver CLAUDE.md.
        self._has_beauty_adapter = False
        base = model_name or getattr(config, "BEAUTY_QWEN35_BASE", "Qwen/Qwen3.5-9B")
        self.model_name = base
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig)

            self.torch = torch
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            info(f"  Backend VLM clasificación: [dim]Qwen3.5-9B {base} · attn=eager · nf4")
            # AutoModelForImageTextToText → Qwen3_5ForConditionalGeneration (con vision tower).
            # attn eager: requerido por la atención lineal GatedDeltaNet de qwen3_5.
            self.model = AutoModelForImageTextToText.from_pretrained(
                base, quantization_config=quant, device_map={"": 0},
                dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation="eager",
            )
            self._maybe_attach_beauty_adapter(base)
            self.model.eval()
            self.model.config.use_cache = True
            self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
            self._loaded = True
        except Exception as e:
            warn(f"  No se pudo cargar el backend Qwen3.5 de clasificación: {e}")
            self._loaded = False

    def _maybe_attach_beauty_adapter(self, base):
        """Envuelve self.model con el LoRA de belleza (PeftModel) para servir ambas tareas
        desde una sola instancia. Solo si el sharing está activo, el backend de belleza es
        el mismo base ("qwen35_cont") y el adapter existe en disco. Falla-suave: si no se
        puede adjuntar, se sigue como clasificador puro (sin belleza compartida)."""
        import os as _os
        if not getattr(config, "BEAUTY_SHARE_CLASSIFIER_BASE", False):
            return
        if getattr(config, "BEAUTY_BACKEND", "") != "qwen35_cont":
            return
        adapter = getattr(config, "BEAUTY_QWEN35_ADAPTER_PATH", "")
        if not (adapter and _os.path.isdir(adapter)):
            warn(f"  Base compartido activo pero adapter de belleza no encontrado: {adapter}")
            return
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self._has_beauty_adapter = True
            info(f"  Base COMPARTIDO: LoRA de belleza adjunto [dim]{adapter}[/] "
                 "(clasif. con adapter OFF · belleza con adapter ON)")
        except Exception as e:
            warn(f"  No se pudo adjuntar el LoRA de belleza (sigo como clasificador puro): {e}")
            self._has_beauty_adapter = False

    def load(self):
        """No-op: el modelo se carga en __init__ (paridad con la factory)."""
        return self._loaded

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, pil_image, prompt, max_new_tokens=None, task_hint=None, **kwargs) -> str:
        """CLASIFICACIÓN. Si hay LoRA de belleza adjunto (base compartido), se genera con el
        adapter DESACTIVADO (`disable_adapter()`) → salida idéntica al base puro."""
        if max_new_tokens is None:
            max_new_tokens = getattr(config, "VLM_MAX_TOKENS", 512)
        if self._has_beauty_adapter:
            with self.model.disable_adapter():
                return self._run(pil_image, prompt, QWEN35_IMG_MAX,
                                 max_new_tokens, think_off=True)
        return self._run(pil_image, prompt, QWEN35_IMG_MAX,
                         max_new_tokens, think_off=True)

    def generate_beauty(self, pil_image, prompt, max_new_tokens=None, **kwargs) -> str:
        """BELLEZA (base compartido). Genera con el LoRA de belleza ACTIVO y el mismo
        preprocesado que el backend dedicado (thumbnail 672, sin thinking). Devuelve el
        texto crudo (un número); el BeautyEstimator lo parsea."""
        if not self._has_beauty_adapter:
            raise RuntimeError("generate_beauty llamado sin LoRA de belleza adjunto")
        if max_new_tokens is None:
            max_new_tokens = getattr(config, "BEAUTY_MAX_NEW_TOKENS", 8)
        return self._run(pil_image, prompt, 672, max_new_tokens, think_off=False)

    def _run(self, pil_image, prompt, img_max, max_new_tokens, think_off) -> str:
        torch = self.torch
        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image)
        if not isinstance(pil_image, Image.Image):
            raise TypeError(f"imagen no soportada: {type(pil_image)}")
        image = pil_image.convert("RGB")
        image.thumbnail((img_max, img_max), Image.LANCZOS)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        # Qwen3.5 hace CoT por defecto → enable_thinking=False para clasificación. Si el
        # template de esta versión no acepta el kwarg, se cae al modo normal y se limpia el
        # <think>…</think> del output (fallback en _strip_think). Para belleza (max_new≈8)
        # no se toca el thinking (el número sale directo).
        template_kw = dict(tokenize=True, add_generation_prompt=True,
                           return_dict=True, return_tensors="pt")
        if think_off:
            try:
                inputs = self.processor.apply_chat_template(
                    messages, enable_thinking=False, **template_kw)
            except (TypeError, ValueError):
                inputs = self.processor.apply_chat_template(messages, **template_kw)
        else:
            inputs = self.processor.apply_chat_template(messages, **template_kw)
        inputs = inputs.to(self.model.device)
        in_len = inputs["input_ids"].shape[1]

        seed = getattr(config, "VLM_SEED", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        text = self.processor.tokenizer.decode(
            out[0][in_len:], skip_special_tokens=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._strip_think(text).strip() if think_off else text.strip()

    @staticmethod
    def _strip_think(text: str) -> str:
        """Elimina bloques <think>…</think> (fallback si enable_thinking no aplicó)."""
        if "<think>" in text:
            text = _THINK_RE.sub("", text)
            # <think> sin cierre (truncado): quedarse con lo posterior si existe.
            if "<think>" in text:
                text = text.split("</think>")[-1] if "</think>" in text else \
                       text.split("<think>")[0]
        return text
