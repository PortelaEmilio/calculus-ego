"""
Backend VLM de CLASIFICACIÓN con Qwen3.5-9B (NF4 por defecto; ver VLM_QUANTIZATION) vía Transformers.

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
import threading as _threading
import time as _time
from collections import deque as _deque

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


class _Throughput:
    """Contador de velocidad de generación del VLM (tokens nuevos / s de `generate`).

    Mide SOLO el tiempo dentro de `model.generate` — no YOLO, ni el preprocesado de
    imagen, ni la E/S — que es lo que dice si el modelo va a la velocidad esperada.
    El cliente va en serie, pero se protege con un lock por si algún path llama en
    paralelo. `reciente` es una media móvil de las últimas llamadas, para ver caídas
    (p.ej. una imagen enorme) sin que las diluya el acumulado del run.
    """

    VENTANA = 20

    def __init__(self):
        self._lock = _threading.Lock()
        self.tokens = 0
        self.segundos = 0.0
        self.llamadas = 0
        self._ultimas = _deque(maxlen=self.VENTANA)

    def registrar(self, n_tokens, segundos):
        if n_tokens <= 0 or segundos <= 0:
            return
        with self._lock:
            self.tokens += n_tokens
            self.segundos += segundos
            self.llamadas += 1
            self._ultimas.append((n_tokens, segundos))

    def stats(self):
        with self._lock:
            glob = self.tokens / self.segundos if self.segundos else 0.0
            tk = sum(t for t, _ in self._ultimas)
            sg = sum(s for _, s in self._ultimas)
            return {"tokens": self.tokens, "segundos": self.segundos,
                    "llamadas": self.llamadas, "tok_s": glob,
                    "tok_s_reciente": (tk / sg if sg else 0.0)}

    def resumen(self):
        """Línea corta para la barra de progreso: '12.4 tok/s (últ. 11.8) · 1.2k llam.'"""
        st = self.stats()
        if not st["llamadas"]:
            return "— tok/s"
        n = st["llamadas"]
        n_txt = f"{n/1000:.1f}k" if n >= 1000 else str(n)
        return (f"{st['tok_s']:.1f} tok/s (últ. {st['tok_s_reciente']:.1f}) · {n_txt} llam.")


THROUGHPUT = _Throughput()


class _OomCounter:
    """Cuenta los OOM de CUDA que NO se pudieron recuperar.

    Sirve para que el orquestador sepa que las clasificaciones de un fichero
    salieron degradadas y NO escriba su `summary_<stem>.json`: si lo escribiera,
    la reanudación lo daría por hecho y el fichero quedaría con todo a
    'no visible' PARA SIEMPRE (pérdida silenciosa, el mismo modo de fallo que el
    phase1 huérfano).
    """

    def __init__(self):
        self._lock = _threading.Lock()
        self.n = 0
        self.recuperados = 0

    def fallo(self):
        with self._lock:
            self.n += 1

    def recuperado(self):
        with self._lock:
            self.recuperados += 1

    def total(self):
        with self._lock:
            return self.n


OOM = _OomCounter()


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
            # Cuantización según config.VLM_QUANTIZATION: "nf4" (por defecto, la de producción
            # en la 4070 de 12 GB), "int8", o "none" = bf16 sin bitsandbytes (GPUs grandes).
            quant_mode = str(getattr(config, "VLM_QUANTIZATION", "nf4")).lower()
            if quant_mode == "nf4":
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                )
            elif quant_mode == "int8":
                quant = BitsAndBytesConfig(load_in_8bit=True)
            elif quant_mode in ("none", "bf16"):
                quant = None
            else:
                raise ValueError(f"VLM_QUANTIZATION no soportado: {quant_mode!r} "
                                 "(nf4 | int8 | none)")
            # GPU 0 entera por defecto; CALCULUS_EGO_DEVICE_MAP admite "auto", "cuda",
            # "cpu" o un índice de GPU.
            dm = os.environ.get("CALCULUS_EGO_DEVICE_MAP", "").strip()
            device_map = ({"": int(dm)} if dm.isdigit() else dm) if dm else {"": 0}
            info(f"  Backend VLM clasificación: [dim]Qwen3.5-9B {base} · attn=eager · {quant_mode}")
            # AutoModelForImageTextToText → Qwen3_5ForConditionalGeneration (con vision tower).
            # attn eager: requerido por la atención lineal GatedDeltaNet de qwen3_5.
            self.model = AutoModelForImageTextToText.from_pretrained(
                base, quantization_config=quant, device_map=device_map,
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

        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      temperature=None, top_p=None, top_k=None,
                      pad_token_id=self.processor.tokenizer.eos_token_id)
        _t0 = _time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kw)
        except torch.cuda.OutOfMemoryError:
            # La causa habitual NO es falta de memoria real sino fragmentación (o un
            # proceso vecino que la libera enseguida). Vaciar la caché del allocator y
            # repetir la MISMA llamada recupera la mayoría, sin tocar la resolución
            # (bajarla cambiaría el resultado y el modelo dejaría de ser reproducible).
            torch.cuda.empty_cache()
            try:
                with torch.inference_mode():
                    out = self.model.generate(**inputs, **gen_kw)
                OOM.recuperado()
            except torch.cuda.OutOfMemoryError:
                OOM.fallo()
                torch.cuda.empty_cache()
                raise
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # generate es async en CUDA: sin esto el tiempo sale falso
        THROUGHPUT.registrar(int(out.shape[1]) - in_len, _time.perf_counter() - _t0)
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
