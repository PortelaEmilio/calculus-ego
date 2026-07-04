"""
Proxy que aplana la salida JSON de `prompts_gemma4_json` a líneas `Label: value`
que los parsers line-based de producción (person_attributes / scene_context)
consumen sin cambios.

`prompts_gemma4_json` emite p.ej. {"gender": {"class": "male"}, ...}, pero los
parsers buscan `^Gender: ...` (la comilla del JSON rompe el regex). Este wrapper
envuelve el backend real: tras generar, si la salida es el objeto JSON de
person_attrs/scene, la convierte a líneas; cualquier otra salida (p.ej. el JSON
parts-visible de social_distance, que tiene su propio parser) pasa sin tocar.

Se activa en `loader.load_all_models()` cuando
`config.VLM_PROMPT_MODULE == "prompts_gemma4_json"`.
"""

import json
import re

# json key → etiqueta de línea esperada por los parsers de producción.
_KEY2LABEL = {
    "gender": "Gender", "age": "Age", "behaviour": "Behaviour",
    "bodydisplay": "BodyDisplay", "bodyweight": "BodyWeight",
    "muscle": "Musculature",  # silueta ELIMINADA 2026-07-04 (peso/musculatura se conservan)
    "makeup": "makeup", "tattoos": "tattoos", "bags": "bags", "belts": "belts",
    "jewelry": "jewelry", "headwear": "headwear", "eyewear": "eyewear",
    "activity": "Activity", "location": "location", "socialdistance": "social distance",
}

# Regex por clave para extraer `"key": { "class": "value"` incluso de JSON TRUNCADO
# (el modelo emite arrays "features" verbosos y en ~4% de imágenes la salida excede
# VLM_MAX_TOKENS → el JSON se corta sin cerrar y json.loads falla). Como cada
# `"class"` va al principio de su objeto, el valor SÍ está presente aunque el JSON no
# cierre. Se compila una vez.
_CLASS_RE = {
    key: re.compile(r'"' + re.escape(key) + r'"\s*:\s*\{\s*"class"\s*:\s*"([^"]*)"')
    for key in _KEY2LABEL
}


def _flatten_by_regex(raw: str):
    """Extrae pares class por regex (tolerante a JSON truncado/incompleto). Devuelve
    lista de líneas `Label: value`, o [] si no encuentra ninguna."""
    lines = []
    for key, label in _KEY2LABEL.items():
        m = _CLASS_RE[key].search(raw)
        if m:
            lines.append(f"{label}: {m.group(1)}")
    return lines


def _extract_json(text):
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def flatten_json_response(raw: str) -> str:
    """JSON de person_attrs/scene → líneas `Label: value`. Otro texto pasa igual.

    Tolerante a JSON TRUNCADO: si `json.loads` falla (salida cortada por
    VLM_MAX_TOKENS con los arrays "features" verbosos), cae a un extractor por regex
    que recupera los `"class"` ya generados (van al inicio de cada objeto). Sin esto,
    ~4% de imágenes con salida larga se aplanaban al JSON crudo y los parsers
    line-based devolvían todo 'no visible' pese a estar la respuesta correcta dentro.
    """
    # social_distance parts-visible: su propio parser lee el JSON crudo, no tocar.
    if raw and ('"shoulders"' in raw or '"frame_filled"' in raw):
        return raw

    obj = _extract_json(raw)
    if isinstance(obj, dict):
        lines = []
        for key, label in _KEY2LABEL.items():
            if key in obj:
                v = obj[key]
                val = v.get("class") if isinstance(v, dict) else v
                if val is not None:
                    lines.append(f"{label}: {val}")
        if lines:
            return "\n".join(lines)

    # Fallback tolerante: JSON incompleto/truncado o `_extract_json` devolvió un
    # sub-objeto sin las claves esperadas → extraer los class por regex.
    if raw and "{" in raw:
        lines = _flatten_by_regex(raw)
        if lines:
            return "\n".join(lines)
    return raw


class JsonFlatteningBackend:
    """Envuelve un VLMBackend y aplana su salida JSON a líneas tras generar."""

    # Forzamos el path 1×1 en los clasificadores para aplanar cada respuesta.
    supports_real_batch = False

    def __init__(self, inner):
        self.inner = inner

    def is_loaded(self):
        return self.inner.is_loaded()

    def load(self):
        return self.inner.load()

    def generate(self, pil_image, prompt, max_new_tokens=50, task_hint=None):
        raw = self.inner.generate(pil_image, prompt,
                                  max_new_tokens=max_new_tokens, task_hint=task_hint)
        return flatten_json_response(raw)

    def generate_batch(self, requests, max_new_tokens=50, task_hint=None):
        return [
            flatten_json_response(
                self.inner.generate(img, prompt,
                                    max_new_tokens=max_new_tokens, task_hint=task_hint)
            )
            for img, prompt in requests
        ]

    def __getattr__(self, name):
        # Delegar cualquier otro atributo (model, processor, etc.) al backend real.
        return getattr(self.inner, name)
