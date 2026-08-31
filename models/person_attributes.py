"""
Clasificador fusionado de atributos por persona (Merge A).

Reemplaza 6 llamadas VLM separadas (gender + age + behaviour + body_display +
body_shape + accessory) por UNA sola llamada con un prompt multi-tarea que
recibe el person crop. Mantiene el gate por keypoints para body_shape: si la
cintura no es visible, esa sección se omite del prompt y se devuelve
"not visible" sin coste de tokens.

La salida del backend se parsea en 6 dicts separados con el MISMO contrato que
los clasificadores originales (BehaviourClassifier, BodyDisplayClassifier,
BodyShapeClassifier, AccessoryClassifier, AgeGenderClassifier) para que el
resto del pipeline (`processing/image.py`) consuma los resultados sin cambios.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from importlib import import_module

import config
from models.accessory import ACCESSORY_CATEGORIES

prompts_module = import_module(f"models.{getattr(config, 'VLM_PROMPT_MODULE', 'prompts_it')}")


# Categorías válidas (idénticas a las de los clasificadores individuales).
VALID_GENDERS = ("male", "female")
VALID_AGE_GROUPS = ("childhood", "youth", "adulthood", "old age")
VALID_BEHAVIOURS = (
    "demand/affiliation",
    "demand/seduction",
    "demand/submission",
    "offer/ideal",
)
VALID_BODY_DISPLAYS = (
    "normal clothes",
    "revealing clothes",
    "partially naked",
    "no clothes at all",
)
# Silueta ELIMINADA 2026-07-04 (κ 0.306 en sample 500; peso/musculatura se conservan).
# Clases de peso de Stunkard (adoption study Table 1). Orden = prioridad de match
# por substring (las más específicas/largas primero para no colisionar; 'overweight'
# antes que cualquier 'weight' suelto).
VALID_BODY_WEIGHTS = (
    "overweight",
    "obese",
    "median",
    "thin",
    "not visible",
)
# Sinónimos que gemma4 suele emitir → clase canónica (se comprueban tras el closed-set).
_BODY_WEIGHT_SYNONYMS = {
    "underweight": "thin", "skinny": "thin", "lean": "thin", "slim": "thin", "light": "thin",
    "normal": "median", "average": "median", "healthy": "median",
}
# Reetiquetado a ADIPOSITY (2026-07-22): el clasificador mide grasa corporal (el prompt
# ya define la tarea como "body fatness"). El modelo SIGUE emitiendo los tokens calibrados
# (light build/median/overweight) — la palabra exacta es crítica (ver CLAUDE.md) — pero el
# valor canónico que se persiste en el campo interno `body_weight` es la clase de adiposidad
# low/medium/high. Es una biyección con thin/median/overweight (κ invariante). El nombre del
# campo interno `body_weight` se conserva; solo cambia el vocabulario del VALOR.
_ADIPOSITY_MAP = {"thin": "low", "median": "medium", "overweight": "high", "obese": "high"}


def _to_adiposity(canon: str | None) -> str | None:
    """Traduce la clase canónica de peso a adiposidad (low/medium/high). 'not visible'/None pasan igual."""
    if canon is None:
        return None
    return _ADIPOSITY_MAP.get(canon, canon)
# Musculatura visible (binaria). Se comprueba el NEGATIVO primero: "visible" es
# substring de "not visible" (mismo caso que female/male en _parse_gender).
VALID_MUSCLES = (
    "not visible",
    "visible",
)
# Vestimenta (estilo/formalidad del atuendo). Orden por prioridad de match substring:
# las etiquetas más largas/específicas primero para no colisionar (p.ej. "underwear/
# swimwear" antes que cualquier "wear" suelto). "not visible" primero como en muscle.
VALID_ATTIRE = (
    "not visible",
    "underwear/swimwear",
    "sportswear",
    "uniform",
    "formal",
    "casual",
)
# Sinónimos que el VLM suele emitir → clase canónica (se comprueban tras el closed-set).
_ATTIRE_SYNONYMS = {
    "swimsuit": "underwear/swimwear", "swimwear": "underwear/swimwear",
    "bikini": "underwear/swimwear", "swim trunks": "underwear/swimwear",
    "trunks": "underwear/swimwear", "lingerie": "underwear/swimwear",
    "underwear": "underwear/swimwear", "bra": "underwear/swimwear",
    "briefs": "underwear/swimwear", "boxers": "underwear/swimwear",
    "athletic": "sportswear", "gym": "sportswear", "tracksuit": "sportswear",
    "jersey": "sportswear", "activewear": "sportswear", "sports": "sportswear",
    "military": "uniform", "police": "uniform", "army": "uniform",
    "firefighter": "uniform", "medical": "uniform", "scrubs": "uniform",
    "suit": "formal", "tuxedo": "formal", "gown": "formal", "tie": "formal",
    "blazer": "formal", "traditional": "formal", "tunic": "formal",
    "kimono": "formal", "sari": "formal", "kilt": "formal", "thobe": "formal",
    "everyday": "casual", "streetwear": "casual", "street": "casual",
    "jeans": "casual", "t-shirt": "casual", "hoodie": "casual", "normal": "casual",
}


# Prompts pre-compilados desde models/prompts_it.py (estilo IT sin razonamiento).
# El loop edita prompts_it.py; aquí sólo se referencian.
# Silueta eliminada → un solo prompt (peso/musculatura siempre presentes).
_PROMPT_WITH_SHAPE = _PROMPT_NO_SHAPE = prompts_module.build_person_attrs_prompt()


# ---------------------------------------------------------------------------
# Parsers (uno por sub-tarea). Devuelven el valor (str|None) o el dict de salida.
# ---------------------------------------------------------------------------


def _parse_gender(text: str) -> str | None:
    match = re.search(r'^\s*Gender\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'no visible'
    if 'female' in raw:
        return 'female'
    if 'male' in raw:
        return 'male'
    return None


def _parse_age(text: str) -> str | None:
    match = re.search(r'^\s*Age\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'no visible'
    return next((g for g in VALID_AGE_GROUPS if g in raw), None)


# Mapa sufijo-desnudo → categoría completa. El prompt pide `demand/affiliation`
# etc., pero según el backend el modelo a veces emite solo el sufijo
# (`behaviour: affiliation`). El sufijo determina unívocamente la categoría
# (los tres affiliation/seduction/submission son demand/*, ideal es offer/*),
# así que mapearlo es seguro. Orden: sufijos más largos primero para evitar
# que un substring corto gane.
_BEHAVIOUR_SUFFIX_MAP = {
    "affiliation": "demand/affiliation",
    "seduction":   "demand/seduction",
    "submission":  "demand/submission",
    "ideal":       "offer/ideal",
}


def _parse_behaviour(text: str) -> str | None:
    match = re.search(r'^\s*Behaviour\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'no visible'
    # 1) forma completa (`demand/affiliation`, `offer/ideal`) — la que emite Ollama.
    hit = next((b for b in VALID_BEHAVIOURS if b in raw), None)
    if hit:
        return hit
    # 2) sufijo desnudo (`affiliation`, `ideal`) — la que emite Transformers.
    return next((full for suf, full in _BEHAVIOUR_SUFFIX_MAP.items() if suf in raw), None)


def _parse_body_display(text: str) -> str | None:
    match = re.search(r'^\s*BodyDisplay\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'no visible'
    return next((bd for bd in VALID_BODY_DISPLAYS if bd in raw), None)


def _parse_body_weight(text: str) -> str | None:
    """Extrae la categoría FRS/IMC de la línea 'BodyWeight: <...>'."""
    match = re.search(r'^\s*BodyWeight\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'not visible'
    hit = next((w for w in VALID_BODY_WEIGHTS if w in raw), None)
    if hit:
        return _to_adiposity(hit)
    syn = next((canon for syn, canon in _BODY_WEIGHT_SYNONYMS.items() if syn in raw), None)
    return _to_adiposity(syn)


def _parse_muscle(text: str) -> str | None:
    """Extrae la musculatura visible (binaria) de la línea 'Musculature: <...>'.

    Comprueba el NEGATIVO primero porque 'visible' es substring de 'not visible'
    (mismo patrón anti-substring que _parse_gender con female/male)."""
    match = re.search(r'^\s*Musculature\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if any(neg in raw for neg in ('not visible', 'no visible', 'none', 'absent', 'soft', 'na', 'n/a', '0')):
        return 'not visible'
    if any(pos in raw for pos in ('visible', 'muscular', 'defined', 'toned', 'athletic', '1')):
        return 'visible'
    return None


def _parse_attire(text: str) -> str | None:
    """Extrae la vestimenta (estilo/formalidad) de la línea 'Attire: <...>'."""
    match = re.search(r'^\s*Attire\s*:\s*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip().lower()
    if raw in ('na', 'n/a'):
        return 'not visible'
    # "informal" contiene "formal" → tratar antes del closed-set para no mapear a formal.
    if 'informal' in raw:
        return 'casual'
    hit = next((a for a in VALID_ATTIRE if a in raw), None)
    if hit:
        return hit
    return next((canon for syn, canon in _ATTIRE_SYNONYMS.items() if syn in raw), None)


def _parse_accessories(text: str) -> dict:
    binary = {cat: 0 for cat in ACCESSORY_CATEGORIES}
    for cat in ACCESSORY_CATEGORIES:
        match = re.search(rf'^\s*{cat}\s*:\s*([01])', text, re.IGNORECASE | re.MULTILINE)
        if match:
            binary[cat] = int(match.group(1))
    return binary


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------


class PersonAttributesClassifier:
    """
    Llama UNA vez al VLM por persona y devuelve resultados para los 6
    clasificadores que comparten el mismo person crop:

        gender + age + behaviour + body_display + body_shape + accessory

    Si `has_waist_visible == False`, el prompt omite el bloque body_shape y la
    salida correspondiente se rellena con "not visible" determinísticamente.
    """

    def __init__(self, backend=None):
        self.backend = backend

    def is_loaded(self) -> bool:
        return self.backend is not None and self.backend.is_loaded()

    def classify_batch(
        self,
        person_crops: list[np.ndarray],
        has_waist_visible_flags: list[bool],
    ) -> tuple[list, list, list, list, list, list]:
        """
        Args:
            person_crops:              recortes BGR (OpenCV) de cada persona.
            has_waist_visible_flags:   True si los keypoints de waist están
                                       visibles para esa persona; False en
                                       caso contrario.

        Returns:
            (gender_results, age_results, behaviour_results,
             body_display_results, body_shape_results, accessory_results)
            con el formato exacto que esperan los clasificadores individuales.
        """
        n = len(person_crops)
        assert len(has_waist_visible_flags) == n, "flags y crops deben alinearse"

        if not self.is_loaded():
            return self._all_errors(n, "Backend not loaded")

        if n == 0:
            return [], [], [], [], [], []

        print(f"  🧬 Atributos de persona (merge A) — {n} persona(s)...")

        gender_results = [None] * n
        age_results = [None] * n
        behaviour_results = [None] * n
        body_display_results = [None] * n
        body_shape_results = [None] * n
        accessory_results = [None] * n

        # ---- Preprocesado: BGR → PIL + prompt por persona --------------------
        prepared: list[tuple[int, bool, Image.Image | None, str | None, Exception | None]] = []
        for i, (crop, has_waist) in enumerate(zip(person_crops, has_waist_visible_flags)):
            try:
                if crop is None or crop.size == 0:
                    raise ValueError("Invalid crop")
                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                prompt = _PROMPT_WITH_SHAPE if has_waist else _PROMPT_NO_SHAPE
                prepared.append((i, has_waist, pil_image, prompt, None))
            except Exception as e:
                prepared.append((i, has_waist, None, None, e))

        # ---- Inferencia: batch real (Transformers) o ThreadPool (Ollama) -----
        valid = [(i, hw, img, p) for (i, hw, img, p, exc) in prepared if exc is None]
        outputs_by_idx: dict[int, str | Exception] = {}

        use_real_batch = getattr(self.backend, "supports_real_batch", False)
        if use_real_batch and valid:
            try:
                texts = self.backend.generate_batch(
                    [(img, p) for (_, _, img, p) in valid],
                    max_new_tokens=config.VLM_MAX_TOKENS,
                    task_hint="person_attrs",
                )
                for (idx, _, _, _), text in zip(valid, texts):
                    outputs_by_idx[idx] = (text or "").strip()
            except Exception as e:
                for (idx, _, _, _) in valid:
                    outputs_by_idx[idx] = e
        elif valid:
            def _call_one(idx, img, prompt):
                try:
                    text = self.backend.generate(
                        img, prompt, max_new_tokens=config.VLM_MAX_TOKENS,
                    ).strip()
                    return idx, text, None
                except Exception as e:
                    return idx, None, e

            inner_workers = min(len(valid), getattr(config, "OLLAMA_INNER_PARALLEL", 1))
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = [pool.submit(_call_one, idx, img, p) for (idx, _, img, p) in valid]
                for fut in futures:
                    idx, text, exc = fut.result()
                    outputs_by_idx[idx] = exc if exc is not None else text

        # ---- Reintento ante salida VACÍA (no excepción) ----------------------
        # En runs largos, una generación greedy borderline puede colapsar a EOS
        # inmediato bajo la presión numérica/de memoria del proceso → texto vacío.
        # No es el parser ni el crop: el MISMO crop clasifica bien en aislamiento
        # (diagnosticado 2026-07-01: ~4% de un run de 500 salían 'no visible' con
        # merge-A vacía mientras merge-B acertaba). Reintentar UNA vez tras liberar
        # la caché CUDA cambia el estado de ejecución y suele recuperar la salida.
        empty_idxs = [idx for (idx, _, _, _) in valid
                      if not isinstance(outputs_by_idx.get(idx), Exception)
                      and not (outputs_by_idx.get(idx) or "").strip()]
        if empty_idxs:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            by_idx = {i: (img, p) for (i, _, img, p) in valid}
            for idx in empty_idxs:
                img, p = by_idx[idx]
                try:
                    retry = (self.backend.generate(
                        img, p, max_new_tokens=config.VLM_MAX_TOKENS,
                    ) or "").strip()
                    if retry:
                        outputs_by_idx[idx] = retry
                        print(f"    ↻ Persona {idx+1}: reintento tras salida vacía → OK")
                    else:
                        print(f"    ⚠ Persona {idx+1}: salida vacía también tras reintento")
                except Exception as e:
                    outputs_by_idx[idx] = e

        # ---- Parseo y construcción de resultados -----------------------------
        for (i, has_waist, _, _, prep_exc) in prepared:
            if prep_exc is not None:
                err = str(prep_exc)
                print(f"    ❌ Persona {i+1}: {err}")
                gender_results[i] = self._error_dict("gender", err)
                age_results[i] = self._error_dict("age_group", err)
                behaviour_results[i] = self._error_dict("behaviour", err)
                body_display_results[i] = self._error_dict("body_display", err)
                body_shape_results[i] = self._error_dict("body_weight", err)
                accessory_results[i] = self._error_dict_accessory(err)
                continue

            out = outputs_by_idx.get(i)
            if isinstance(out, Exception):
                err = str(out)
                print(f"    ❌ Persona {i+1}: {err}")
                gender_results[i] = self._error_dict("gender", err)
                age_results[i] = self._error_dict("age_group", err)
                behaviour_results[i] = self._error_dict("behaviour", err)
                body_display_results[i] = self._error_dict("body_display", err)
                body_shape_results[i] = self._error_dict("body_weight", err)
                accessory_results[i] = self._error_dict_accessory(err)
                continue

            output_text = out or ""
            gender_results[i] = self._build_gender_result(output_text)
            age_results[i] = self._build_age_result(output_text)
            behaviour_results[i] = self._build_behaviour_result(output_text)
            body_display_results[i] = self._build_body_display_result(output_text)
            # Silueta eliminada: este slot ahora solo lleva peso (FRS/IMC) + musculatura,
            # que se preguntan siempre (no dependen del gate de cintura).
            body_shape_results[i] = self._build_body_shape_result(output_text)
            accessory_results[i] = self._build_accessory_result(output_text)

            summary_bits = [
                gender_results[i].get("gender") or "?",
                age_results[i].get("age_group") or "?",
                behaviour_results[i].get("behaviour") or "?",
            ]
            print(f"    ✅ Persona {i+1}: {', '.join(summary_bits)}")

        return (
            gender_results,
            age_results,
            behaviour_results,
            body_display_results,
            body_shape_results,
            accessory_results,
        )

    # ---- helpers de construcción de resultados con el contrato esperado ---

    @staticmethod
    def _build_gender_result(text: str) -> dict:
        gender = _parse_gender(text) or 'no visible'
        return {
            "gender": gender,
            "all_predictions": [{"label": gender, "score": None}],
            "raw_response": text,
            "success": gender is not None,
        }

    @staticmethod
    def _build_age_result(text: str) -> dict:
        age_group = _parse_age(text) or 'no visible'
        return {
            "age_group": age_group,
            "raw_response": text,
            "success": age_group is not None,
        }

    @staticmethod
    def _build_behaviour_result(text: str) -> dict:
        behaviour = _parse_behaviour(text) or 'no visible'
        return {
            "behaviour": behaviour,
            "raw_response": text,
            "success": behaviour is not None,
        }

    @staticmethod
    def _build_body_display_result(text: str) -> dict:
        body_display = _parse_body_display(text) or 'no visible'
        return {
            "body_display": body_display,
            "raw_response": text,
            "success": body_display is not None,
        }

    @staticmethod
    def _build_body_shape_result(text: str) -> dict:
        # Silueta eliminada 2026-07-04: este slot lleva peso + musculatura + vestimenta.
        return {
            "body_weight": _parse_body_weight(text) or "not visible",
            "muscle": _parse_muscle(text) or "not visible",
            "attire": _parse_attire(text) or "not visible",
            "raw_response": text,
            "success": True,
        }

    @staticmethod
    def _build_accessory_result(text: str) -> dict:
        binary = _parse_accessories(text)
        result = dict(binary)
        result.update({
            "raw_response": text,
            "success": True,
        })
        return result

    # ---- helpers de error -------------------------------------------------

    @staticmethod
    def _error_dict(field: str, error: str) -> dict:
        return {field: None, "error": error, "success": False}

    @staticmethod
    def _error_dict_accessory(error: str) -> dict:
        result = {cat: 0 for cat in ACCESSORY_CATEGORIES}
        result.update({"error": error, "success": False})
        return result

    def _all_errors(self, n: int, error: str) -> tuple[list, list, list, list, list, list]:
        return (
            [self._error_dict("gender", error) for _ in range(n)],
            [self._error_dict("age_group", error) for _ in range(n)],
            [self._error_dict("behaviour", error) for _ in range(n)],
            [self._error_dict("body_display", error) for _ in range(n)],
            [self._error_dict("body_weight", error) for _ in range(n)],
            [self._error_dict_accessory(error) for _ in range(n)],
        )
