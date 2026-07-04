"""
Clasificador fusionado de contexto de escena (Merge B).

Reemplaza 2-3 llamadas VLM separadas (activity + location [+ social_distance])
por UNA sola llamada con un prompt multi-tarea que recibe la imagen completa
con la persona destacada en amarillo.

El prompt se construye por crop según `flags_include_social_distance`: cuando
el gate determinista de social_distance trigea (≥4 personas o 0 hombros), la
distancia social ya está decidida y NO se incluye en este prompt; cuando no
trigea, la social_distance se delega aquí (1 sola VLM call para 3 categorías
globales en frame completo).
"""

import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from importlib import import_module

import config

prompts_module = import_module(f"models.{getattr(config, 'VLM_PROMPT_MODULE', 'prompts_qwen3')}")


VALID_ACTIVITIES = ("sports", "romance", "posing", "other")
ACTIVITY_OTHER_ALIASES = {
    "entertaining", "everyday doings", "no activities", "na", "n/a"
}

VALID_LOCATIONS = ("indoors", "wilderness", "city", "no background")
# Default fallback cuando el parser no puede extraer ninguna categoría válida
# (modelo truncado, out-of-vocab, markdown no cubierto). Política del usuario
# 2026-06-02: location NUNCA debe ser "no visible". `no background` es la
# opción más conservadora — señala "sin escena identificable" sin asumir
# entorno concreto.
LOCATION_FALLBACK = "no background"

VALID_SOCIAL_DISTANCES = (
    "intimate distance",
    "close personal distance",
    "far personal distance",
    "close social distance",
    "far social distance",
    "public distance",
)


# Acepta prefijos de bullet/numbered list/markdown bold antes de la etiqueta
# (cubre `**Location:**`, `2. **Location:**`, `* Location:`, `> Location:`).
_LABEL_PREFIX = r'^[\s>*\-\d\.]*\**\s*'
_LABEL_SUFFIX = r'\**\s*:\s*\**\s*'


def _clean_value(raw: str) -> str:
    raw = raw.strip().lower()
    raw = re.sub(r'^[\*\s"\']+|[\*\s"\']+$', '', raw)
    raw = raw.rstrip('.,;:!')
    return raw


def _parse_activity(text: str) -> str:
    pattern = _LABEL_PREFIX + r'activity' + _LABEL_SUFFIX + r'([^\n]+)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    for raw_match in reversed(matches):
        raw = _clean_value(raw_match)
        if '<' in raw or '>' in raw:
            continue
        if raw in ACTIVITY_OTHER_ALIASES:
            return "other"
        for a in VALID_ACTIVITIES:
            if a in raw:
                return a
    return "other"


def _parse_location(text: str) -> str:
    pattern = _LABEL_PREFIX + r'location' + _LABEL_SUFFIX + r'([^\n]+)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    for raw_match in reversed(matches):
        raw = _clean_value(raw_match)
        if '<' in raw or '>' in raw:
            continue
        # NA ya no es opción del prompt v2 (closed-set de 4 categorías).
        # Si el modelo legacy aún lo emite, lo mapeamos al fallback.
        if raw in ('na', 'n/a'):
            return LOCATION_FALLBACK
        for loc in VALID_LOCATIONS:
            if loc in raw:
                return loc
    return LOCATION_FALLBACK


def _parse_social_distance(text: str) -> str | None:
    pattern = _LABEL_PREFIX + r'social\s*distance' + _LABEL_SUFFIX + r'([^\n]+)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    for raw_match in reversed(matches):
        raw = _clean_value(raw_match)
        if '<' in raw or '>' in raw:
            continue
        for cat in VALID_SOCIAL_DISTANCES:
            if cat in raw:
                return cat
    return None


class SceneContextClassifier:
    """
    Llama UNA vez al VLM por persona destacada y devuelve resultados para los
    clasificadores que comparten el mismo full-frame highlighted:

        activity + location [+ social_distance]
    """

    def __init__(self, backend=None):
        self.backend = backend

    def is_loaded(self) -> bool:
        return self.backend is not None and self.backend.is_loaded()

    def classify_batch(
        self,
        highlighted_frames: list[np.ndarray],
        flags_include_social_distance: list[bool] | None = None,
    ) -> tuple[list, list, list]:
        """
        Args:
            highlighted_frames: lista de frames BGR completos con la persona
                                marcada en amarillo.
            flags_include_social_distance: lista bool por crop indicando si el
                                prompt debe pedir también `Social Distance:`.
                                Si es None, asume todo False (legacy).

        Returns:
            (activity_results, location_results, social_distance_results).
            Cada entrada de social_distance_results es None si el flag fue False
            (la distancia social la resolvió el gate determinista upstream).
        """
        n = len(highlighted_frames)
        if flags_include_social_distance is None:
            flags_include_social_distance = [False] * n
        elif len(flags_include_social_distance) != n:
            raise ValueError(
                f"flags_include_social_distance length {len(flags_include_social_distance)} "
                f"!= highlighted_frames length {n}"
            )

        if not self.is_loaded():
            return self._all_errors(n, "Backend not loaded")
        if n == 0:
            return [], [], []

        any_social = any(flags_include_social_distance)
        print(
            f"  🌆 Contexto de escena (merge B) — {n} persona(s)"
            f"{' (incluye social_distance)' if any_social else ''}..."
        )

        activity_results = [None] * n
        location_results = [None] * n
        social_distance_results: list[dict | None] = [None] * n

        # ---- Preprocesado: BGR → PIL ----------------------------------------
        prepared: list[tuple[int, Image.Image | None, Exception | None]] = []
        for i, frame in enumerate(highlighted_frames):
            try:
                if frame is None or frame.size == 0:
                    raise ValueError("Invalid frame")
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prepared.append((i, pil_image, None))
            except Exception as e:
                prepared.append((i, None, e))

        # ---- Inferencia: batch real (Transformers) o ThreadPool (Ollama) ----
        valid = [(i, img) for (i, img, exc) in prepared if exc is None]
        outputs_by_idx: dict[int, str | Exception] = {}

        use_real_batch = getattr(self.backend, "supports_real_batch", False)
        if use_real_batch and valid:
            # Agrupamos por flag para que cada batch comparta prompt.
            groups: dict[bool, list[tuple[int, Image.Image]]] = {True: [], False: []}
            for (idx, img) in valid:
                groups[flags_include_social_distance[idx]].append((idx, img))
            for flag, items in groups.items():
                if not items:
                    continue
                try:
                    prompt = prompts_module.build_scene_prompt(flag)
                    texts = self.backend.generate_batch(
                        [(img, prompt) for (_, img) in items],
                        max_new_tokens=config.VLM_MAX_TOKENS,
                        task_hint="scene",
                    )
                    for (idx, _), text in zip(items, texts):
                        outputs_by_idx[idx] = (text or "").strip()
                except Exception as e:
                    for (idx, _) in items:
                        outputs_by_idx[idx] = e
        elif valid:
            def _call_one(idx, img, flag):
                try:
                    prompt = prompts_module.build_scene_prompt(flag)
                    text = self.backend.generate(
                        img, prompt, max_new_tokens=config.VLM_MAX_TOKENS,
                    ).strip()
                    return idx, text, None
                except Exception as e:
                    return idx, None, e

            inner_workers = min(len(valid), getattr(config, "OLLAMA_INNER_PARALLEL", 1))
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = [
                    pool.submit(_call_one, idx, img, flags_include_social_distance[idx])
                    for (idx, img) in valid
                ]
                for fut in futures:
                    idx, text, exc = fut.result()
                    outputs_by_idx[idx] = exc if exc is not None else text

        # ---- Parseo --------------------------------------------------------
        for (i, _, prep_exc) in prepared:
            flag = flags_include_social_distance[i]
            if prep_exc is not None:
                err = str(prep_exc)
                print(f"    ❌ Persona {i+1}: {err}")
                activity_results[i] = {"activity": None, "error": err, "success": False}
                location_results[i] = {"location": None, "error": err, "success": False}
                if flag:
                    social_distance_results[i] = {
                        "category": None, "error": err, "success": False,
                    }
                continue

            out = outputs_by_idx.get(i)
            if isinstance(out, Exception):
                err = str(out)
                print(f"    ❌ Persona {i+1}: {err}")
                activity_results[i] = {"activity": None, "error": err, "success": False}
                location_results[i] = {"location": None, "error": err, "success": False}
                if flag:
                    social_distance_results[i] = {
                        "category": None, "error": err, "success": False,
                    }
                continue

            output_text = out or ""
            activity = _parse_activity(output_text)
            location = _parse_location(output_text)

            activity_results[i] = {
                "activity": activity,
                "raw_response": output_text,
                "success": True,
            }
            location_results[i] = {
                "location": location,
                "raw_response": output_text,
                "success": True,
            }

            sd_log = ""
            if flag:
                sd_label = _parse_social_distance(output_text)
                if sd_label is None:
                    social_distance_results[i] = {
                        "category": None,
                        "raw_response": output_text,
                        "success": False,
                        "error": "social_distance line not found in scene response",
                    }
                    sd_log = ", social_distance=PARSE_FAIL"
                else:
                    social_distance_results[i] = {
                        "category": sd_label,
                        "raw_response": output_text,
                        "success": True,
                    }
                    sd_log = f", {sd_label}"

            print(f"    ✅ Persona {i+1}: {activity}, {location}{sd_log}")

        return activity_results, location_results, social_distance_results

    @staticmethod
    def _all_errors(n: int, error: str) -> tuple[list, list, list]:
        empty_act = {"activity": None, "error": error, "success": False}
        empty_loc = {"location": None, "error": error, "success": False}
        empty_sd = {"category": None, "error": error, "success": False}
        return [empty_act] * n, [empty_loc] * n, [empty_sd] * n
