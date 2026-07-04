"""
Clasificador de distancia social proxémica usando el modelo VLM compartido.
Basado en la teoría de proxémica de Edward T. Hall.

Modo activo (2026-06-01): el VLM emite un JSON con la visibilidad de partes
del cuerpo del TARGET y Python aplica el mapeo determinista a categoría.
El VLM nunca ve los nombres de las 6 categorías → reduce la hallucination
de iter6 (parts-enumeration + categoría en el mismo paso).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from importlib import import_module

import config

prompts_module = import_module(f"models.{getattr(config, 'VLM_PROMPT_MODULE', 'prompts_qwen3')}")


def _map_parts_to_category(parts: dict) -> str | None:
    """
    Mapeo determinista de visibilidad de partes → categoría proxémica.
    Reglas en orden (STOP at first match):
      shoulders=0                              → intimate distance        (head only)
      shoulders=1, waist=0, legs=0, feet=0     → close personal distance  (head + shoulders [± chest])
      waist=1,    legs=0,  feet=0              → far personal distance    (waist up)
      legs=1,     feet=0                       → close social distance    (legs cropped at feet)
      feet=1, frame_filled=1                   → close social distance    (full body, fills frame)
      feet=1, frame_filled=0                   → far social distance      (full body, empty space dominates)
    """
    try:
        sh = int(parts.get("shoulders", 0))
        wa = int(parts.get("waist", 0))
        lg = int(parts.get("legs", 0))
        ft = int(parts.get("feet", 0))
        ff = int(parts.get("frame_filled", 1))
    except (TypeError, ValueError):
        return None

    if sh == 0:
        return "intimate distance"
    if ft == 1:
        return "close social distance" if ff == 1 else "far social distance"
    if lg == 1:
        return "close social distance"
    if wa == 1:
        return "far personal distance"
    return "close personal distance"


class SocialDistanceClassifier:
    """
    Clasificador de distancia social usando el backend VLM compartido.
    Recibe la imagen completa con la persona objetivo marcada con un recuadro amarillo etiquetado "TARGET".
    """

    CATEGORIES = [
        "intimate distance",
        "close personal distance",
        "far personal distance",
        "close social distance",
        "far social distance",
        "public distance",
    ]

    def __init__(self, backend=None):
        self.backend = backend
        # Prompt pre-compilado desde models/prompts_it.py (estilo IT sin razonamiento).
        self.prompt = prompts_module.SOCIAL_DISTANCE_PROMPT

    def classify_from_keypoints(self, kpts, bbox: tuple, frame_shape: tuple,
                                num_persons: int) -> dict | None:
        """
        Clasificación proxémica DETERMINISTA por keypoints (2026-06-29).

        Con `config.SOCIAL_DISTANCE_KEYPOINT_ONLY=True` (defecto) mapea las 6
        categorías a partir de la parte COCO visible más baja de la persona +
        el tamaño de su bbox, espejando `_map_parts_to_category` pero con la
        visibilidad derivada de los keypoints de YOLO26-pose. NUNCA delega al VLM
        salvo que falten keypoints. Validado sobre el banco FLUX social_distance
        (31/36 = 86%; los 5 fallos son imágenes FLUX mal generadas → ~36/36 en
        imágenes válidas).

        Tabla (STOP at first match, espeja producción):
        - num_persons >= 4                              → public distance
        - sin keypoints                                 → None (fallback VLM/legacy)
        - no hombros (5,6) visibles                     → intimate distance
        - piernas (rodillas 13,14 o tobillos 15,16)     → close social (bbox llena)
                                                          / far social (bbox pequeña)
        - caderas (11,12) visibles, sin piernas         → far personal distance
        - solo hombros                                  → close personal distance

        Modo legacy (`SOCIAL_DISTANCE_KEYPOINT_ONLY=False`): solo extremos
        (>=4→public, 0 hombros@0.65→intimate), el resto → None → VLM.

        Args:
            kpts: array numpy (17, 3) con [x, y, conf] por keypoint. None si no disponible.
            bbox: (x1, y1, x2, y2) del bounding box de la persona en el frame.
            frame_shape: (height, width[, channels]) del frame completo.
            num_persons: número total de personas detectadas en el frame.

        Returns:
            dict con category/success, o None para delegar al VLM
            (solo cuando faltan keypoints).
        """
        if num_persons >= 4:
            return {"category": "public distance",
                    "success": True, "raw_response": "keypoints:>=4_persons"}

        if kpts is None or len(kpts) < 17:
            return None  # sin keypoints → fallback a VLM/legacy

        if not getattr(config, "SOCIAL_DISTANCE_KEYPOINT_ONLY", True):
            # --- Modo legacy: solo extremos, el resto al VLM ---
            shoulders_visible = sum(1 for idx in (5, 6) if kpts[idx][2] >= 0.65)
            if shoulders_visible == 0:
                return {"category": "intimate distance",
                        "success": True, "raw_response": "keypoints:no_shoulders"}
            return None

        # --- Mapeo determinista completo por keypoints ---
        part_conf = getattr(config, "SOCIAL_DISTANCE_PART_CONF", 0.5)
        fill_t = getattr(config, "SOCIAL_DISTANCE_FILL_RATIO", 0.375)

        def _vis(pair):
            return any(kpts[i][2] >= part_conf for i in pair)

        sh = _vis((5, 6))            # hombros
        wa = _vis((11, 12))          # caderas/cintura
        legs = _vis((13, 14)) or _vis((15, 16))  # rodillas o tobillos = cuerpo

        x1, y1, x2, y2 = bbox
        fh, fw = frame_shape[0], frame_shape[1]
        bbox_ratio = (max(0, x2 - x1) * max(0, y2 - y1)) / max(1, fh * fw)
        ff = bbox_ratio >= fill_t    # bbox llena el frame
        raw = f"keypoints:sh{int(sh)}_wa{int(wa)}_legs{int(legs)}_ff{int(ff)}_r{bbox_ratio:.3f}"

        if not sh:
            cat = "intimate distance"
        elif legs:
            cat = "close social distance" if ff else "far social distance"
        elif wa:
            cat = "far personal distance"
        else:
            cat = "close personal distance"

        return {"category": cat, "success": True,
                "raw_response": raw}

    def classify_batch(self, images: list) -> list:
        """
        Clasifica la distancia social de múltiples personas.

        Args:
            images: Lista de imágenes BGR completas con la persona objetivo marcada en rojo

        Returns:
            Lista de dicts con categoría de distancia social para cada imagen
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"category": None, "error": "Backend not loaded", "success": False}] * len(images)

        if not images:
            return []

        if len(images) >= 4:
            print(f"  📏 {len(images)} personas detectadas → atajo determinista a 'public distance' (sin LLM)")
            return [
                {
                    "category": "public distance",
                    "raw_response": "shortcut: >=4 persons detected",
                    "success": True,
                }
                for _ in images
            ]

        print(f"  📏 Analizando distancia social de {len(images)} persona(s)...")

        n = len(images)
        results: list[dict | None] = [None] * n

        # ---- Preprocesado: BGR → PIL ----------------------------------------
        prepared: list[tuple[int, Image.Image | None, str | None]] = []
        for i, img in enumerate(images):
            if img is None or img.size == 0:
                prepared.append((i, None, "Invalid image"))
                continue
            try:
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                prepared.append((i, pil_image, None))
            except Exception as e:
                prepared.append((i, None, str(e)))

        valid = [(i, img) for (i, img, err) in prepared if err is None]
        outputs_by_idx: dict[int, str | Exception] = {}

        use_real_batch = getattr(self.backend, "supports_real_batch", False)
        if use_real_batch and valid:
            try:
                texts = self.backend.generate_batch(
                    [(img, self.prompt) for (_, img) in valid],
                    max_new_tokens=config.VLM_MAX_TOKENS,
                    task_hint="social_distance",
                )
                for (idx, _), text in zip(valid, texts):
                    outputs_by_idx[idx] = text or ""
            except Exception as e:
                for (idx, _) in valid:
                    outputs_by_idx[idx] = e
        elif valid:
            def _call_one(idx, img):
                try:
                    text = self.backend.generate(
                        img, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS,
                    )
                    return idx, text, None
                except Exception as e:
                    return idx, None, e

            inner_workers = min(len(valid), getattr(config, "OLLAMA_INNER_PARALLEL", 1))
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = [pool.submit(_call_one, idx, img) for (idx, img) in valid]
                for fut in futures:
                    idx, text, exc = fut.result()
                    outputs_by_idx[idx] = exc if exc is not None else text

        # ---- Parseo --------------------------------------------------------
        for (i, _, err) in prepared:
            if err is not None:
                results[i] = {"category": None, "error": err, "success": False}
                continue

            out = outputs_by_idx.get(i)
            if isinstance(out, Exception):
                results[i] = {"category": None, "error": str(out), "success": False}
            else:
                category = self._parse_response(out or "")
                results[i] = {
                    "category": category,
                    "raw_response": (out or "").strip(),
                    "success": category is not None,
                }

            r = results[i]
            if r.get("success"):
                print(f"    ✅ Persona {i+1}: {r.get('category', 'Unknown')}")
            else:
                print(f"    ❌ Persona {i+1}: Error - {r.get('error', 'Unknown')}")

        successful = sum(1 for r in results if r and r.get('success'))
        print(f"  ✅ Distancia social completado: {successful} exitosos de {len(results)}")
        return results

    def _classify_single(self, pil_image: Image.Image) -> dict:
        """Clasifica una sola imagen PIL."""
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            category = self._parse_response(output_text)
            return {
                "category": category,
                "raw_response": output_text.strip(),
                "success": category is not None
            }
        except Exception as e:
            return {"category": None, "error": str(e), "success": False}

    def _parse_response(self, text: str) -> str | None:
        """
        Parsea el JSON parts-visible emitido por el VLM y aplica el mapping determinista.
        Devuelve None si el JSON está malformado o faltan campos clave.
        """
        if not text:
            return None
        # Localiza el primer objeto JSON en el output (puede venir con preamble/thinking).
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            parts = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parts, dict):
            return None
        return _map_parts_to_category(parts)
