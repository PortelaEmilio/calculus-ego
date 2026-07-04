"""
Estimador de belleza facial 1-5 (SCUT-FBP5500).

Usa un backend VLM con `.generate(image, prompt, max_new_tokens)`:
  - en el pase separado de belleza → Gemma4BeautyBackend (gemma-4-E4B 4-bit + LoRA, Transformers/PEFT)
  - (legacy) el backend Ollama compartido, si alguna vez se activara en el flujo en vivo.
El prompt y la escala (1-5) deben coincidir con el entrenamiento del adapter.
"""

import cv2
import numpy as np
from PIL import Image

import config


class BeautyEstimator:
    """
    Estimador de belleza usando el backend VLM compartido.
    Se activa solo para personas detectadas de frente.
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        # Prompt + escala según el backend de belleza activo. DEBEN ser coherentes con
        # el modelo servido (el prompt no tiene que clavar el de entrenamiento — sobre
        # Qwen3-VL el EN 1-10 rinde igual o mejor que el ES, ver config).
        #   - "ollama"       → Qwen3.5-9B deciles 1-10 (Ollama)
        #   - "qwen35_cont"  → Qwen3.5-9B CONTINUO 1-10 (Transformers, checkpoint-1400) ← producción
        #   - "transformers" → gemma4 1-5 (legacy)
        backend_kind = getattr(config, "BEAUTY_BACKEND", "transformers")
        if backend_kind in ("ollama", "qwen35_cont"):
            self.prompt = getattr(config, "BEAUTY_PROMPT_QWEN", (
                "Rate the beauty of this person on a scale from 1 to 10. "
                "Respond with only the number."))
            self.scale = tuple(getattr(config, "BEAUTY_SCALE_QWEN", (1.0, 10.0)))
        else:
            self.prompt = getattr(config, "BEAUTY_PROMPT", (
                "Rate the facial attractiveness of this face on a scale of 1 to 5, "
                "where 1 is least attractive and 5 is most attractive. "
                "Respond with only the number."))
            self.scale = tuple(getattr(config, "BEAUTY_SCALE", (1.0, 5.0)))

    def estimate(self, image_crop: np.ndarray) -> dict:
        """
        Estima la belleza de un rostro.

        Args:
            image_crop: Recorte de imagen BGR (OpenCV) de la persona

        Returns:
            Dict con score de belleza y metadata
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"score": None, "error": "Backend not loaded"}

        try:
            # Convertir BGR a RGB y luego a PIL
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Usar el método interno para clasificar
            result = self._estimate_single(pil_image)

            return result

        except Exception as e:
            return {
                "score": None,
                "error": str(e),
                "success": False
            }

    def _estimate_single(self, pil_image: Image.Image) -> dict:
        """Estima belleza de una sola imagen PIL."""
        try:
            max_tok = getattr(config, "BEAUTY_MAX_NEW_TOKENS", 8)
            response = self.backend.generate(pil_image, self.prompt, max_new_tokens=max_tok)
            score = self._extract_score(response)

            return {
                "score": score,
                "raw_response": response,
                "success": score is not None
            }

        except Exception as e:
            return {
                "score": None,
                "error": str(e),
                "raw_response": None,
                "success": False
            }

    def _extract_score(self, text: str) -> float | str | None:
        """
        Extrae el score numérico de la respuesta del modelo.

        Args:
            text: Texto generado por el modelo

        Returns:
            Score de belleza (0-100), "NA" si no hay ojos visibles/cara pixelada, o None si no se puede extraer
        """
        import re

        # Limpiar texto
        text_clean = text.strip()

        # Verificar si la respuesta es "NA" - ser más específico para evitar falsos positivos
        # Solo buscar patrones que indiquen claramente que no se puede evaluar
        na_patterns = [
            r'^N\s*/\s*A$',                 # "N/A" solo (principio a fin)
            r'^NA$',                         # "NA" solo (principio a fin)
            r'^n/a$',                        # "n/a" solo (principio a fin)
            r'eyes\s+(are\s+)?(completely|fully|not|cannot)\s+(hidden|visible|seen)', # ojos no visibles
            r'cannot\s+see\s+(the\s+)?eyes',  # no puede ver los ojos
            r'eyes\s+not\s+visible',         # ojos no visibles
            r'heavily\s+pixelated',          # muy pixelada
            r'fully\s+covered',              # completamente cubierta
            r'completely\s+(obscured|covered)', # completamente oscurecida/cubierta
            r'cannot\s+rate',                # no puede calificar
            r'unable\s+to\s+rate',           # incapaz de calificar
        ]

        for pattern in na_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return "no visible"

        # Escala según el backend (1-5 gemma4 / 1-10 Qwen). El modelo finetuneado emite
        # el número solo (ej. "2.8" o "8"); priorizamos el número al inicio y validamos
        # rango contra self.scale, devolviéndolo SIN reescalar.
        lo, hi = self.scale
        patterns = [
            r'^\s*(\d+(?:\.\d+)?)',            # número al inicio (salida típica del finetune)
            rf'(\d+(?:\.\d+)?)\s*/\s*{int(hi)}',  # "8/10" o "4/5"
            r'score.*?(\d+(?:\.\d+)?)',        # "score: 3"
            r'rating.*?(\d+(?:\.\d+)?)',       # "rating: 3"
            r'\b(\d{1,2}(?:\.\d+)?)\b',        # fallback (1-2 dígitos; rango lo filtra)
        ]

        for pattern in patterns:
            match = re.search(pattern, text_clean.lower())
            if match:
                try:
                    score = float(match.group(1))
                    if lo <= score <= hi:
                        return round(score, 2)
                except ValueError:
                    continue

        # Si no se encontró ni NA ni un número válido, devolver None
        return None
