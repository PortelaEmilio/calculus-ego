"""
Clasificador de actividades usando el modelo Qwen3-VL compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


class ActivityClassifier:
    """
    Clasificador de actividades usando el backend VLM compartido.

    Categorías:
    - sports: gym, exercise, physical activity
    - romance: kissing, hugging romantically
    - posing: posing for camera, modeling
    - other: entertaining, everyday doings, no activities, na — all merged
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        # Prompt para clasificación de actividades
        self.prompt = """Classify the main activity in the image into ONE label.

You MUST start your response with the verdict in this exact format, BEFORE any reasoning:
Activity: <one category>

Then, on the lines below, you may explain your reasoning. The verdict line at the top is mandatory; without it the answer is invalid.

Categories:
- sports = active physical exercise or gym workout: running, lifting weights,
  playing football/tennis/etc., yoga, cycling. Does NOT include simply wearing
  sportswear while standing or posing.

- romance = two or more people kissing on the lips or embracing in a clearly
  romantic/intimate way (not a friendly hug or casual contact). A couple
  cradling, holding, or leaning into each other while laughing or gazing
  at each other counts as romance.

- posing = person(s) whose PRIMARY action is being photographed. Their body
  is arranged for the camera and there is no other meaningful activity
  happening. Includes: modeling clothes/products, fashion shots, mirror
  selfies, full-body or portrait shots where the person is simply standing
  or sitting in a styled way for the photo, including landscape/scenery
  shots where the person poses in front of a backdrop.
  KEY TEST: if you removed the camera from the scene, would the person
  still be doing something? If NO (they would just be standing there) →
  posing. If YES (eating, walking, talking, etc.) → other.

- other = the person is engaged in a real activity beyond being photographed,
  even if they happen to look at the camera. Includes: eating, drinking,
  cooking, using a phone, chatting/laughing with others, walking through a
  place, working, holding a baby/pet, dancing at an event, singing,
  playing an instrument, traveling (sitting on a plane/train), or any
  everyday action. Also includes images where the activity cannot be
  determined or the person is doing something ambiguous.
  KEY TEST: if there is a real action happening — even a small one like
  drinking coffee or holding something while doing it — and the person is
  not just standing/sitting decoratively → other, regardless of whether
  they look at the camera.

TIE-BREAKER for ambiguous cases:
  - Two people clearly embracing / cradling / holding each other →
    romance (this beats posing even if it looks staged)
  - Standing/sitting still + styled pose + no other action → posing
  - Any active behavior (eating, holding a phone in use, walking,
    interacting with someone or something) → other
  - Holding an object decoratively as a prop (e.g., holding a coffee cup
    just for the photo, not drinking it) → posing
  - Holding an object while using it (e.g., actually drinking the coffee,
    typing on the phone) → other

REMINDER: Output the "Activity: <category>" line FIRST, before any reasoning. Then optionally explain.
"""

    def classify_batch(self, image_crops: list) -> list:
        """
        Clasifica actividades de múltiples personas usando procesamiento batch.

        Args:
            image_crops: Lista de recortes de imagen BGR (OpenCV) de personas

        Returns:
            Lista de dicts con predicción de actividad para cada crop
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"activity": None, "error": "Backend not loaded"}] * len(image_crops)

        if not image_crops:
            return []

        print(f"  🎯 Analizando actividades de {len(image_crops)} persona(s)...")

        try:
            # Convertir todos los crops BGR a PIL RGB
            pil_images = []
            for i, crop in enumerate(image_crops):
                if crop.size > 0:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(image_rgb))
                else:
                    pil_images.append(None)

            # Procesar cada imagen individualmente (VLM no soporta batch nativo)
            results = []
            for i, img in enumerate(pil_images):
                if img is not None:
                    try:
                        result = self._classify_single(img)
                        results.append(result)
                        if result.get('success'):
                            print(f"    ✅ Persona {i+1}: {result.get('activity', 'Unknown')}")
                        else:
                            print(f"    ❌ Persona {i+1}: Error - {result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"    ❌ Error en persona {i+1}: {e}")
                        results.append({
                            "activity": None,
                            "error": str(e),
                            "success": False
                        })
                else:
                    results.append({
                        "activity": None,
                        "error": "Invalid crop",
                        "success": False
                    })

            successful_count = len([r for r in results if r.get('success')])
            print(f"  ✅ Actividades completado: {successful_count} exitosos de {len(results)}")
            return results

        except Exception as e:
            print(f"  ❌ Error general en análisis de actividades: {e}")
            return [{"activity": None, "error": str(e), "success": False}] * len(image_crops)

    def classify(self, image_crop: np.ndarray) -> dict:
        """
        Clasifica la actividad de una persona.

        Args:
            image_crop: Recorte de imagen BGR (OpenCV) de la persona

        Returns:
            Dict con predicción de actividad
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"activity": None, "error": "Backend not loaded"}

        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return self._classify_single(pil_image)

        except Exception as e:
            return {
                "activity": None,
                "error": str(e),
                "success": False
            }

    def _classify_single(self, pil_image: Image.Image) -> dict:
        """Clasifica una sola imagen PIL."""
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            activity = self._parse_response(output_text)

            return {
                "activity": activity,
                "raw_response": output_text.strip(),
                "success": activity is not None
            }

        except Exception as e:
            return {
                "activity": None,
                "error": str(e),
                "success": False
            }

    def _parse_response(self, text: str) -> str:
        """
        Parsea la respuesta del modelo para extraer etiqueta de actividad.
        """
        import re

        OTHER_ALIASES = {"entertaining", "everyday doings", "no activities", "na", "n/a"}

        text_original = text.strip()
        valid_categories = ["sports", "romance", "posing", "other"]

        activity_match = re.search(r'(?:Activity|Label):\s*([^\n]+)', text_original, re.IGNORECASE)
        if activity_match:
            activity_text = activity_match.group(1).strip().lower()
            if activity_text in OTHER_ALIASES:
                return "other"
            for category in valid_categories:
                if category.lower() in activity_text:
                    return category
            return "other"

        return "other"
