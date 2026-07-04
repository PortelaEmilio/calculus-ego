"""
Clasificador de comportamiento basado en Visual Social Semiotics.
"""

import cv2
import numpy as np
from PIL import Image

import config


class BehaviourClassifier:
    """
    Clasificador de comportamiento basado en Visual Social Semiotics.

    Categorías:
    - demand/affiliation: Eye level, mirada directa, expresión neutral/amigable
    - demand/seduction: Mirada hacia arriba, cabeza inclinada, expresión seductora
    - demand/submission: Mirada hacia abajo desde posición superior, expresión seria
    - offer/ideal: No engagement directo, mirada indirecta, presentación impersonal
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada (TransformersBackend u OllamaBackend)
        """
        self.backend = backend

        self.prompt = """Classify the highlighted person (yellow box labeled TARGET) into ONE category based on eye gaze,
camera angle, and facial expression, following visual social semiotics principles.

CRITICAL RULE — Burden of proof for `demand`:
The categories `demand/*` require BOTH:
  (a) the HEAD facing the camera frontally (NOT in 3/4 profile, NOT in side
      profile, NOT tilted away from the viewer); AND
  (b) BOTH pupils (or, for illustrations, both drawn eyes) clearly centered
      on the camera lens / viewer.

If EITHER (a) or (b) fails — eyes slightly turned, looking off-axis, looking
past the viewer, head turned to the side even slightly, head in 3/4 view, or
gaze ambiguous — classify as `offer/ideal`. Do NOT use `demand/seduction`
just because the expression looks intense or the lips are parted: those
signals only matter AFTER both head and gaze are confirmed frontal.

Stylized illustrations, cartoons, anime characters, and digital avatars are
evaluated by the SAME rule: judge head orientation and drawn eye direction.
A cartoon whose head is in 3/4 profile (one ear visible, the other hidden;
the nose pointing slightly off-camera; jawline asymmetric) is OFFER, even if
the eyes appear to look toward the viewer — the head orientation is
disqualifying. Only frontal head + frontal pupils → demand.

Decision tree (apply in order):

STEP 1 — Head orientation: Is the head facing the camera FRONTALLY (face
visible head-on, both ears symmetric or both equally hidden, nose pointing
straight at the lens)?
  - If NO (head in 3/4 profile, side profile, looking away, looking up/down
    significantly, tilted) → offer/ideal. Stop here.
  - If YES → continue to Step 2.

STEP 2 — Eye contact: Are BOTH pupils (or drawn eyes, for illustrations)
CLEARLY and DIRECTLY centered on the camera lens / viewer (not slightly
off-axis, not past the lens)?
  - If NO or UNCERTAIN → offer/ideal. Stop here.
  - If YES (high confidence direct gaze, drawn or photographic) → continue
    to Step 3.

STEP 3 — Expression and pose: Does the person show deliberate seductive signals?
Seductive signals include ANY of:
    · parted lips or pouting
    · direct smoldering/intense gaze (not a warm smile)
    · head tilted AND a suggestive or provocative expression
    · body language clearly oriented toward the viewer in a flirtatious way
  - If ONE OR MORE seductive signals are present → demand/seduction. Stop here.
  - If NONE (neutral, friendly smile, relaxed expression) → continue to Step 4.

STEP 4 — Camera angle: Is the camera positioned CLEARLY ABOVE the person,
shooting downward so the person appears to look UP at the viewer?
  - If YES → demand/submission.
  - If NO or uncertain → demand/affiliation.

Categories (summary):
- offer/ideal     = gaze NOT directed at viewer, ambiguous, or stylized
- demand/seduction  = direct gaze + seductive expression/pose (priority over camera angle)
- demand/submission = direct gaze + no seductive cues + camera clearly above person
- demand/affiliation = direct gaze + no seductive cues + camera at or below eye level

Response format:
"Behaviour: <category>"
"Key signals: <brief list of observed cues that drove the decision>"
"""

    def classify_batch(self, image_crops: list) -> list:
        """
        Clasifica el comportamiento de múltiples personas.

        Args:
            image_crops: Lista de recortes de imagen BGR (OpenCV) de personas

        Returns:
            Lista de dicts con predicción de comportamiento para cada crop
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"behaviour": None, "error": "Backend not loaded"}] * len(image_crops)

        if not image_crops:
            return []

        print(f"  🎭 Analizando comportamiento de {len(image_crops)} persona(s)...")

        try:
            pil_images = []
            for crop in image_crops:
                if crop.size > 0:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(image_rgb))
                else:
                    pil_images.append(None)

            results = []
            for i, img in enumerate(pil_images):
                if img is not None:
                    try:
                        result = self._classify_single(img)
                        results.append(result)
                        if result.get('success'):
                            print(f"    ✅ Persona {i+1}: {result.get('behaviour', 'Unknown')}")
                        else:
                            print(f"    ❌ Persona {i+1}: Error - {result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"    ❌ Error en persona {i+1}: {e}")
                        results.append({
                            "behaviour": None,
                            "error": str(e),
                            "success": False
                        })
                else:
                    results.append({
                        "behaviour": None,
                        "error": "Invalid crop",
                        "success": False
                    })

            successful_count = len([r for r in results if r.get('success')])
            print(f"  ✅ Comportamiento completado: {successful_count} exitosos de {len(results)}")
            return results

        except Exception as e:
            print(f"  ❌ Error general en análisis de comportamiento: {e}")
            return [{"behaviour": None, "error": str(e), "success": False}] * len(image_crops)

    def classify(self, image_crop: np.ndarray) -> dict:
        """
        Clasifica el comportamiento de una persona.

        Args:
            image_crop: Recorte de imagen BGR (OpenCV) de la persona

        Returns:
            Dict con predicción de comportamiento y explicación
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"behaviour": None, "error": "Backend not loaded"}

        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return self._classify_single(pil_image)

        except Exception as e:
            return {
                "behaviour": None,
                "error": str(e),
                "success": False
            }

    def _classify_single(self, pil_image: Image.Image) -> dict:
        """Clasifica una sola imagen PIL."""
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            behaviour = self._parse_response(output_text)

            return {
                "behaviour": behaviour,
                "raw_response": output_text.strip(),
                "success": behaviour is not None
            }

        except Exception as e:
            return {
                "behaviour": None,
                "error": str(e),
                "success": False
            }

    def _parse_response(self, text: str) -> str | None:
        """
        Parsea la respuesta del modelo para extraer etiqueta.
        """
        import re

        text = text.strip()

        valid_categories = [
            "demand/affiliation",
            "demand/seduction",
            "demand/submission",
            "offer/ideal"
        ]

        behaviour_match = re.search(r'Behaviour:\s*([^\n]+)', text, re.IGNORECASE)
        if behaviour_match:
            behaviour_text = behaviour_match.group(1).strip().lower()
            if behaviour_text in ('na', 'n/a'):
                return 'no visible'
            for category in valid_categories:
                if category.lower() in behaviour_text:
                    return category

        text_lower = text.lower()
        for category in valid_categories:
            if category.lower() in text_lower:
                return category

        if "demand" in text_lower:
            if "affiliation" in text_lower or "friendly" in text_lower:
                return "demand/affiliation"
            if "seduction" in text_lower or "seductive" in text_lower:
                return "demand/seduction"
            if "submission" in text_lower or "dominant" in text_lower:
                return "demand/submission"
        if "offer" in text_lower or "ideal" in text_lower:
            return "offer/ideal"

        return None
