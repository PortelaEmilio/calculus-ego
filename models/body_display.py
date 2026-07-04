"""
Clasificador de exposición del cuerpo usando el modelo Qwen3-VL compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


class BodyDisplayClassifier:
    """
    Clasificador de exposición del cuerpo usando el backend VLM compartido.

    Categorías:
    - revealing clothes: underwear, shirt unbuttoned/open, swimming trunks, shorts, towel around body
    - partially naked: missing upper body OR lower body clothing, but not both
    - no clothes at all: completely nude
    - normal clothes: regular everyday clothing fully covering the body
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        self.prompt = """Classify the body display of the highlighted person (yellow box labeled TARGET) into ONE label.

Categories:
- normal clothes = everyday clothing that covers both upper and lower body
  without significant skin exposure: t-shirts (including short sleeves),
  shirts, blouses, sweaters, jackets, trousers, jeans, skirts of normal
  length. Includes sportswear (leggings, sports bras worn as outerwear) if
  no large skin areas are exposed.
  IMPORTANT: short-sleeve t-shirts ARE normal clothes. Only classify upper
  garments as revealing if they are sleeveless, strapless, have spaghetti
  straps, or show a deep neckline / bare midriff.
  DEFAULT RULE: if clothing is not clearly visible (image cropped, body
  not shown, only head/face visible, or unclear), classify as normal clothes.
  There is no NA option — when in doubt, default here.

- revealing clothes = clothing that covers the body but deliberately exposes
  significant skin: tank tops, sleeveless tops, muscle shirts, vests,
  spaghetti straps, strapless or open-back tops, bikini tops, deep necklines,
  crop tops showing the midriff, mini-skirts, shorts exposing most of the
  thigh, tight or see-through fabrics. Any garment that leaves the shoulders
  and/or arms bare counts as revealing, regardless of how the rest of the
  body is covered.
  KEY RULE: if ANY fabric, strap, or piece of clothing covers a body area —
  even minimally (e.g. a single strap across the chest, a crossed cloth, a
  thin piece of fabric over the shoulders or torso) — that area counts as
  COVERED, and the image is revealing clothes, NOT partially naked. Partial
  coverage with revealing garments is still revealing clothes.

- partially naked = one or more body areas are COMPLETELY bare with NO
  garment, fabric, strap, or covering of any kind on them: bare chest/torso
  with nothing across it, fully shirtless, bare legs/lower body with no
  shorts/skirt/underwear visible. If even a thin strap or small piece of
  cloth crosses the area, it is revealing clothes instead.
  If nudity is complete but the image is cropped above the knees, use this
  category instead of "no clothes at all".

- no clothes at all = completely nude with no clothing visible anywhere AND
  the body is visible at least down to the knees, confirming full nudity
  rather than a cropped partial view.

Response format: "Body Display: <one label>"
"""

    def classify_batch(self, image_crops: list) -> list:
        """
        Clasifica la exposición del cuerpo de múltiples personas usando procesamiento batch.

        Args:
            image_crops: Lista de recortes de imagen BGR (OpenCV) de personas

        Returns:
            Lista de dicts con predicción de exposición del cuerpo para cada crop
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"body_display": None, "error": "Backend not loaded"}] * len(image_crops)

        if not image_crops:
            return []

        print(f"  👔 Analizando exposición del cuerpo de {len(image_crops)} persona(s)...")

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
                            print(f"    ✅ Persona {i+1}: {result.get('body_display', 'Unknown')}")
                        else:
                            print(f"    ❌ Persona {i+1}: Error - {result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"    ❌ Error en persona {i+1}: {e}")
                        results.append({
                            "body_display": None,
                            "error": str(e),
                            "success": False
                        })
                else:
                    results.append({
                        "body_display": None,
                        "error": "Invalid crop",
                        "success": False
                    })

            successful_count = len([r for r in results if r.get('success')])
            print(f"  ✅ Exposición del cuerpo completado: {successful_count} exitosos de {len(results)}")
            return results

        except Exception as e:
            print(f"  ❌ Error general en análisis de exposición del cuerpo: {e}")
            return [{"body_display": None, "error": str(e), "success": False}] * len(image_crops)

    def classify(self, image_crop: np.ndarray) -> dict:
        """
        Clasifica la exposición del cuerpo de una persona.

        Args:
            image_crop: Recorte de imagen BGR (OpenCV) de la persona

        Returns:
            Dict con predicción de exposición del cuerpo
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"body_display": None, "error": "Backend not loaded"}

        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return self._classify_single(pil_image)

        except Exception as e:
            return {
                "body_display": None,
                "error": str(e),
                "success": False
            }

    def _classify_single(self, pil_image: Image.Image) -> dict:
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            body_display = self._parse_response(output_text)

            return {
                "body_display": body_display,
                "raw_response": output_text.strip(),
                "success": body_display is not None
            }

        except Exception as e:
            return {
                "body_display": None,
                "error": str(e),
                "success": False
            }

    def _parse_response(self, text: str) -> str | None:
        """
        Parsea la respuesta del modelo para extraer etiqueta de exposición del cuerpo.
        """
        import re

        text_original = text.strip()
        text_lower = text_original.lower()

        valid_categories = [
            "normal clothes",
            "revealing clothes",
            "partially naked",
            "no clothes at all",
        ]

        body_display_match = re.search(r'Body Display:\s*([^\n]+)', text_original, re.IGNORECASE)
        if body_display_match:
            body_display_text = body_display_match.group(1).strip().lower()
            if body_display_text in ('na', 'n/a'):
                return 'no visible'
            for category in valid_categories:
                if category in body_display_text:
                    return category

        for category in valid_categories:
            if category in text_lower:
                return category

        if "partially" in text_lower and "naked" in text_lower:
            return "partially naked"
        if "no clothes at all" in text_lower or "nude" in text_lower or ("naked" in text_lower and "partially" not in text_lower):
            return "no clothes at all"
        if "revealing" in text_lower or "straps" in text_lower or "bikini" in text_lower or "swimwear" in text_lower:
            return "revealing clothes"
        if "no shirt" in text_lower or "no top" in text_lower or "no pants" in text_lower or "shirtless" in text_lower or "bare chest" in text_lower:
            return "partially naked"
        if "normal" in text_lower or "regular" in text_lower or "covered" in text_lower or "everyday" in text_lower:
            return "normal clothes"

        return None
