"""
Clasificador de ubicación usando el modelo Qwen3-VL compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


class LocationClassifier:
    """
    Clasificador de ubicación usando el backend VLM compartido.

    Categorías:
    - indoors: inside a building (apartment, bedroom, office, classroom, nightclub, etc.)
    - wilderness: natural outdoor setting (woods, park, garden, lake/sea/river, etc.)
    - city: urban outdoor setting (street, buildings, famous landmarks, etc.)
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        self.prompt = """Classify the location of the highlighted person (yellow box labeled TARGET) in the image.

Follow these steps IN ORDER. STOP at the first matching rule.

STEP 1 — is the yellow box labeled TARGET around a profile-picture avatar?
   Look at the yellow box labeled TARGET. Is it drawn around a small circular or rounded profile-picture thumbnail (typical of tweets, Instagram, TikTok, YouTube overlays — usually a small circle next to a username/handle, often in a corner of the frame)?
   - If YES → classify ONLY by what is INSIDE that circular thumbnail. Ignore the rest of the image entirely, including any larger photo or scene visible elsewhere in the frame.
     * If the avatar's own background inside the circle is a solid color, dark/black backdrop, gradient, stylized illustration, or any non-scenic backdrop → "no background". STOP.
     * If the avatar's own background inside the circle clearly shows an indoor room, wilderness, or city scene → use that category. STOP.
   - If NO (the yellow box labeled TARGET is around a regular person, not a tiny avatar thumbnail) → continue to STEP 2.

STEP 2 — classify by the background immediately around the highlighted person:

indoors = inside a building (apartment, bedroom, office, classroom, nightclub, etc.)
wilderness = natural outdoor setting (woods, park, garden, lake/sea/river, sky with clouds, mountains, beach, etc.)
city = urban outdoor setting (street, buildings, famous landmarks, etc.)
no background = solid color background (white, grey, black, red, any flat color), plain studio/photography backdrop, gradient, seamless blur with no identifiable location. A white, red, grey, or beige wall visible immediately behind the person with NO other room context (furniture, windows, decor) counts as "no background", NOT indoors. Only classify as indoors if you can clearly identify that the person is inside a room or building.
NA = the location cannot be determined (image too blurry or incomprehensible)

CRITICAL RULES:
- An avatar's location is determined ONLY by what is inside its circle. NEVER inherit the background from a larger photo elsewhere in the same image.
- If the same image contains both a tweet/post avatar AND a separate large photo, those are TWO different scenes. Each highlighted person is classified by their own immediate surroundings.

Respond in this exact format:
Location: <one label>

Valid labels: indoors, wilderness, city, no background, NA

Example outputs:
Location: no background

Location: indoors"""

    def classify_batch(self, image_crops: list) -> list:
        """
        Clasifica la ubicación de múltiples personas usando procesamiento batch.

        Args:
            image_crops: Lista de recortes de imagen BGR (OpenCV) de personas

        Returns:
            Lista de dicts con predicción de ubicación para cada crop
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"location": None, "error": "Backend not loaded"}] * len(image_crops)

        if not image_crops:
            return []

        print(f"  🏠 Analizando ubicación de {len(image_crops)} persona(s)...")

        try:
            # Convertir todos los crops BGR a PIL RGB
            pil_images = []
            for crop in image_crops:
                if crop.size > 0:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(image_rgb))
                else:
                    pil_images.append(None)

            # Procesar cada imagen
            results = []
            for i, pil_image in enumerate(pil_images, 1):
                if pil_image is not None:
                    result = self._classify_single(pil_image)

                    if result.get('success'):
                        print(f"    ✅ Persona {i}: {result.get('location', 'Unknown')}")
                    else:
                        print(f"    ❌ Persona {i}: Error")

                    results.append(result)
                else:
                    results.append({
                        "location": None,
                        "error": "Invalid crop",
                        "success": False
                    })

            print(f"  ✅ Ubicación completado: {sum(1 for r in results if r.get('success'))} exitosos de {len(results)}")
            return results

        except Exception as e:
            print(f"  ❌ Error en análisis de ubicación: {e}")
            return [{"location": None, "error": str(e), "success": False}] * len(image_crops)

    def classify(self, image_crop: np.ndarray) -> dict:
        """
        Clasifica la ubicación de una persona.

        Args:
            image_crop: Recorte de imagen BGR (OpenCV) de la persona

        Returns:
            Dict con predicción de ubicación
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"location": None, "error": "Backend not loaded"}

        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return self._classify_single(pil_image)

        except Exception as e:
            return {
                "location": None,
                "error": str(e),
                "success": False
            }

    def _classify_single(self, pil_image: Image.Image) -> dict:
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            location = self._parse_response(output_text)

            return {
                "location": location,
                "raw_response": output_text.strip(),
                "success": location is not None
            }

        except Exception as e:
            return {
                "location": None,
                "error": str(e),
                "success": False
            }

    def _parse_response(self, text: str) -> str:
        """
        Parsea la respuesta del modelo para extraer etiqueta de ubicación.
        """
        import re

        text_original = text.strip()
        text_lower = text_original.lower()

        valid_categories = [
            "indoors",
            "wilderness",
            "city",
            "no background",
        ]

        location_match = re.search(r'Location:\s*([^\n]+)', text_original, re.IGNORECASE)
        if location_match:
            location_text = location_match.group(1).strip().lower()
            if location_text in ('na', 'n/a'):
                return 'no visible'
            for category in valid_categories:
                if category in location_text:
                    return category

        for category in valid_categories:
            if category in text_lower:
                return category

        if "no background" in text_lower or "solid color" in text_lower or "plain" in text_lower or "studio" in text_lower or "gradient" in text_lower:
            return "no background"
        if "indoor" in text_lower or "inside" in text_lower or "building" in text_lower or "apartment" in text_lower or "bedroom" in text_lower:
            return "indoors"
        if "wilderness" in text_lower or "natural" in text_lower or "park" in text_lower or "garden" in text_lower or "woods" in text_lower or "lake" in text_lower or "sea" in text_lower or "river" in text_lower:
            return "wilderness"
        if "city" in text_lower or "urban" in text_lower or "street" in text_lower or "buildings" in text_lower or "landmark" in text_lower:
            return "city"

        return 'no visible'
