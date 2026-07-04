"""
Clasificador de accesorios visibles (binario por categoría) usando el modelo Qwen3-VL compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


ACCESSORY_CATEGORIES = ["makeup", "tattoos", "bags", "belts", "jewelry", "headwear", "eyewear"]


class AccessoryClassifier:
    """
    Clasificador de accesorios visibles usando el backend VLM compartido.
    Devuelve un diccionario binario (0/1) por categoría de accesorio.
    NA solo cuando no hay persona detectada (se asigna desde el pipeline, no desde el modelo).
    """

    def __init__(self, backend=None):
        self.backend = backend

        self.prompt = """Analyze the visible accessories of the highlighted person (yellow box labeled TARGET) in the image.

For each accessory category, answer 1 if visible on the person, or 0 if not visible.

Categories:
makeup = visible cosmetics on the face (lipstick, eyeliner, eyeshadow) or painted nails
tattoos = visible ink or body art on any exposed skin — must be an identifiable
          design (letters, words, symbols, figures, pictures, geometric patterns).
          For illustrations / cartoons / avatars: do NOT count shading lines,
          muscle-definition contours, anatomical line art, or stylistic shadows
          as tattoos. EXAMPLES of NOT-tattoos in cartoons: dark contour lines
          around pectorals or shoulders, shadows from dramatic lighting,
          red/orange backlighting that creates dark patches on skin, line-art
          used to define muscles or anatomy. A real tattoo must be a clearly
          bounded ink design (letter, symbol, figure) that contrasts with the
          surrounding skin tone — not just darker shading.
bags = handbags, backpacks, clutches, or any item carried to hold belongings.
          Requires the actual bag/pouch/container to be VISIBLE in the frame.
          Do NOT count a strap alone (a shoulder strap, lanyard, or string
          across the body without the bag part visible) as a bag — straps
          could belong to clothing, raincoats, halter tops, or backpacks
          worn out of frame. Only mark 1 if you can see the bag's body
          (the part that holds belongings).
belts = any decorative or functional strap worn around the waist
jewelry = earrings, necklaces, bracelets, rings, or visible piercings — for
          illustrations, do NOT count clothing details (collars, straps,
          neckline trim) as jewelry; require an identifiable jewelry item
          distinct from the garment.
headwear = hats, caps, beanies, or headbands
eyewear = glasses, sunglasses, or goggles

Respond in this exact format:
makeup: <0 or 1>
tattoos: <0 or 1>
bags: <0 or 1>
belts: <0 or 1>
jewelry: <0 or 1>
headwear: <0 or 1>
eyewear: <0 or 1>

Example output:
makeup: 1
tattoos: 0
bags: 0
belts: 0
jewelry: 1
headwear: 0
eyewear: 0"""

    def classify_batch(self, image_crops: list) -> list:
        """
        Clasifica accesorios de múltiples personas.

        Args:
            image_crops: Lista de imágenes BGR completas con la persona marcada en rojo

        Returns:
            Lista de dicts con valores binarios por categoría de accesorio
        """
        if self.backend is None or not self.backend.is_loaded():
            return [{"error": "Backend not loaded", "success": False}] * len(image_crops)

        if not image_crops:
            return []

        print(f"  👜 Analizando accesorios de {len(image_crops)} persona(s)...")

        try:
            pil_images = []
            for crop in image_crops:
                if crop.size > 0:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(image_rgb))
                else:
                    pil_images.append(None)

            results = []
            for i, pil_image in enumerate(pil_images, 1):
                if pil_image is not None:
                    try:
                        result = self._classify_single(pil_image)
                        results.append(result)
                        if result.get('success'):
                            found = [k for k in ACCESSORY_CATEGORIES if result.get(k) == 1]
                            print(f"    ✅ Persona {i}: {', '.join(found) if found else 'ninguno'}")
                        else:
                            print(f"    ❌ Persona {i}: Error - {result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"    ❌ Error en persona {i}: {e}")
                        results.append({"error": str(e), "success": False})
                else:
                    results.append({"error": "Invalid crop", "success": False})

            print(f"  ✅ Accesorios completado: {sum(1 for r in results if r.get('success'))} exitosos de {len(results)}")
            return results

        except Exception as e:
            print(f"  ❌ Error en análisis de accesorios: {e}")
            return [{"error": str(e), "success": False}] * len(image_crops)

    def classify(self, image_crop: np.ndarray) -> dict:
        if self.backend is None or not self.backend.is_loaded():
            return {"error": "Backend not loaded", "success": False}

        try:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return self._classify_single(pil_image)
        except Exception as e:
            return {"error": str(e), "success": False}

    def _classify_single(self, pil_image: Image.Image) -> dict:
        """Clasifica una sola imagen PIL."""
        try:
            output_text = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            binary_values = self._parse_response(output_text)

            result = dict(binary_values)
            result["raw_response"] = output_text.strip()
            result["success"] = True
            return result

        except Exception as e:
            return {"error": str(e), "success": False}

    def _parse_response(self, text: str) -> dict:
        """
        Parsea la respuesta del modelo para extraer valores binarios por categoría.
        """
        import re

        text = text.strip()
        binary_values = {cat: 0 for cat in ACCESSORY_CATEGORIES}

        for cat in ACCESSORY_CATEGORIES:
            pattern = rf'{cat}:\s*([01])'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                binary_values[cat] = int(match.group(1))
            else:
                if re.search(rf'\b{cat}\b', text, re.IGNORECASE):
                    binary_values[cat] = 1

        return binary_values
