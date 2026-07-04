"""
Clasificador OCR — extrae texto visible en imágenes usando el backend VLM compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


class OCRClassifier:
    """
    Clasificador de OCR (reconocimiento de texto) usando el backend VLM compartido.
    Extrae y reconoce texto visible en la imagen completa.
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        # Prompt para OCR
        self.prompt = """Extract and transcribe ALL visible text from this image.

Instructions:
- Read all text that appears in the image (words, letters, numbers, symbols)
- Include text from signs, captions, overlays, billboards, screens, clothing, etc.
- Preserve the original spelling and capitalization
- List each text element on a new line
- If there is no text, respond with: "No text detected"

Provide ONLY the extracted text. Do not add explanations or descriptions.

Respond in this exact format:
Text: <extracted text or "No text detected">

Example output:
Text: SALE 50% OFF
Open 9AM-5PM"""

    def classify(self, image: np.ndarray) -> dict:
        """
        Realiza OCR en una imagen completa.

        Args:
            image: Imagen BGR completa (OpenCV)

        Returns:
            Dict con texto extraído
        """
        if self.backend is None or not self.backend.is_loaded():
            return {"text": None, "error": "Backend not loaded", "success": False}

        try:
            # Convertir BGR a RGB y luego a PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            return self._classify_single(pil_image)

        except Exception as e:
            return {
                "text": None,
                "error": str(e),
                "raw_response": None,
                "success": False
            }

    def _classify_single(self, pil_image: Image.Image) -> dict:
        """Clasifica una sola imagen PIL."""
        try:
            response = self.backend.generate(pil_image, self.prompt, max_new_tokens=config.VLM_MAX_TOKENS)
            extracted_text = self._parse_response(response)

            return {
                "text": extracted_text,
                "raw_response": response,
                "success": True
            }

        except Exception as e:
            return {
                "text": None,
                "error": str(e),
                "raw_response": None,
                "success": False
            }

    def _parse_response(self, response: str) -> str:
        """
        Parsea la respuesta del modelo para extraer el texto.
        """
        if not response:
            return "No text detected"

        # Buscar patrón "Text: ..."
        import re
        match = re.search(r'Text:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            return text if text else "No text detected"

        # Si no hay patrón, devolver la respuesta completa
        cleaned = response.strip()
        return cleaned if cleaned else "No text detected"
