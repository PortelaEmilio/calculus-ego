"""
Clasificador unificado de edad y género usando el modelo Qwen3-VL compartido.
"""

import cv2
import numpy as np
from PIL import Image

import config


class AgeGenderClassifier:
    """
    Clasificador unificado de edad y género usando el backend VLM compartido.

    Edad: grupo de edad estimado (childhood, youth, adulthood, old age)
    Género: male / female
    """

    def __init__(self, backend=None):
        """
        Args:
            backend: Instancia de VLMBackend ya cargada
        """
        self.backend = backend

        # Prompt para clasificación de género
        self.gender_prompt = """Classify the gender of the person in the image.

Look at the person's physical appearance and classify as one of:

male = the person appears to be male
female = the person appears to be female
na = the person's gender cannot be clearly determined (face not visible, ambiguous appearance, etc.)

Respond in this exact format:
Gender: <male, female, or na>

Example output:
Gender: male"""

        # Prompt para clasificación de edad
        self.age_prompt = """Estimate the age group of the person in the image.

Look at the person's physical appearance (face, body, skin, hair) and classify into ONE age group:

childhood = baby, toddler, or child (approximately 0-12 years old)
youth = teenager or young adult (approximately 13-29 years old) — INCLUDES anyone in their 20s
adulthood = adult (approximately 30-59 years old) — requires clear signs of aging: visible wrinkles, grey hair, mature/sagging facial structure, receding hairline
old age = senior or elderly (approximately 60+ years old)
na = the person's age cannot be clearly determined (face not visible, body too obscured, etc.)

Important:
- When unsure between youth and adulthood, ALWAYS prefer youth. Adulthood requires CLEAR aging signs ON THE SKIN ITSELF: visible wrinkles, grey or white hair, receding hairline / balding, sagging facial structure.
- Facial hair (beard, moustache, stubble) is NOT an aging cue — many men in their late teens and 20s have full thick beards. A dense beard, intense gaze, muscular build, or strong jawline alone DO NOT indicate adulthood. Do not let these features override the default to youth.
- For cartoons, anime-style avatars, or stylised illustrations, judge by the apparent age the artist depicts. Smooth skin, no wrinkles, no grey hair → youth, even if the character has a beard or looks intimidating. A "powerful adult man" archetype in art is usually drawn as a young adult; classify as youth unless explicit aging cues are drawn in.

Respond in this exact format:
Age: <age group or na>

Example outputs:
Age: youth

Age: adulthood"""

    def classify_batch(self, image_crops: list) -> tuple[list, list]:
        """
        Clasifica edad y género de múltiples personas usando el modelo Qwen compartido.

        Args:
            image_crops: Lista de recortes de imagen BGR (OpenCV) de personas

        Returns:
            Tupla de (gender_results, age_results) con formato compatible
        """
        if self.backend is None or not self.backend.is_loaded():
            empty_gender = {"gender": None, "error": "Backend not loaded", "success": False}
            empty_age = {"age_group": None, "error": "Backend not loaded", "success": False}
            return (
                [empty_gender] * len(image_crops),
                [empty_age] * len(image_crops)
            )

        if not image_crops:
            return [], []

        print(f"  👤🎂 Analizando edad y género de {len(image_crops)} persona(s)...")

        try:
            # Convertir todos los crops BGR a PIL RGB
            pil_images = []
            for i, crop in enumerate(image_crops):
                if crop.size > 0:
                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(image_rgb))
                else:
                    pil_images.append(None)

            gender_results = [None] * len(image_crops)
            age_results = [None] * len(image_crops)

            for i, img in enumerate(pil_images):
                if img is not None:
                    try:
                        gender_result = self._classify_single(img, 'gender')
                        if gender_result.get('success'):
                            gender_results[i] = {
                                "gender": gender_result["gender"],
                                "all_predictions": [{"label": gender_result["gender"], "score": None}],
                                "raw_response": gender_result.get("raw_response"),
                                "success": True
                            }
                        else:
                            gender_results[i] = {"gender": None, "error": gender_result.get("error"), "success": False}

                        age_result = self._classify_single(img, 'age')
                        if age_result.get('success'):
                            age_results[i] = {
                                "age_group": age_result["age_group"],
                                "raw_response": age_result.get("raw_response"),
                                "success": True
                            }
                        else:
                            age_results[i] = {"age_group": None, "error": age_result.get("error"), "success": False}

                        g_label = gender_result.get('gender', '?') if gender_result.get('success') else '?'
                        a_label = age_result.get('age_group', '?') if age_result.get('success') else '?'
                        print(f"    ✅ Persona {i+1}: {g_label}, {a_label}")
                    except Exception as e:
                        print(f"    ❌ Error en persona {i+1}: {e}")
                        gender_results[i] = {"gender": None, "error": str(e), "success": False}
                        age_results[i] = {"age_group": None, "error": str(e), "success": False}
                else:
                    gender_results[i] = {"gender": None, "error": "Invalid crop", "success": False}
                    age_results[i] = {"age_group": None, "error": "Invalid crop", "success": False}

            successful_count = len([r for r in gender_results if r and r.get('success')])
            print(f"  ✅ Edad y género completado: {successful_count} exitosos de {len(image_crops)}")
            return gender_results, age_results

        except Exception as e:
            print(f"  ❌ Error general en análisis de edad y género: {e}")
            error_gender = {"gender": None, "error": str(e), "success": False}
            error_age = {"age_group": None, "error": str(e), "success": False}
            return (
                [error_gender] * len(image_crops),
                [error_age] * len(image_crops)
            )

    def _classify_single(self, pil_image: Image.Image, task: str) -> dict:
        """
        Clasifica una sola imagen PIL para género o edad.

        Args:
            pil_image: Imagen PIL en RGB
            task: 'gender' o 'age'
        """
        try:
            prompt = self.gender_prompt if task == 'gender' else self.age_prompt
            output_text = self.backend.generate(pil_image, prompt, max_new_tokens=config.VLM_MAX_TOKENS)

            if task == 'gender':
                gender = self._parse_gender_response(output_text)
                return {
                    "gender": gender,
                    "raw_response": output_text.strip(),
                    "success": gender is not None
                }
            else:
                age_group = self._parse_age_response(output_text)
                return {
                    "age_group": age_group,
                    "raw_response": output_text.strip(),
                    "success": age_group is not None
                }

        except Exception as e:
            if task == 'gender':
                return {"gender": None, "error": str(e), "success": False}
            else:
                return {"age_group": None, "error": str(e), "success": False}

    def _parse_gender_response(self, text: str) -> str | None:
        """
        Parsea la respuesta del modelo para extraer género.
        """
        import re
        text = text.strip()
        text_lower = text.lower()

        gender_match = re.search(r'Gender:\s*([^\n]+)', text, re.IGNORECASE)
        if gender_match:
            gender_text = gender_match.group(1).strip().lower()
            if gender_text in ('na', 'n/a'):
                return 'no visible'
            if 'male' in gender_text and 'female' not in gender_text:
                return 'male'
            if 'female' in gender_text:
                return 'female'

        if text_lower.strip() in ('na', 'n/a') or 'n/a' in text_lower:
            return 'no visible'
        if 'female' in text_lower:
            return 'female'
        if 'male' in text_lower:
            return 'male'
        return None

    def _parse_age_response(self, text: str) -> str | None:
        """
        Parsea la respuesta del modelo para extraer grupo de edad.
        """
        import re
        text = text.strip()
        text_lower = text.lower()

        valid_age_groups = [
            "childhood", "youth", "adulthood", "old age"
        ]

        age_match = re.search(r'Age:\s*([^\n]+)', text, re.IGNORECASE)
        if age_match:
            age_text = age_match.group(1).strip().lower()
            if age_text in ('na', 'n/a'):
                return 'no visible'
            for group in valid_age_groups:
                if group in age_text:
                    return group

        for group in valid_age_groups:
            if group in text_lower:
                return group

        if 'child' in text_lower or 'baby' in text_lower or 'toddler' in text_lower or 'kid' in text_lower:
            return 'childhood'
        if 'teenager' in text_lower or 'adolescent' in text_lower or 'young adult' in text_lower or 'young' in text_lower:
            return 'youth'
        if 'adult' in text_lower or 'middle' in text_lower:
            return 'adulthood'
        if 'senior' in text_lower or 'elderly' in text_lower or 'old' in text_lower:
            return 'old age'

        number_match = re.search(r'(\d{1,3})', text)
        if number_match:
            age_num = int(number_match.group(1))
            if age_num <= 100:
                return self._age_to_group(age_num)

        return None

    @staticmethod
    def _age_to_group(age: int) -> str:
        """
        Convierte edad numérica a grupo de edad.
        """
        if age < 13:
            return "childhood"
        elif age < 30:
            return "youth"
        elif age < 60:
            return "adulthood"
        else:
            return "old age"
