"""
Clase base abstracta para backends de modelos de lenguaje visual (VLM).
"""

from PIL import Image


class VLMBackend:
    """
    Interfaz común para todos los backends VLM.
    Cada backend implementa generate() con su propia lógica de inferencia.
    """

    # Backends que soporten batching real en GPU (un solo forward pass para N
    # pares (imagen, prompt)) deben sobrescribir este atributo a True. Los
    # clasificadores lo consultan para decidir entre `generate_batch()` y el
    # ThreadPool tradicional (usado por OllamaBackend, que serializa en el
    # servidor de todos modos).
    supports_real_batch: bool = False

    def generate(self, pil_image: Image.Image, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Genera una respuesta de texto dada una imagen y un prompt.

        Args:
            pil_image: Imagen PIL en formato RGB
            prompt: Texto de instrucción para el modelo
            max_new_tokens: Máximo de tokens a generar

        Returns:
            Respuesta de texto generada por el modelo
        """
        raise NotImplementedError

    def generate_batch(
        self,
        requests: list[tuple["Image.Image", str]],
        max_new_tokens: int = 50,
        task_hint: str | None = None,
    ) -> list[str]:
        """
        Procesa N pares (imagen, prompt) y devuelve N respuestas alineadas.

        Implementación por defecto: loop secuencial sobre `generate()`. Los
        backends que puedan agruparlos en un único forward pass (Transformers,
        vLLM) deben sobrescribirlo para obtener speedup real y poner
        `supports_real_batch = True` a nivel de clase.

        Args:
            requests:  lista de tuplas (imagen PIL, prompt).
            max_new_tokens:  límite común para todas las respuestas.
            task_hint: identificador opcional ("person_attrs", "scene",
                "social_distance", "ocr", "beauty") usado por backends que
                ajustan el visual token budget por tarea.
        """
        return [self.generate(img, prompt, max_new_tokens) for img, prompt in requests]

    def is_loaded(self) -> bool:
        """Devuelve True si el backend está listo para inferencia."""
        return True
