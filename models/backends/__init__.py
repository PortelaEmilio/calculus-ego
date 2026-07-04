"""
VLM backend factory.

This build ships a single classification backend: Qwen3.5-9B served via
Transformers (4-bit NF4, eager attention) through ``Qwen35VLMBackend``. The
factory keeps the ``create_backend("transformers", ...)`` entry point so the
rest of the pipeline is untouched.
"""

from .base import VLMBackend


def create_backend(backend_type: str, **kwargs) -> VLMBackend:
    """
    Create and load the VLM backend.

    Only ``backend_type="transformers"`` is supported. When the model name
    contains ``"qwen3.5"`` it routes to ``Qwen35VLMBackend`` (own loader: eager
    attention, ``trust_remote_code``, thinking off).

    Args:
        backend_type: must be "transformers".
        **kwargs: ``model_name`` (e.g. "Qwen/Qwen3.5-9B").

    Returns:
        A loaded ``VLMBackend`` instance.
    """
    if backend_type == "transformers":
        model_name = kwargs["model_name"]
        if "qwen3.5" in str(model_name).lower():
            from models.qwen35_vlm_backend import Qwen35VLMBackend
            backend = Qwen35VLMBackend(model_name=model_name)
            backend.load()
            return backend
        raise ValueError(
            f"This build only ships the Qwen3.5 classifier; got model '{model_name}'. "
            "Set BEHAVIOUR_MODEL_NAME to a Qwen3.5 checkpoint."
        )

    raise ValueError(
        f"Unsupported backend: '{backend_type}'. This build only supports 'transformers'."
    )


__all__ = ["VLMBackend", "create_backend"]
