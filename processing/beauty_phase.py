"""Pase de belleza DIFERIDO compartido por main.py (single/manual) y batch.py.

Tras clasificar todas las imágenes con gemma4 (:11434, Ollama 0.23.2), este módulo:
  1. hace `ollama stop gemma4:e4b` → libera ~10 GB de VRAM en :11434,
  2. carga el estimador Qwen3-VL-4B en la instancia Ollama ≥0.30 de :11435,
  3. puntúa belleza SOLO de los rostros demand/* (IoU>=0.5 con un bbox behaviour=demand
     de la fase 1, leído de los summary_*.json del run).

Reutiliza la lógica de `beauty_pass.py` (DRY): `load_phase1_demand`,
`build_items_from_phase1`, `score_demand_beauty`. Degrada con elegancia: si el pase está
desactivado (`config.ENABLE_BEAUTY_PASS=False` / `--no-beauty`), no hay personas demand/*,
o el servidor de belleza :11435 no responde → devuelve (None, None) sin abortar el run.

Módulo neutro (no importa main) para evitar el ciclo main ↔ batch.
"""

import subprocess

import config
from ui import header, info, success, warn
from beauty_pass import (
    load_phase1_demand,
    build_items_from_phase1,
    score_demand_beauty,
)


def _stop_pipeline_model():
    """`ollama stop <gemma4>` en la instancia del pipeline (:11434) para liberar VRAM.

    El CLI `ollama` por defecto apunta a :11434 (0.23.2). No abortar si falla: puede que
    el modelo ya esté descargado o que el swap lo gestione el propio servidor de belleza.
    """
    tag = getattr(config, "OLLAMA_MODEL", "gemma4:e4b")
    try:
        subprocess.run(["ollama", "stop", tag], check=False,
                       capture_output=True, timeout=30)
        info(f"  ↯ `ollama stop {tag}` (libera VRAM en :11434)")
    except Exception as e:
        warn(f"  no se pudo parar {tag} ({e}); continúo igualmente")


def shared_beauty_backend_from_models(models):
    """Devuelve un `SharedQwen35BeautyBackend` si el clasificador ya cargado trae el LoRA de
    belleza adjunto (base COMPARTIDO), o None si no aplica. Desenvuelve el JsonFlatteningBackend.

    Permite al pase diferido reutilizar el Qwen3.5-9B ya en VRAM (togglando el adapter) en vez
    de cargar una 2ª copia del modelo. Falla-suave: cualquier problema → None (pase carga su
    propio backend como antes)."""
    if not models:
        return None
    try:
        clf = models.get("behaviour_classifier")
        inner = getattr(clf, "backend", None)
        inner = getattr(inner, "inner", inner)  # desenvolver JsonFlatteningBackend
        if inner is not None and getattr(inner, "_has_beauty_adapter", False):
            from models.beauty_backend_qwen35 import SharedQwen35BeautyBackend
            return SharedQwen35BeautyBackend(inner)
    except Exception:
        pass
    return None


def run_beauty_phase(json_dir, model_pose, only_stem=None, shared_backend=None):
    """Ejecuta el pase de belleza diferido sobre los summary_*.json de `json_dir`.

    Args:
        json_dir: dir con summary_*.json de la fase 1 (OUTPUT_DIR o runs/<id>/json).
        model_pose: modelo YOLO pose ya cargado (se reutiliza, no se recarga).
        only_stem: si se indica, limita a la imagen cuyo stem coincide (modo single).
        shared_backend: si se indica (SharedQwen35BeautyBackend), se REUTILIZA el clasificador
            Qwen3.5-9B ya cargado (togglando el LoRA de belleza) → sin 2ª copia ni `ollama stop`.

    Returns:
        (df_rows, score_col) con la puntuación por rostro demand/* (df_rows lleva
        `Img ID` y `track_id` para enlazar con la fila por-persona), o (None, None) si
        el pase no aplica / no hay nada que puntuar / el backend no está disponible.
    """
    if not getattr(config, "ENABLE_BEAUTY_PASS", True):
        return None, None

    header("Pase de belleza diferido (solo demand/*)")

    demand_map = load_phase1_demand(str(json_dir))
    if only_stem is not None:
        demand_map = {k: v for k, v in demand_map.items()
                      if v["path"].stem == only_stem}
    if not demand_map:
        info("  sin summaries de fase 1; omito belleza")
        return None, None

    items = build_items_from_phase1(demand_map, solo_demand=True)
    if not items:
        info("  sin personas demand/* en este run; omito belleza")
        return None, None

    from models.beauty import BeautyEstimator

    # Base COMPARTIDO: reutilizar el clasificador Qwen3.5-9B ya cargado (adapter de belleza).
    if shared_backend is not None and shared_backend.is_loaded():
        backend = shared_backend
        info("  Beauty backend: [dim]base COMPARTIDO (Qwen3.5-9B ya en VRAM, LoRA belleza ON)")
        # NO se para el modelo del pipeline: ES el mismo objeto que sirve la belleza.
    else:
        # Liberar el modelo del pipeline ANTES de cargar una 2ª copia (no caben en 12 GB).
        _stop_pipeline_model()

        # Beauty backend: Qwen3.5-9B continuous (Transformers, checkpoint-1400).
        from models.beauty_backend_qwen35 import Qwen35BeautyBackend
        backend = Qwen35BeautyBackend()
        info("  Beauty backend: [dim]Transformers Qwen3.5-9B continuo (checkpoint-1400)")

        if not backend.is_loaded():
            warn("  backend de belleza no disponible; omito belleza")
            return None, None

    estimator = BeautyEstimator(backend=backend)
    score_col = f"IA Belleza (1-{int(estimator.scale[1])})"

    info(f"  puntuando belleza de rostros demand/* en {len(items)} imágenes…")
    df_rows, _df_img = score_demand_beauty(
        items, demand_map, model_pose, estimator, solo_demand=True)

    if df_rows.empty:
        info("  ningún rostro demand/* puntuable (sin keypoints faciales / sin match IoU)")
        return None, score_col

    n_scored = df_rows[score_col].apply(lambda v: isinstance(v, (int, float))).sum()
    success(f"  {len(df_rows)} rostros demand/* · {n_scored} puntuados → columna '{score_col}'")
    return df_rows, score_col
