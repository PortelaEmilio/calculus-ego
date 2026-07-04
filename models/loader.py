"""
Carga todos los modelos necesarios para el análisis de participación en multimedia.
"""

from ultralytics import YOLO

from ui import header, info, success, warn, error
from config import (
    DETECTION_MODEL, POSE_MODEL, BEHAVIOUR_MODEL_NAME,
    USE_FINETUNED_MODEL, FINETUNED_LORA_PATH,
    VLM_BACKEND, OLLAMA_MODEL, OLLAMA_HOST,
    OLLAMA_HOSTS,
    ENABLE_POSE_ESTIMATION, ENABLE_BEHAVIOUR_CLASSIFICATION,
    ENABLE_ACTIVITY_CLASSIFICATION, ENABLE_BODY_DISPLAY_CLASSIFICATION,
    ENABLE_LOCATION_CLASSIFICATION, ENABLE_BODY_SHAPE_CLASSIFICATION,
    ENABLE_ACCESSORY_CLASSIFICATION, ENABLE_OCR,
    ENABLE_GENDER_CLASSIFICATION, ENABLE_AGE_CLASSIFICATION,
    ENABLE_BEAUTY_ESTIMATION, ENABLE_SOCIAL_DISTANCE,
    YOLO_DEVICE,
)
from models.backends import create_backend
from models.behaviour import BehaviourClassifier
from models.activity import ActivityClassifier
from models.body_display import BodyDisplayClassifier
from models.location import LocationClassifier
from models.accessory import AccessoryClassifier
from models.ocr import OCRClassifier
from models.age_gender import AgeGenderClassifier
from models.beauty import BeautyEstimator
from models.social_distance import SocialDistanceClassifier
from models.person_attributes import PersonAttributesClassifier
from models.scene_context import SceneContextClassifier


def load_all_models():
    """
    Carga todos los modelos necesarios para el análisis.

    Returns:
        Diccionario con todos los modelos y clasificadores cargados, o None si falla YOLO.
    """
    import config as _cfg
    use_pose_as_detector = getattr(_cfg, "USE_POSE_AS_DETECTOR", False) and ENABLE_POSE_ESTIMATION

    header("Cargando modelos")
    if use_pose_as_detector:
        info(f"  YOLO pose-único (detección + pose): [dim]{POSE_MODEL}")
    else:
        info(f"  YOLO detección: [dim]{DETECTION_MODEL}")
        if ENABLE_POSE_ESTIMATION:
            info(f"  YOLO pose: [dim]{POSE_MODEL}")

    try:
        if use_pose_as_detector:
            # Pose-único: un solo modelo hace detección Y pose. model_detect y
            # model_pose apuntan al MISMO objeto → 1 carga, keypoints alineados.
            model_pose = YOLO(POSE_MODEL)
            model_detect = model_pose
        else:
            model_detect = YOLO(DETECTION_MODEL)
            model_pose = YOLO(POSE_MODEL) if ENABLE_POSE_ESTIMATION else None
        # Forzar dispositivo (CPU si VRAM justa por VLM; ver config.YOLO_DEVICE).
        if YOLO_DEVICE:
            info(f"  YOLO device: [dim]{YOLO_DEVICE}")
            try:
                model_detect.to(YOLO_DEVICE)
                if model_pose is not None:
                    model_pose.to(YOLO_DEVICE)
            except Exception as e:
                warn(f"  No se pudo mover YOLO a {YOLO_DEVICE}: {e}")
    except Exception as e:
        error(f"  Error al cargar YOLO: {e}")
        return None

    # Backend VLM compartido por todos los clasificadores de visión
    behaviour_classifier = None
    activity_classifier = None
    body_display_classifier = None
    location_classifier = None
    body_shape_classifier = None
    accessory_classifier = None
    ocr_classifier = None
    gender_classifier = None
    age_classifier = None
    person_attributes_classifier = None
    scene_context_classifier = None

    needs_vlm = (
        ENABLE_BEHAVIOUR_CLASSIFICATION or ENABLE_ACTIVITY_CLASSIFICATION or
        ENABLE_BODY_DISPLAY_CLASSIFICATION or ENABLE_LOCATION_CLASSIFICATION or
        ENABLE_BODY_SHAPE_CLASSIFICATION or ENABLE_ACCESSORY_CLASSIFICATION or
        ENABLE_OCR or ENABLE_GENDER_CLASSIFICATION or ENABLE_AGE_CLASSIFICATION
    )

    if needs_vlm:
        backend_label = VLM_BACKEND
        if VLM_BACKEND == "ollama":
            try:
                from config import ACTIVE_OLLAMA_PROFILE
                backend_label = f"ollama · {ACTIVE_OLLAMA_PROFILE} ({OLLAMA_MODEL})"
            except ImportError:
                pass
        info(f"  Backend VLM: [dim]{backend_label}")
        # Override del modelo VLM por entorno (benchmark gemma-4 vs Qwen3.5) sin
        # editar config.py: VLM_MODEL_OVERRIDE="Qwen/Qwen3.5-9B" → create_backend
        # devuelve el Qwen35VLMBackend (guardado por nombre). Vacío → producción.
        import os as _os
        _model_name = _os.environ.get("VLM_MODEL_OVERRIDE") or BEHAVIOUR_MODEL_NAME
        if _model_name != BEHAVIOUR_MODEL_NAME:
            info(f"  [yellow]VLM_MODEL_OVERRIDE activo:[/] {_model_name}")
        try:
            backend = create_backend(
                VLM_BACKEND,
                model_name=_model_name,
                use_finetuned=USE_FINETUNED_MODEL,
                lora_path=FINETUNED_LORA_PATH,
                ollama_model=OLLAMA_MODEL,
                ollama_host=OLLAMA_HOST,
                ollama_hosts=OLLAMA_HOSTS,
            )

            # PROD 2026-06-29: con prompts_gemma4_json el modelo emite JSON
            # {"gender":{"class":...}}; los parsers line-based esperan `Label: value`.
            # Envolvemos el backend con un proxy que aplana JSON→líneas.
            # prompts_qwen3 = variante JSON (test de joyería) → mismo proxy.
            if getattr(_cfg, "VLM_PROMPT_MODULE", "") in ("prompts_gemma4_json", "prompts_qwen3"):
                from models.backends.json_flatten_backend import JsonFlatteningBackend
                backend = JsonFlatteningBackend(backend)
                info(f"  Proxy JSON→líneas activo ({_cfg.VLM_PROMPT_MODULE})")

            behaviour_classifier = BehaviourClassifier(backend=backend)

            if ENABLE_ACTIVITY_CLASSIFICATION:
                activity_classifier = ActivityClassifier(backend=backend)
            if ENABLE_BODY_DISPLAY_CLASSIFICATION:
                body_display_classifier = BodyDisplayClassifier(backend=backend)
            if ENABLE_LOCATION_CLASSIFICATION:
                location_classifier = LocationClassifier(backend=backend)
            # Silueta ELIMINADA (2026-07-04): el clasificador standalone BodyShapeClassifier
            # ya no existe. Peso/musculatura viajan en el JSON de person_attributes (merge A),
            # gateado por ENABLE_BODY_SHAPE_CLASSIFICATION → body_shape_classifier queda None.
            if ENABLE_ACCESSORY_CLASSIFICATION:
                accessory_classifier = AccessoryClassifier(backend=backend)
            if ENABLE_OCR:
                ocr_classifier = OCRClassifier(backend=backend)
            if ENABLE_GENDER_CLASSIFICATION or ENABLE_AGE_CLASSIFICATION:
                age_gender_classifier = AgeGenderClassifier(backend=backend)
                gender_classifier = age_gender_classifier
                age_classifier = age_gender_classifier

            # Clasificadores fusionados (Merge A + Merge B). Reemplazan en
            # tiempo de ejecución a las llamadas individuales en el flujo
            # `processing/image.py:annotate_frame()` y en
            # `processing/video.py:analyze_scene_vlm()`.
            person_attributes_classifier = PersonAttributesClassifier(backend=backend)
            scene_context_classifier = SceneContextClassifier(backend=backend)

        except Exception as e:
            warn(f"  Error al cargar backend VLM: {e}")
            behaviour_classifier = None
            activity_classifier = None
            body_display_classifier = None
            location_classifier = None
            body_shape_classifier = None
            accessory_classifier = None
            ocr_classifier = None
            gender_classifier = None
            age_classifier = None
            person_attributes_classifier = None
            scene_context_classifier = None

    social_distance_classifier = None
    if ENABLE_SOCIAL_DISTANCE:
        try:
            sd_backend = behaviour_classifier.backend if behaviour_classifier is not None else None
            social_distance_classifier = SocialDistanceClassifier(backend=sd_backend)
            if sd_backend is None or not sd_backend.is_loaded():
                warn("  Backend VLM no disponible para distancia social")
                social_distance_classifier = None
        except Exception as e:
            warn(f"  Error al inicializar distancia social: {e}")
            social_distance_classifier = None

    beauty_estimator = None
    if ENABLE_BEAUTY_ESTIMATION:
        import config as _cfg
        # PRIORIDAD 1 — base COMPARTIDO: si el clasificador Qwen3.5 ya trae el LoRA de
        # belleza adjunto (BEAUTY_SHARE_CLASSIFIER_BASE), la belleza inline reutiliza esa
        # MISMA instancia vía generate_beauty() (adapter ON), sin cargar un 2º modelo. Esto
        # gana a la rama gemma4 legacy —que carga un modelo aparte y OOMea junto al Qwen3.5.
        shared_inner = None
        if behaviour_classifier is not None and behaviour_classifier.backend is not None:
            shared_inner = getattr(behaviour_classifier.backend, "inner", behaviour_classifier.backend)
        if shared_inner is not None and getattr(shared_inner, "_has_beauty_adapter", False):
            try:
                from models.beauty_backend_qwen35 import SharedQwen35BeautyBackend
                beauty_estimator = BeautyEstimator(backend=SharedQwen35BeautyBackend(shared_inner))
                info("  Belleza inline vía base COMPARTIDO (adapter ON en generate_beauty)")
            except Exception as e:
                warn(f"  Error al inicializar beauty compartido: {e}")
                beauty_estimator = None
        elif needs_vlm and behaviour_classifier is not None and behaviour_classifier.backend is not None and behaviour_classifier.backend.is_loaded():
            try:
                beauty_estimator = BeautyEstimator(backend=behaviour_classifier.backend)
            except Exception as e:
                warn(f"  Error al inicializar beauty estimator: {e}")
                beauty_estimator = None
        else:
            warn("  Backend VLM no disponible para belleza")

    success("  Modelos listos")

    return {
        'model_detect': model_detect,
        'model_pose': model_pose,
        'beauty_estimator': beauty_estimator,
        'gender_classifier': gender_classifier,
        'age_classifier': age_classifier,
        'behaviour_classifier': behaviour_classifier,
        'activity_classifier': activity_classifier,
        'body_display_classifier': body_display_classifier,
        'location_classifier': location_classifier,
        'body_shape_classifier': body_shape_classifier,
        'accessory_classifier': accessory_classifier,
        'ocr_classifier': ocr_classifier,
        'social_distance_classifier': social_distance_classifier,
        'person_attributes_classifier': person_attributes_classifier,
        'scene_context_classifier': scene_context_classifier,
    }
