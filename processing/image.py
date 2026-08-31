"""
Procesamiento de imágenes: anotación, detección de collages y process_image().
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import config

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    YOLO = None
    Results = None

from config import (
    ENABLE_POSE_ESTIMATION, ENABLE_TRACKING, ENABLE_BEAUTY_ESTIMATION,
    ENABLE_GENDER_CLASSIFICATION, ENABLE_AGE_CLASSIFICATION,
    ENABLE_BEHAVIOUR_CLASSIFICATION, ENABLE_ACTIVITY_CLASSIFICATION,
    ENABLE_BODY_DISPLAY_CLASSIFICATION, ENABLE_LOCATION_CLASSIFICATION,
    ENABLE_BODY_SHAPE_CLASSIFICATION,
    ENABLE_ACCESSORY_CLASSIFICATION,
    ENABLE_SOCIAL_DISTANCE, ENABLE_COLLAGE_DETECTION, ENABLE_OCR,
    ENABLE_PER_PERSON_ANNOTATED_IMAGE, PER_PERSON_CROP_MARGIN,
    COLLAGE_MIN_PANELS, COLLAGE_MIN_PANEL_SIZE_PERCENT,
    COLLAGE_HOUGH_THRESHOLD, COLLAGE_HOUGH_MIN_LINE_LENGTH, COLLAGE_HOUGH_MAX_LINE_GAP,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, PERSON_CLASS_ID, OUTPUT_DIR, VISUALIZATION,
    MIN_PERSON_CROP_SIZE,
)
from utils.visualization import (
    put_text_pil, draw_skeleton, draw_detection_with_info, save_person_crop,
    draw_corner_bbox, draw_all_person_chips, build_person_display_attrs,
    render_person_annotated_crop, save_person_annotated_crop,
    is_waist_visible, is_shoulder_visible, is_frontal_pose_with_waist, has_five_face_keypoints_visible,
    extract_face_crop, get_color_for_track_id, count_visible_keypoints, draw_scene_number,
    bbox_occupancy,
)
from models.accessory import ACCESSORY_CATEGORIES


def create_person_highlighted_image(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """Devuelve copia del frame con recuadro amarillo + etiqueta 'TARGET'.

    El grosor del recuadro y el tamaño del texto escalan con la diagonal del
    frame para que la marca sea visible incluso cuando el bbox es pequeño
    (ej. avatares en miniatura). Esto ayuda a VLMs pequeños (gemma) que de
    otro modo ignoran marcas finas y describen al sujeto dominante.
    """
    img = frame.copy()
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    diag = (h * h + w * w) ** 0.5
    # Grosor adaptativo: tres criterios, gana el máximo (con un cap superior
    # para no saturar bboxes grandes).
    #  - base 8 px (subido desde 4 — gemma4 ignora bordes finos tras downsize).
    #  - diag/200 escala con el tamaño del frame.
    #  - bbox_diag/10 escala con el tamaño del TARGET (CRÍTICO para avatares
    #    pequeños: si el bbox es 80×80, el borde debe ser ~8 px para que el
    #    VLM lo registre tras el resize a 896 que aplica Ollama).
    bbox_diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    thickness = max(8, int(diag / 200), int(bbox_diag / 10))
    thickness = min(thickness, max(8, int(diag / 30)))  # cap para no saturar
    font_scale = max(0.9, diag / 1500.0)
    font_thickness = max(2, int(font_scale * 2))

    YELLOW = (0, 255, 255)  # BGR
    BLACK = (0, 0, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, thickness)

    label = "TARGET"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    pad = max(4, thickness)

    label_y_top = y1 - th - 2 * pad
    if label_y_top < 0:
        label_y_top = y2 + pad
        label_y_bottom = label_y_top + th + 2 * pad
    else:
        label_y_bottom = y1

    label_x_left = max(0, x1)
    label_x_right = min(w, label_x_left + tw + 2 * pad)

    cv2.rectangle(img, (label_x_left, label_y_top),
                  (label_x_right, label_y_bottom), YELLOW, -1)
    cv2.putText(img, label,
                (label_x_left + pad, label_y_bottom - pad),
                font, font_scale, BLACK, font_thickness, cv2.LINE_AA)

    return img


def are_shoulders_visible(keypoints: np.ndarray, min_conf: float = 0.65) -> bool:
    """Verifica que AMBOS hombros sean claramente visibles (umbral estricto para evitar falsos positivos)."""
    if keypoints is None or len(keypoints) < 7:
        return False
    return keypoints[5][2] >= min_conf and keypoints[6][2] >= min_conf


def are_hips_visible(keypoints: np.ndarray, min_conf: float = 0.5) -> bool:
    """Verifica que al menos una cadera sea visible."""
    if keypoints is None or len(keypoints) < 13:
        return False
    return keypoints[11][2] >= min_conf or keypoints[12][2] >= min_conf


def _iou_xyxy(a: tuple, b: tuple) -> float:
    """IoU entre dos cajas (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def annotate_frame(frame: np.ndarray, results_detect, results_pose,
                   with_tracking: bool = False, beauty_estimator=None,
                   gender_classifier=None, age_classifier=None,
                   behaviour_classifier=None, activity_classifier=None,
                   body_display_classifier=None, location_classifier=None,
                   body_shape_classifier=None, accessory_classifier=None, social_distance_classifier=None,
                   person_attributes_classifier=None, scene_context_classifier=None,
                   frame_idx: int = 0, output_dir: Path = None,
                   behaviour_cache: dict | None = None,
                   activity_cache: dict | None = None,
                   body_display_cache: dict | None = None,
                   location_cache: dict | None = None,
                   body_shape_cache: dict | None = None,
                   accessory_cache: dict | None = None,
                   beauty_cache: dict | None = None,
                   gender_cache: dict | None = None,
                   age_cache: dict | None = None,
                   scene_number: int = 1, total_scenes: int = 1,
                   output_basename: str | None = None,
                   fade_alpha: float = 1.0):
    """
    Anota un frame con detecciones, keypoints, tracking, género, edad, belleza y número de escena.

    Returns:
        Frame anotado, estadísticas de detección, y los 9 caches actualizados.
    """
    if behaviour_cache is None:
        behaviour_cache = {}

    if activity_cache is None:
        activity_cache = {}

    if body_display_cache is None:
        body_display_cache = {}

    if location_cache is None:
        location_cache = {}

    if body_shape_cache is None:
        body_shape_cache = {}

    if accessory_cache is None:
        accessory_cache = {}

    if beauty_cache is None:
        beauty_cache = {}

    if gender_cache is None:
        gender_cache = {}

    if age_cache is None:
        age_cache = {}

    annotated = frame.copy()
    num_persons = 0
    # Conteo total de detecciones YOLO26 válidas (post MIN_PERSON_CROP_SIZE,
    # pre BBOX_MIN_FRAME_RATIO + dedup IoU). Se reporta como `IA Nº pers.` para
    # que la métrica de recuento refleje todo lo que YOLO ha detectado, no sólo
    # lo que llega al VLM.
    num_persons_yolo = 0
    track_ids = []
    beauty_scores = []
    gender_classifications = []
    age_classifications = []
    behaviour_classifications = []
    activity_classifications = []
    body_display_classifications = []
    location_classifications = []
    body_shape_classifications = []
    accessory_classifications = []
    social_distance_classifications = []

    # Obtener keypoints de pose para detectar orientación frontal
    keypoints_list = []
    if ENABLE_POSE_ESTIMATION and results_pose[0].keypoints is not None:
        keypoints_data = results_pose[0].keypoints.data
        keypoints_list = [kpts.cpu().numpy() for kpts in keypoints_data]

    # Emparejado de keypoints por GEOMETRÍA (2026-06-29): en vez de asociar
    # keypoints a cada persona por índice del detector (frágil: el orden del
    # detector y del modelo de pose no tiene por qué coincidir → mal-asignación
    # en multipersona), se empareja cada caja con los keypoints cuya caja de
    # pose tiene mayor IoU. Robusto tanto en pose-único (IoU=1, se empareja
    # consigo mismo) como en el dual legacy. Si no hay solape suficiente → None.
    _use_pose_detector = getattr(config, "USE_POSE_AS_DETECTOR", False)
    _pose_kpt_pairs = []  # [(xyxy, kpts)]
    if (_use_pose_detector and keypoints_list
            and results_pose[0].boxes is not None and len(results_pose[0].boxes) > 0):
        try:
            _pkb = results_pose[0].boxes.xyxy.cpu().numpy()
            for _i, _kp in enumerate(keypoints_list):
                if _i < len(_pkb):
                    _pose_kpt_pairs.append((tuple(map(int, _pkb[_i])), _kp))
        except Exception:
            _pose_kpt_pairs = []

    def _match_kpts_by_iou(_bbox, _person_idx, _fallback_idx):
        """Keypoints de la caja de pose con mayor IoU (pose-único); si no, índice legacy."""
        if _use_pose_detector and _pose_kpt_pairs:
            best, best_iou = None, 0.3
            for _pb, _kp in _pose_kpt_pairs:
                v = _iou_xyxy(_bbox, _pb)
                if v > best_iou:
                    best, best_iou = _kp, v
            return best
        return keypoints_list[_fallback_idx] if _fallback_idx < len(keypoints_list) else None

    # Procesar detecciones
    has_detect_boxes = results_detect[0].boxes is not None and len(results_detect[0].boxes) > 0
    pose_boxes = None
    if ENABLE_POSE_ESTIMATION and results_pose and results_pose[0].boxes is not None and len(results_pose[0].boxes) > 0:
        pose_boxes = results_pose[0].boxes

    # Píldoras de atributos ("chips") de TODAS las personas del frame, acumuladas
    # durante el bucle por persona y dibujadas juntas al final (draw_all_person_chips)
    # para que el layout evite solapes entre bloques de personas distintas.
    chip_entries = []

    if has_detect_boxes or pose_boxes is not None:
        boxes = results_detect[0].boxes if has_detect_boxes else None

        # Obtener tracking IDs si están disponibles
        ids = boxes.id.cpu().numpy() if boxes is not None and boxes.id is not None and ENABLE_TRACKING else None

        # BATCH PROCESSING: Acumular todos los crops primero
        all_crops = []
        # (person_idx, track_id, x1, y1, x2, y2, has_shoulder_visible, kpts, box_ref)
        all_metadata = []

        person_idx = 0  # Contador específico para personas detectadas
        # xyxy de TODA detección contada en num_persons_yolo (aceptada O filtrada
        # por BBOX_MIN_FRAME_RATIO). El pose-loop deduplica contra esta lista;
        # si solo comparara contra las aceptadas, una caja filtrada por ratio se
        # recontaría al reaparecer como pose-box (con pose-único son las mismas
        # cajas → doble conteo sistemático en `IA Nº pers.`).
        seen_xyxy = []
        if has_detect_boxes:
            for idx, box in enumerate(boxes):
                if int(box.cls) != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                # Filtrar crops demasiado pequeños (fotos de perfil mínimas)
                crop_h, crop_w = person_crop.shape[:2]
                if crop_h < MIN_PERSON_CROP_SIZE or crop_w < MIN_PERSON_CROP_SIZE:
                    continue

                # YOLO26 sí detectó persona aquí — cuenta en el total aunque
                # luego se filtre por tamaño de bbox.
                num_persons_yolo += 1
                seen_xyxy.append((x1, y1, x2, y2))

                # Filtro por ratio del frame: ignorar detecciones cuyo bbox
                # cubre menos del BBOX_MIN_FRAME_RATIO del área total. El VLM
                # no puede resolver TARGETs minúsculos tras el downsize y
                # describe al sujeto dominante del panel cercano.
                bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
                frame_area = max(1, frame.shape[0] * frame.shape[1])
                if bbox_area / frame_area < getattr(config, "BBOX_MIN_FRAME_RATIO", 0.01):
                    continue

                num_persons += 1
                if ids is not None:
                    track_id = int(ids[idx])
                else:
                    track_id = f"det_{frame_idx}_{idx}"

                track_ids.append(track_id)

                kpts = _match_kpts_by_iou((x1, y1, x2, y2), person_idx, person_idx)
                has_shoulder_visible = False
                if ENABLE_POSE_ESTIMATION and ENABLE_BODY_SHAPE_CLASSIFICATION and kpts is not None:
                    has_shoulder_visible = is_shoulder_visible(kpts)

                all_crops.append(person_crop)
                all_metadata.append((person_idx, track_id, x1, y1, x2, y2, has_shoulder_visible, kpts, box))
                person_idx += 1

        # Segundo paso: añadir personas detectadas sólo por el modelo de pose
        # (cuando el detector las pierde pero el pose-model sí las encuentra).
        # Dedup por IoU contra TODAS las cajas ya contadas (aceptadas Y filtradas
        # por ratio) para evitar doble conteo: con pose-único las pose-boxes son
        # las mismas del detect-loop, y una caja filtrada por ratio reaparecería
        # aquí y se recontaría en num_persons_yolo (bug corregido 2026-07-09).
        if pose_boxes is not None:
            existing_xyxy = list(seen_xyxy)
            for pose_idx, pose_box in enumerate(pose_boxes):
                x1, y1, x2, y2 = map(int, pose_box.xyxy[0])
                candidate = (x1, y1, x2, y2)

                # Dedup contra toda caja ya contada en el detect-loop o en este loop
                max_iou = max((_iou_xyxy(candidate, e) for e in existing_xyxy), default=0.0)
                if max_iou >= 0.3:
                    continue

                person_crop = frame[y1:y2, x1:x2]
                crop_h, crop_w = person_crop.shape[:2]
                if crop_h < MIN_PERSON_CROP_SIZE or crop_w < MIN_PERSON_CROP_SIZE:
                    continue

                # Pose-only detección válida (no duplicada vs detect-loop por
                # el dedup IoU de arriba) — cuenta en el total aunque luego se
                # filtre por tamaño.
                num_persons_yolo += 1
                existing_xyxy.append(candidate)

                # Filtro por ratio del frame (mismo motivo que en detect-loop)
                bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
                frame_area = max(1, frame.shape[0] * frame.shape[1])
                if bbox_area / frame_area < getattr(config, "BBOX_MIN_FRAME_RATIO", 0.01):
                    continue

                num_persons += 1
                track_id = f"pose_{frame_idx}_{pose_idx}"
                track_ids.append(track_id)

                kpts = _match_kpts_by_iou((x1, y1, x2, y2), pose_idx, pose_idx)
                has_shoulder_visible = False
                if ENABLE_BODY_SHAPE_CLASSIFICATION and kpts is not None:
                    has_shoulder_visible = is_shoulder_visible(kpts)

                all_crops.append(person_crop)
                all_metadata.append((person_idx, track_id, x1, y1, x2, y2, has_shoulder_visible, kpts, pose_box))
                person_idx += 1

        # Resultados completos — inicializados antes del bucle de recolección
        # para poder pre-asignar 'na' a personas sin hombros visibles
        gender_results = [None] * len(all_crops)
        age_results = [None] * len(all_crops)
        behaviour_results = [None] * len(all_crops)
        activity_results = [None] * len(all_crops)
        body_display_results = [None] * len(all_crops)
        location_results = [None] * len(all_crops)
        body_shape_results = [None] * len(all_crops)
        accessory_results = [None] * len(all_crops)
        beauty_results = [None] * len(all_crops)
        social_distance_results = [None] * len(all_crops)

        # ====================================================================
        # BATCH CLASSIFICATION (merged prompts).
        # Group A — atributos por persona (gender + age + behaviour +
        #           body_display + body_shape + accessory) → person_attributes
        #           Una sola llamada VLM por persona sobre el person crop.
        # Group B — contexto de escena (activity + location) →
        #           scene_context_classifier. Una sola llamada VLM por persona
        #           sobre el frame completo con la persona destacada en rojo.
        # social_distance se mantiene separado (atajos por keypoints + entrada
        # propia). Beauty está desactivado por defecto en config.
        # ====================================================================
        crops_to_analyze_person_attrs = []
        flags_include_body_shape = []
        indices_to_analyze_person_attrs = []

        images_to_analyze_scene = []
        indices_to_analyze_scene = []
        flags_include_social_distance = []  # aligned to images_to_analyze_scene

        images_to_analyze_social_distance = []
        indices_to_analyze_social_distance = []

        crops_to_analyze_beauty = []
        indices_to_analyze_beauty = []

        # Track cuáles caches deben actualizarse para este crop (uno por
        # categoría) tras la llamada a person_attributes.
        person_attrs_target_caches = []
        # Track caches a actualizar tras scene_context.
        scene_target_caches = []

        # ¿Necesitamos algún campo del Group A para esta persona?
        def _needs_person_attrs(track_id) -> bool:
            return (
                (ENABLE_BEHAVIOUR_CLASSIFICATION and track_id not in behaviour_cache) or
                (ENABLE_BODY_DISPLAY_CLASSIFICATION and track_id not in body_display_cache) or
                (ENABLE_BODY_SHAPE_CLASSIFICATION and track_id not in body_shape_cache) or
                (ENABLE_ACCESSORY_CLASSIFICATION and track_id not in accessory_cache) or
                (ENABLE_GENDER_CLASSIFICATION and track_id not in gender_cache) or
                (ENABLE_AGE_CLASSIFICATION and track_id not in age_cache)
            )

        def _needs_scene_context(track_id) -> bool:
            return (
                (ENABLE_ACTIVITY_CLASSIFICATION and track_id not in activity_cache) or
                (ENABLE_LOCATION_CLASSIFICATION and track_id not in location_cache)
            )

        for crop_idx, (person_idx, track_id, x1, y1, x2, y2, has_shoulder_visible, kpts, _box_ref) in enumerate(all_metadata):
            highlighted = create_person_highlighted_image(frame, (x1, y1, x2, y2))

            # ---- Group A: person attributes (gender, age, behaviour,
            #               body_display, body_shape, accessory) ------------
            if person_attributes_classifier is not None and _needs_person_attrs(track_id):
                # Silueta ELIMINADA (2026-07-04): ya no se pide al VLM. El "slot"
                # body_shape ahora solo transporta peso (FRS/IMC) + musculatura, que
                # se preguntan SIEMPRE (no dependen del gate de hombros). El flag
                # include_body_shape queda en False y person_attributes lo ignora.
                crops_to_analyze_person_attrs.append(all_crops[crop_idx])
                flags_include_body_shape.append(False)
                indices_to_analyze_person_attrs.append(crop_idx)

            # ---- Social distance (atajo por keypoints; sino delega a
            #      SCENE_PROMPT con frame completo highlighted —o, si
            #      USE_LEGACY_SOCIAL_DISTANCE=True, al módulo standalone
            #      con per-person crop). Calculamos PRIMERO porque el flag
            #      se necesita en el bloque scene siguiente.
            include_sd_in_scene = False
            if ENABLE_SOCIAL_DISTANCE and social_distance_classifier is not None:
                kp_result = social_distance_classifier.classify_from_keypoints(
                    kpts, (x1, y1, x2, y2), frame.shape, len(all_metadata)
                )
                if kp_result is not None:
                    social_distance_results[crop_idx] = kp_result
                elif getattr(config, "SOCIAL_DISTANCE_KEYPOINT_ONLY", True):
                    # 100% keypoints: NUNCA delega al VLM (ahorra 1 llamada VLM
                    # por persona sin pose). Sin keypoints → no evaluable.
                    social_distance_results[crop_idx] = {
                        "category": None, "success": False,
                        "raw_response": "keypoints:none",
                    }
                elif getattr(config, "USE_LEGACY_SOCIAL_DISTANCE", False):
                    images_to_analyze_social_distance.append(all_crops[crop_idx])
                    indices_to_analyze_social_distance.append(crop_idx)
                else:
                    include_sd_in_scene = True  # delegar a SCENE_PROMPT

            # ---- Group B: scene context (activity, location [+ social_distance])
            need_scene = scene_context_classifier is not None and (
                _needs_scene_context(track_id) or include_sd_in_scene
            )
            if need_scene:
                images_to_analyze_scene.append(highlighted)
                indices_to_analyze_scene.append(crop_idx)
                flags_include_social_distance.append(include_sd_in_scene)

            # ---- Beauty (gate por keypoints faciales) ------------------
            if ENABLE_BEAUTY_ESTIMATION and beauty_estimator is not None and track_id not in beauty_cache:
                if kpts is not None:
                    if has_five_face_keypoints_visible(kpts):
                        face_crop = extract_face_crop(frame, kpts, (x1, y1, x2, y2))
                        if face_crop is not None and face_crop.size > 0:
                            crops_to_analyze_beauty.append(face_crop)
                            indices_to_analyze_beauty.append(crop_idx)
                    else:
                        beauty_cache[track_id] = {"score": None, "not_calculable": True}
                else:
                    beauty_cache[track_id] = {"score": None, "not_calculable": True}

        if all_crops:
            # ---- Llamada Group A (1 por persona analizada) ---------------
            # ---- Lanzamos Merge A + Merge B + social_distance EN PARALELO --
            # Las 3 llamadas son independientes y al backend Ollama el cliente
            # libera el GIL durante la espera HTTP, así que ThreadPoolExecutor
            # paraleliza efectivamente. Requiere `OLLAMA_NUM_PARALLEL>=3` en
            # el servidor para que las 3 corran a la vez en GPU.
            from concurrent.futures import ThreadPoolExecutor

            tasks = {}
            _outer = max(1, getattr(config, "OLLAMA_OUTER_PARALLEL", 3))
            # Backend transformers: las 3 tareas comparten un único modelo en
            # GPU. Paralelizar 3 forward passes simultáneos en una RTX 4070
            # de 12 GiB acumula activaciones (~4 GiB cada una) y OOMea.
            # Serializa para que cada tarea libere VRAM antes de la siguiente.
            if getattr(config, "VLM_BACKEND", "ollama") == "transformers":
                _outer = 1
            with ThreadPoolExecutor(max_workers=_outer) as pool:
                if crops_to_analyze_person_attrs:
                    tasks["person_attrs"] = pool.submit(
                        person_attributes_classifier.classify_batch,
                        crops_to_analyze_person_attrs,
                        flags_include_body_shape,
                    )
                if images_to_analyze_scene:
                    tasks["scene"] = pool.submit(
                        scene_context_classifier.classify_batch,
                        images_to_analyze_scene,
                        flags_include_social_distance,
                    )
                if images_to_analyze_social_distance:
                    tasks["social"] = pool.submit(
                        social_distance_classifier.classify_batch,
                        images_to_analyze_social_distance,
                    )

                if "person_attrs" in tasks:
                    (
                        temp_gender_results,
                        temp_age_results,
                        temp_behaviour_results,
                        temp_body_display_results,
                        temp_body_shape_results,
                        temp_accessory_results,
                    ) = tasks["person_attrs"].result()
                    for i, crop_idx in enumerate(indices_to_analyze_person_attrs):
                        if ENABLE_GENDER_CLASSIFICATION:
                            gender_results[crop_idx] = temp_gender_results[i]
                        if ENABLE_AGE_CLASSIFICATION:
                            age_results[crop_idx] = temp_age_results[i]
                        if ENABLE_BEHAVIOUR_CLASSIFICATION:
                            behaviour_results[crop_idx] = temp_behaviour_results[i]
                        if ENABLE_BODY_DISPLAY_CLASSIFICATION:
                            body_display_results[crop_idx] = temp_body_display_results[i]
                        if ENABLE_BODY_SHAPE_CLASSIFICATION:
                            body_shape_results[crop_idx] = temp_body_shape_results[i]
                        if ENABLE_ACCESSORY_CLASSIFICATION:
                            accessory_results[crop_idx] = temp_accessory_results[i]

                if "scene" in tasks:
                    (
                        temp_activity_results,
                        temp_location_results,
                        temp_scene_social_distance_results,
                    ) = tasks["scene"].result()
                    for i, crop_idx in enumerate(indices_to_analyze_scene):
                        if ENABLE_ACTIVITY_CLASSIFICATION:
                            activity_results[crop_idx] = temp_activity_results[i]
                        if ENABLE_LOCATION_CLASSIFICATION:
                            location_results[crop_idx] = temp_location_results[i]
                        # Si esta persona delegó social_distance al SCENE_PROMPT,
                        # absorbe aquí su resultado.
                        if flags_include_social_distance[i]:
                            social_distance_results[crop_idx] = temp_scene_social_distance_results[i]

                if "social" in tasks:
                    temp_social_distance_results = tasks["social"].result()
                    for i, crop_idx in enumerate(indices_to_analyze_social_distance):
                        social_distance_results[crop_idx] = temp_social_distance_results[i]

            # ---- Beauty --------------------------------------------------
            if crops_to_analyze_beauty:
                temp_beauty_results = []
                for face_crop in crops_to_analyze_beauty:
                    beauty_result = beauty_estimator.estimate(face_crop)
                    temp_beauty_results.append(beauty_result)
                for i, crop_idx in enumerate(indices_to_analyze_beauty):
                    beauty_results[crop_idx] = temp_beauty_results[i]

        # Procesar resultados batch y continuar con otras clasificaciones
        for crop_idx, (person_idx, track_id, x1, y1, x2, y2, has_shoulder_visible, kpts, box) in enumerate(all_metadata):
            person_crop = all_crops[crop_idx]

            # Tamaño / ocupación de la persona: área del bbox y % del frame que ocupa.
            # Se adjunta a la entrada ancla (gender_classifications) → build_rows lo
            # superficie como columnas IA Tamaño/BBox. Misma fórmula que BBOX_MIN_FRAME_RATIO.
            bbox_xyxy = [int(x1), int(y1), int(x2), int(y2)]
            bbox_area, occupancy = bbox_occupancy(
                bbox_xyxy, frame.shape[1], frame.shape[0]
            )

            gender_info = gender_results[crop_idx] if crop_idx < len(gender_results) else None
            age_info = age_results[crop_idx] if crop_idx < len(age_results) else None
            behaviour_info = behaviour_results[crop_idx] if crop_idx < len(behaviour_results) else None
            activity_info = activity_results[crop_idx] if crop_idx < len(activity_results) else None
            body_display_info = body_display_results[crop_idx] if crop_idx < len(body_display_results) else None
            location_info = location_results[crop_idx] if crop_idx < len(location_results) else None
            body_shape_info = body_shape_results[crop_idx] if crop_idx < len(body_shape_results) else None
            accessory_info = accessory_results[crop_idx] if crop_idx < len(accessory_results) else None
            beauty_info = beauty_results[crop_idx] if crop_idx < len(beauty_results) else None

            # Guardar clasificaciones de género
            if gender_info and gender_info.get("success"):
                gender_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "gender": gender_info.get("gender"),
                    "all_predictions": gender_info.get("all_predictions"),
                    "bbox": bbox_xyxy,
                    "bbox_area": bbox_area,
                    "occupancy": occupancy,
                })
                gender_cache[track_id] = {
                    "gender": gender_info.get("gender"),
                    "last_frame": frame_idx,
                    "raw_response": gender_info.get("raw_response")
                }
            else:
                if track_id in gender_cache:
                    cached_gender = gender_cache[track_id]
                    gender_info = {
                        "success": True,
                        "gender": cached_gender["gender"],
                    }
                    gender_classifications.append({
                        "track_id": track_id,
                        "frame": frame_idx,
                        "gender": cached_gender["gender"],
                        "bbox": bbox_xyxy,
                        "bbox_area": bbox_area,
                        "occupancy": occupancy,
                    })

            # Guardar clasificaciones de edad
            if age_info and age_info.get("success"):
                age_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "age_group": age_info.get("age_group")
                })
                age_cache[track_id] = {
                    "age_group": age_info.get("age_group"),
                    "last_frame": frame_idx,
                    "raw_response": age_info.get("raw_response")
                }
            else:
                if track_id in age_cache:
                    cached_age = age_cache[track_id]
                    age_info = {
                        "success": True,
                        "age_group": cached_age["age_group"],
                    }
                    age_classifications.append({
                        "track_id": track_id,
                        "frame": frame_idx,
                        "age_group": cached_age["age_group"],
                    })

            # (2026-07-27) Aquí vivía el gate de pose que reescribía
            # demand/affiliation → demand/submission cuando shoulders_below_eyes < 2.88.
            # ELIMINADO: medía postura encorvada, no elevación de cámara (sample 500 de
            # IG: 16 de 21 submissions falsas). Ahora la clase la emite el VLM desde el
            # prompt (`prompts_qwen3` VH), igual en imagen y en vídeo.

            # Guardar clasificaciones de comportamiento
            if behaviour_info and behaviour_info.get("success"):
                behaviour_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "behaviour": behaviour_info.get("behaviour"),
                    "raw_response": behaviour_info.get("raw_response"),
                    # bbox de la persona → enlace robusto (IoU) para el pase diferido
                    # de belleza, que solo puntúa rostros con behaviour demand/*.
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
                behaviour_cache[track_id] = {
                    "behaviour": behaviour_info.get("behaviour"),
                    "last_frame": frame_idx,
                    "raw_response": behaviour_info.get("raw_response")
                }
            else:
                if track_id in behaviour_cache:
                    cached_behaviour = behaviour_cache[track_id]
                    behaviour_info = {
                        "success": True,
                        "behaviour": cached_behaviour["behaviour"],
                        "raw_response": cached_behaviour["raw_response"]
                    }

            # Guardar clasificaciones de actividades
            if activity_info and activity_info.get("success"):
                activity_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "activity": activity_info.get("activity"),
                    "raw_response": activity_info.get("raw_response")
                })
                activity_cache[track_id] = {
                    "activity": activity_info.get("activity"),
                    "last_frame": frame_idx,
                    "raw_response": activity_info.get("raw_response")
                }
            else:
                if track_id in activity_cache:
                    cached_activity = activity_cache[track_id]
                    activity_info = {
                        "success": True,
                        "activity": cached_activity["activity"],
                        "raw_response": cached_activity["raw_response"]
                    }

            # Guardar clasificaciones de exposición del cuerpo
            if body_display_info and body_display_info.get("success"):
                body_display_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "body_display": body_display_info.get("body_display"),
                    "raw_response": body_display_info.get("raw_response")
                })
                body_display_cache[track_id] = {
                    "body_display": body_display_info.get("body_display"),
                    "last_frame": frame_idx,
                    "raw_response": body_display_info.get("raw_response")
                }
            else:
                if track_id in body_display_cache:
                    cached_body_display = body_display_cache[track_id]
                    body_display_info = {
                        "success": True,
                        "body_display": cached_body_display["body_display"],
                        "raw_response": cached_body_display["raw_response"]
                    }

            # Guardar clasificaciones de ubicación
            if location_info and location_info.get("success"):
                location_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "location": location_info.get("location"),
                    "raw_response": location_info.get("raw_response")
                })
                location_cache[track_id] = {
                    "location": location_info.get("location"),
                    "last_frame": frame_idx,
                    "raw_response": location_info.get("raw_response")
                }
            else:
                if track_id in location_cache:
                    cached_location = location_cache[track_id]
                    location_info = {
                        "success": True,
                        "location": cached_location["location"],
                        "raw_response": cached_location["raw_response"]
                    }
                else:
                    err_response = (location_info.get("raw_response") or location_info.get("error", "failed")) if location_info else "no result"
                    location_classifications.append({
                        "track_id": track_id,
                        "frame": frame_idx,
                        "location": "no visible",
                        "raw_response": err_response
                    })

            # Guardar peso corporal + musculatura + vestimenta (silueta ELIMINADA 2026-07-04)
            if body_shape_info and body_shape_info.get("success"):
                # Vestimenta: el VLM decide directamente (incl. "not visible"). El gate de
                # hombros se ELIMINÓ el 2026-07-08 — degradaba κ 0.85→0.71 en el sample 500
                # (forzaba a "not visible" prendas identificables con pose parcial; el VLM
                # ya acierta el "not visible" real con recall 1.0). Ver CLAUDE.md.
                attire_eff = body_shape_info.get("attire")
                body_shape_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "body_weight": body_shape_info.get("body_weight"),
                    "muscle": body_shape_info.get("muscle"),
                    "attire": attire_eff,
                    "raw_response": body_shape_info.get("raw_response")
                })
                body_shape_cache[track_id] = {
                    "body_weight": body_shape_info.get("body_weight"),
                    "muscle": body_shape_info.get("muscle"),
                    "attire": attire_eff,
                    "last_frame": frame_idx,
                    "raw_response": body_shape_info.get("raw_response")
                }
            else:
                if track_id in body_shape_cache:
                    cached_body_shape = body_shape_cache[track_id]
                    body_shape_info = {
                        "success": True,
                        "body_weight": cached_body_shape.get("body_weight"),
                        "muscle": cached_body_shape.get("muscle"),
                        "attire": cached_body_shape.get("attire"),
                        "raw_response": cached_body_shape.get("raw_response")
                    }

            # Guardar clasificaciones de accesorios (formato binario por categoría)
            if accessory_info and accessory_info.get("success"):
                acc_entry = {"track_id": track_id, "frame": frame_idx, "raw_response": accessory_info.get("raw_response")}
                cache_entry = {"last_frame": frame_idx, "raw_response": accessory_info.get("raw_response")}
                for cat in ACCESSORY_CATEGORIES:
                    acc_entry[cat] = accessory_info.get(cat, 0)
                    cache_entry[cat] = accessory_info.get(cat, 0)
                accessory_classifications.append(acc_entry)
                accessory_cache[track_id] = cache_entry
            else:
                if track_id in accessory_cache:
                    cached_accessory = accessory_cache[track_id]
                    acc_entry = {"track_id": track_id, "frame": frame_idx, "raw_response": cached_accessory.get("raw_response"), "success": True}
                    for cat in ACCESSORY_CATEGORIES:
                        acc_entry[cat] = cached_accessory.get(cat, 0)
                    accessory_info = acc_entry

            # Guardar clasificaciones de belleza
            beauty_score = None
            if beauty_info and beauty_info.get("success"):
                beauty_score = beauty_info.get("score")
                beauty_scores.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "score": beauty_score,
                    "raw_response": beauty_info.get("raw_response")
                })
                beauty_cache[track_id] = {
                    "score": beauty_score,
                    "last_frame": frame_idx,
                    "raw_response": beauty_info.get("raw_response")
                }
            elif track_id in beauty_cache:
                cached_beauty = beauty_cache[track_id]
                if cached_beauty.get("score") is not None and not cached_beauty.get("not_calculable", False):
                    beauty_score = cached_beauty["score"]
                    beauty_scores.append({
                        "track_id": track_id,
                        "frame": frame_idx,
                        "score": beauty_score,
                        "raw_response": cached_beauty.get("raw_response")
                    })

            # Clasificación de distancia social (VLM con imagen completa marcada)
            social_distance_info = social_distance_results[crop_idx] if crop_idx < len(social_distance_results) else None
            if social_distance_info and social_distance_info.get("success"):
                social_distance_classifications.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "category": social_distance_info.get("category"),
                    "raw_response": social_distance_info.get("raw_response"),
                })

            crop_basename = output_basename if output_basename else f"frame_{frame_idx}"

            # Guardar crop de la persona (opcional)
            if output_dir is not None and person_crop.size > 0:
                save_person_crop(person_crop, output_dir, crop_basename, person_idx, track_id)

            person_color = get_color_for_track_id(track_id)

            # Recorte anotado por persona (frame limpio, antes de aplicar el bbox global)
            if ENABLE_PER_PERSON_ANNOTATED_IMAGE and output_dir is not None:
                annotated_person_crop = render_person_annotated_crop(
                    frame, box,
                    margin_percent=PER_PERSON_CROP_MARGIN,
                    keypoints=kpts,
                    track_id=track_id, beauty_score=beauty_score,
                    gender_info=gender_info, age_info=age_info,
                    behaviour_info=behaviour_info, activity_info=activity_info,
                    body_display_info=body_display_info, location_info=location_info,
                    body_shape_info=body_shape_info, accessory_info=accessory_info,
                    social_distance_info=social_distance_info,
                    person_color=person_color,
                )
                save_person_annotated_crop(
                    annotated_person_crop, output_dir, crop_basename, person_idx, track_id
                )

            if VISUALIZATION.get("style", "chips") == "chips":
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                attrs = build_person_display_attrs(
                    gender_info, age_info, behaviour_info, activity_info,
                    body_display_info, location_info, body_shape_info,
                    accessory_info, social_distance_info, beauty_score,
                    occupancy=occupancy,
                )
                annotated = draw_corner_bbox(
                    annotated, bx1, by1, bx2, by2,
                    VISUALIZATION.get("corner_color", (255, 255, 255)),
                    VISUALIZATION.get("corner_bracket_thickness", 3),
                    VISUALIZATION.get("corner_bracket_len", 20), fade_alpha,
                )
                chip_entries.append({
                    "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                    "track_id": track_id, "person_color": person_color, "attrs": attrs,
                })
            else:
                annotated = draw_detection_with_info(
                    annotated, box, track_id, beauty_score, gender_info, age_info,
                    behaviour_info, activity_info, body_display_info, location_info, body_shape_info, accessory_info, social_distance_info,
                    person_color=person_color, occupancy=occupancy
                )

    # Dibujar las píldoras de TODAS las personas juntas (tras conocer a todo el mundo)
    # para que el layout pueda evitar que los bloques de personas distintas se solapen.
    if chip_entries:
        annotated = draw_all_person_chips(annotated, chip_entries, fade_alpha=fade_alpha)

    # Dibujar esqueletos
    if ENABLE_POSE_ESTIMATION:
        for kpts in keypoints_list:
            annotated = draw_skeleton(annotated, kpts, fade_alpha=fade_alpha)

    stats = {
        "num_persons": num_persons,
        "num_persons_yolo": num_persons_yolo,
        "track_ids": track_ids,
        "beauty_scores": beauty_scores,
        "gender_classifications": gender_classifications,
        "age_classifications": age_classifications,
        "behaviour_classifications": behaviour_classifications,
        "activity_classifications": activity_classifications,
        "body_display_classifications": body_display_classifications,
        "location_classifications": location_classifications,
        "body_shape_classifications": body_shape_classifications,
        "accessory_classifications": accessory_classifications,
        "social_distance_classifications": social_distance_classifications
    }

    if scene_number is not None and total_scenes is not None:
        annotated = draw_scene_number(annotated, scene_number, total_scenes)

    return annotated, stats, behaviour_cache, activity_cache, body_display_cache, location_cache, body_shape_cache, accessory_cache, beauty_cache, gender_cache, age_cache


# ============================================================================
# DETECCIÓN Y PROCESAMIENTO DE COLLAGES
# ============================================================================

def detect_collage_by_contours(image: np.ndarray, min_panel_size_percent: float = 8) -> list:
    """
    Detecta paneles en un collage mediante detección de contornos/márgenes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    total_area = height * width
    min_area = (min_panel_size_percent / 100.0) * total_area

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_percent = (area / total_area) * 100

        if area >= min_area and area_percent < 90:
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

            if 0.2 < aspect_ratio < 20:
                if w > width * 0.15 and h > height * 0.15:
                    panels.append((x, y, w, h))

    if panels:
        panels.sort(key=lambda p: (p[1] // (height // 3), p[0]))

    return panels


def detect_collage_by_lines(image: np.ndarray, threshold: int = 100,
                            min_line_length: int = 200, max_line_gap: int = 50) -> list:
    """
    Detecta paneles en un collage mediante detección de líneas rectas divisorias.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv2.Canny(gray, 30, 100)

    adaptive_min_length = max(min_line_length, int(min(width, height) * 0.4))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=adaptive_min_length, maxLineGap=max_line_gap)

    if lines is None or len(lines) == 0:
        return []

    def is_dividing_line(line_pos, orientation, margin=10):
        """Verifica si hay alto contraste a ambos lados de la línea"""
        if orientation == 'horizontal':
            if line_pos < margin or line_pos > height - margin:
                return False
            region_above = gray[max(0, line_pos-margin):line_pos, :]
            region_below = gray[line_pos:min(height, line_pos+margin), :]
            if region_above.size == 0 or region_below.size == 0:
                return False
            diff = abs(float(np.mean(region_above)) - float(np.mean(region_below)))
            return diff > 30
        else:  # vertical
            if line_pos < margin or line_pos > width - margin:
                return False
            region_left = gray[:, max(0, line_pos-margin):line_pos]
            region_right = gray[:, line_pos:min(width, line_pos+margin)]
            if region_left.size == 0 or region_right.size == 0:
                return False
            diff = abs(float(np.mean(region_left)) - float(np.mean(region_right)))
            return diff > 30

    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length < min(width, height) * 0.4:
            continue

        if 85 < angle < 95:
            if abs(y2 - y1) > height * 0.5:
                x_pos = int((x1 + x2) / 2)
                if is_dividing_line(x_pos, 'vertical'):
                    vertical_lines.append((x1, y1, x2, y2))

        elif angle < 5 or angle > 175:
            if abs(x2 - x1) > width * 0.5:
                y_pos = int((y1 + y2) / 2)
                if is_dividing_line(y_pos, 'horizontal'):
                    horizontal_lines.append((x1, y1, x2, y2))

    if len(vertical_lines) == 0 and len(horizontal_lines) == 0:
        return []

    def cluster_positions(positions, tolerance):
        if not positions:
            return []
        positions = sorted(positions)
        clusters = [[positions[0]]]
        for pos in positions[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        return [int(np.mean(cluster)) for cluster in clusters]

    v_positions = cluster_positions(
        [int((x1 + x2) / 2) for x1, _, x2, _ in vertical_lines],
        width * 0.03
    )
    h_positions = cluster_positions(
        [int((y1 + y2) / 2) for _, y1, _, y2 in horizontal_lines],
        height * 0.03
    )

    v_positions = [0] + [p for p in v_positions if width * 0.1 < p < width * 0.9] + [width]
    h_positions = [0] + [p for p in h_positions if height * 0.1 < p < height * 0.9] + [height]

    panels = []
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            x = v_positions[j]
            y = h_positions[i]
            w = v_positions[j + 1] - x
            h = h_positions[i + 1] - y

            if w > width * 0.15 and h > height * 0.15:
                panels.append((x, y, w, h))

    return panels


def detect_collage_by_yolo(image, model_detect, min_panel_size_percent=10):
    """
    Detecta paneles usando YOLO para detectar personas y agruparlas espacialmente.
    """
    height, width = image.shape[:2]
    min_panel_area = (height * width) * (min_panel_size_percent / 100.0)

    results = model_detect(image, verbose=False, conf=0.3)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes
    person_indices = [i for i, cls in enumerate(boxes.cls) if int(cls) == 0]

    if len(person_indices) == 0:
        return []

    person_boxes = []
    for idx in person_indices:
        x1, y1, x2, y2 = map(int, boxes[idx].xyxy[0])
        person_boxes.append((x1, y1, x2, y2))

    centers_x = [(x1 + x2) / 2 for x1, _, x2, _ in person_boxes]
    centers_y = [(y1 + y2) / 2 for _, y1, _, y2 in person_boxes]

    centers_x_sorted = sorted(centers_x)
    vertical_divisions = [0]

    for i in range(len(centers_x_sorted) - 1):
        gap = centers_x_sorted[i + 1] - centers_x_sorted[i]
        if gap > width * 0.15:
            division_x = int((centers_x_sorted[i] + centers_x_sorted[i + 1]) / 2)
            vertical_divisions.append(division_x)

    vertical_divisions.append(width)

    centers_y_sorted = sorted(centers_y)
    horizontal_divisions = [0]

    for i in range(len(centers_y_sorted) - 1):
        gap = centers_y_sorted[i + 1] - centers_y_sorted[i]
        if gap > height * 0.15:
            division_y = int((centers_y_sorted[i] + centers_y_sorted[i + 1]) / 2)
            horizontal_divisions.append(division_y)

    horizontal_divisions.append(height)

    panels = []
    for i in range(len(horizontal_divisions) - 1):
        for j in range(len(vertical_divisions) - 1):
            x = vertical_divisions[j]
            y = horizontal_divisions[i]
            w = vertical_divisions[j + 1] - x
            h = horizontal_divisions[i + 1] - y

            panel_has_person = False
            for px1, py1, px2, py2 in person_boxes:
                cx = (px1 + px2) / 2
                cy = (py1 + py2) / 2

                if x < cx < x + w and y < cy < y + h:
                    panel_has_person = True
                    break

            if panel_has_person and (w * h) >= min_panel_area:
                panels.append((x, y, w, h))

    panels.sort(key=lambda p: (p[1], p[0]))

    return panels


def detect_collage_panels(image_path: Path, min_panels: int = 2,
                          min_panel_size_percent: float = 8, model_detect=None) -> tuple:
    """
    Detecta si una imagen es un collage usando tres métodos:
    1. YOLO (personas) - PRIORIDAD
    2. Contornos/márgenes
    3. Líneas divisorias (fallback)

    Returns:
        (is_collage: bool, panels: list, method: str)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return False, [], "none"

    height, width = image.shape[:2]
    total_area = height * width

    # Método 1: YOLO
    if model_detect is not None:
        panels_yolo = detect_collage_by_yolo(image, model_detect, min_panel_size_percent)

        if len(panels_yolo) >= min_panels:
            panels_area = sum(w * h for _, _, w, h in panels_yolo)
            coverage = panels_area / total_area

            def rectangles_overlap(r1, r2):
                x1, y1, w1, h1 = r1
                x2, y2, w2, h2 = r2
                return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

            overlaps = 0
            for i in range(len(panels_yolo)):
                for j in range(i + 1, len(panels_yolo)):
                    if rectangles_overlap(panels_yolo[i], panels_yolo[j]):
                        overlaps += 1

            if 0.5 < coverage < 1.5 and overlaps < len(panels_yolo) // 2:
                return True, panels_yolo, "yolo"

    # Método 2: Contornos
    panels_contours = detect_collage_by_contours(image, min_panel_size_percent)

    if len(panels_contours) >= min_panels:
        panels_area = sum(w * h for _, _, w, h in panels_contours)
        coverage = panels_area / total_area

        def rectangles_overlap(r1, r2):
            x1, y1, w1, h1 = r1
            x2, y2, w2, h2 = r2
            return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

        overlaps = 0
        for i in range(len(panels_contours)):
            for j in range(i + 1, len(panels_contours)):
                if rectangles_overlap(panels_contours[i], panels_contours[j]):
                    overlaps += 1

        if 0.5 < coverage < 1.5 and overlaps < len(panels_contours) // 2:
            return True, panels_contours, "contours"

    # Método 3: Líneas (fallback)
    panels_lines = detect_collage_by_lines(
        image,
        threshold=COLLAGE_HOUGH_THRESHOLD,
        min_line_length=COLLAGE_HOUGH_MIN_LINE_LENGTH,
        max_line_gap=COLLAGE_HOUGH_MAX_LINE_GAP
    )

    if len(panels_lines) >= min_panels:
        panels_area = sum(w * h for _, _, w, h in panels_lines)
        coverage = panels_area / total_area

        if 0.85 < coverage < 1.15:
            return True, panels_lines, "hough_lines"

    return False, [], "none"


def extract_and_save_panels(image_path: Path, panels: list, output_dir: Path) -> list:
    """
    Extrae y guarda subimágenes según coordenadas de paneles.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    panel_paths = []
    stem = image_path.stem
    ext = image_path.suffix

    for i, (x, y, w, h) in enumerate(panels, 1):
        panel = image[y:y+h, x:x+w]

        panel_filename = f"{stem}_panel_{i}{ext}"
        panel_path = output_dir / panel_filename
        cv2.imwrite(str(panel_path), panel)
        panel_paths.append(panel_path)

    return panel_paths


def visualize_collage_detection(image_path: Path, panels: list, method: str, output_dir: Path) -> Path:
    """
    Crea visualización del collage completo con paneles marcados.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, (x, y, w, h) in enumerate(panels, 1):
        color = colors[(i - 1) % len(colors)]

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 4)

        label = f"Panel {i}"
        font_scale = max(0.8, min(w, h) / 300)
        thickness = max(2, int(font_scale * 2))

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x, y), (x + text_w + 10, y + text_h + 10), color, -1)
        cv2.putText(image, label, (x + 5, y + text_h + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        dims_text = f"{w}x{h}"
        cv2.putText(image, dims_text, (x + 5, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, thickness)

    info_text = f"Collage: {len(panels)} panels | Method: {method}"
    cv2.putText(image, info_text, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    stem = image_path.stem
    output_path = output_dir / f"{stem}_collage_detection.jpg"
    cv2.imwrite(str(output_path), image)

    return output_path


def create_panels_montage(original_image_path: Path, panel_paths: list,
                          panels_coords: list, output_dir: Path) -> Path:
    """
    Crea un montaje mostrando la imagen original y todos los paneles procesados.
    """
    original = cv2.imread(str(original_image_path))
    if original is None:
        return None

    annotated_panels = []
    for panel_path in panel_paths:
        annotated_path = output_dir / f"{panel_path.stem}_annotated.jpg"
        if annotated_path.exists():
            panel = cv2.imread(str(annotated_path))
            if panel is not None:
                annotated_panels.append(panel)

    if not annotated_panels:
        return None

    orig_h, orig_w = original.shape[:2]

    n_panels = len(annotated_panels)
    cols = int(np.ceil(np.sqrt(n_panels)))
    rows = int(np.ceil(n_panels / cols))

    max_panel_w = max(p.shape[1] for p in annotated_panels)
    max_panel_h = max(p.shape[0] for p in annotated_panels)

    montage_w = max(orig_w, cols * max_panel_w)
    montage_h = orig_h + rows * max_panel_h + 50

    montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 255

    x_offset = (montage_w - orig_w) // 2
    montage[0:orig_h, x_offset:x_offset+orig_w] = original

    cv2.line(montage, (0, orig_h + 25), (montage_w, orig_h + 25), (0, 0, 0), 3)

    y_start = orig_h + 50
    for i, panel in enumerate(annotated_panels):
        row = i // cols
        col = i % cols

        ph, pw = panel.shape[:2]
        y = y_start + row * max_panel_h
        x = col * max_panel_w

        x_offset_panel = (max_panel_w - pw) // 2
        y_offset_panel = (max_panel_h - ph) // 2

        montage[y+y_offset_panel:y+y_offset_panel+ph,
                x+x_offset_panel:x+x_offset_panel+pw] = panel

        label = f"Panel {i+1}"
        cv2.putText(montage, label, (x + 10, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    stem = original_image_path.stem
    output_path = output_dir / f"{stem}_collage_montage.jpg"
    cv2.imwrite(str(output_path), montage)

    return output_path


# ============================================================================
# PROCESAMIENTO DE IMAGEN
# ============================================================================

def process_image(image_path: Path, model_detect, model_pose,
                  output_dir: Path, beauty_estimator=None, gender_classifier=None,
                  age_classifier=None, behaviour_classifier=None,
                  activity_classifier=None, body_display_classifier=None,
                  location_classifier=None, body_shape_classifier=None,
                  accessory_classifier=None, social_distance_classifier=None,
                  person_attributes_classifier=None, scene_context_classifier=None) -> dict:
    """
    Procesa una imagen con todas las funcionalidades activadas.
    """
    from ui import info, warn
    info(f"[bold]{image_path.name}[/]  [dim]→ leyendo imagen")

    img = cv2.imread(str(image_path))
    if img is None:
        warn(f"  Error al leer imagen: {image_path}")
        return None

    h, w = img.shape[:2]

    _use_pose_detector = getattr(config, "USE_POSE_AS_DETECTOR", False) and ENABLE_POSE_ESTIMATION

    _detect_kwargs = dict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )
    if _use_pose_detector:
        _detect_kwargs["imgsz"] = getattr(config, "YOLO_POSE_IMGSZ", 640)
    results_detect = model_detect.predict(**_detect_kwargs)

    results_pose = []
    if _use_pose_detector:
        # Pose-único: model_detect ES el modelo de pose → results_detect ya trae
        # boxes + keypoints alineados de UNA inferencia. Reusar (no re-inferir).
        results_pose = results_detect
    elif ENABLE_POSE_ESTIMATION:
        results_pose = model_pose.predict(
            source=img,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
    else:
        from ultralytics.engine.results import Results
        results_pose = [Results(orig_img=img, path=str(image_path), names={}, boxes=None, keypoints=None)]

    # Belleza inline solo si además está permitida en el path de IMAGEN
    # (ENABLE_BEAUTY_INLINE_IMAGES=False → la belleza la da SOLO el pase
    # diferido demand/*; evita puntuar 2 veces cada cara). Vídeo no pasa por aquí.
    _beauty_inline_ok = ENABLE_BEAUTY_ESTIMATION and getattr(
        config, "ENABLE_BEAUTY_INLINE_IMAGES", True)
    # SAVE_ANNOTATED_OUTPUTS=False (modo carpeta): no persistir nada — output_dir=None
    # suprime person_crops/ y person_crops_annotated/ dentro de annotate_frame, y abajo
    # se salta el imwrite de {stem}_annotated.jpg. La clasificación no cambia.
    _save_outputs = getattr(config, "SAVE_ANNOTATED_OUTPUTS", True)
    annotated, stats, _, _, _, _, _, _, _, _, _ = annotate_frame(
        img, results_detect, results_pose,
        with_tracking=False,
        beauty_estimator=beauty_estimator if _beauty_inline_ok else None,
        gender_classifier=gender_classifier if ENABLE_GENDER_CLASSIFICATION else None,
        age_classifier=age_classifier if ENABLE_AGE_CLASSIFICATION else None,
        behaviour_classifier=behaviour_classifier if ENABLE_BEHAVIOUR_CLASSIFICATION else None,
        activity_classifier=activity_classifier if ENABLE_ACTIVITY_CLASSIFICATION else None,
        body_display_classifier=body_display_classifier if ENABLE_BODY_DISPLAY_CLASSIFICATION else None,
        location_classifier=location_classifier if ENABLE_LOCATION_CLASSIFICATION else None,
        body_shape_classifier=body_shape_classifier if ENABLE_BODY_SHAPE_CLASSIFICATION else None,
        accessory_classifier=accessory_classifier if ENABLE_ACCESSORY_CLASSIFICATION else None,
        social_distance_classifier=social_distance_classifier if ENABLE_SOCIAL_DISTANCE else None,
        person_attributes_classifier=person_attributes_classifier,
        scene_context_classifier=scene_context_classifier,
        frame_idx=0,
        output_dir=output_dir if _save_outputs else None,
        scene_number=1,
        total_scenes=1,
        output_basename=image_path.stem,
    )

    # ── Ubicación de la ESCENA cuando NO hay personas (2026-08-31) ──────────
    # Sin detecciones el bucle por persona no corre y `location_classifications`
    # sale vacía → la fila del xlsx quedaba con `IA Ubic.` en blanco (main.py, rama
    # n == 0). Aquí se lanza UNA llamada sobre el frame COMPLETO con el prompt sin
    # TARGET. Va en `process_image` y NO en `annotate_frame` a propósito: `video.py`
    # llama a `annotate_frame` por frame con los clasificadores a None solo para
    # renderizar, y ahí no debe dispararse ninguna inferencia.
    if (stats['num_persons'] == 0
            and ENABLE_LOCATION_CLASSIFICATION
            and getattr(config, "ENABLE_SCENE_LOCATION_NO_PERSON", True)
            and scene_context_classifier is not None):
        scene_loc = scene_context_classifier.classify_location_only(img)
        if scene_loc and scene_loc.get("success"):
            stats['location_classifications'].append({
                "track_id": None,          # entrada de ESCENA, no de persona
                "frame": 0,
                "scene_level": True,
                "location": scene_loc.get("location"),
                "raw_response": scene_loc.get("raw_response"),
            })

    num_keypoints = 0
    if ENABLE_POSE_ESTIMATION and results_pose[0].keypoints is not None:
        kpts = results_pose[0].keypoints.data
        for person_kpts in kpts:
            visible = (person_kpts[:, 2] > VISUALIZATION["min_keypoint_conf"]).sum()
            num_keypoints += int(visible)

    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    if _save_outputs:
        cv2.imwrite(str(output_path), annotated)
    info(
        f"  [dim]{w}×{h}[/]  personas=[cyan]{stats['num_persons']}[/]"
        + (f"[dim]/{stats['num_persons_yolo']}yolo[/]" if stats['num_persons_yolo'] != stats['num_persons'] else "")
        + (f"  kpts=[cyan]{num_keypoints}[/]" if ENABLE_POSE_ESTIMATION else "")
        + (f"  [dim]→ {output_path.name}" if _save_outputs else "  [dim](sin anotada: SAVE_ANNOTATED_OUTPUTS=False)")
    )

    result_info = {
        "input_path": str(image_path),
        "output_path": str(output_path) if _save_outputs else None,
        "resolution": {"width": w, "height": h},
        "persons_detected": stats['num_persons'],
        "persons_detected_yolo": stats['num_persons_yolo'],
        "visible_keypoints": num_keypoints,
        "gender_classifications": stats['gender_classifications'],
        "age_classifications": stats['age_classifications'],
        "behaviour_classifications": stats['behaviour_classifications'],
        "activity_classifications": stats['activity_classifications'],
        "body_display_classifications": stats['body_display_classifications'],
        "location_classifications": stats['location_classifications'],
        "body_shape_classifications": stats['body_shape_classifications'],
        "accessory_classifications": stats['accessory_classifications'],
        "social_distance_classifications": stats['social_distance_classifications'],
        "beauty_scores": stats['beauty_scores'],
        "timestamp": datetime.now().isoformat()
    }

    n_persons = stats['num_persons']
    vlm_lists = [
        ('gender_classifications',          ENABLE_GENDER_CLASSIFICATION),
        ('age_classifications',             ENABLE_AGE_CLASSIFICATION),
        ('behaviour_classifications',       ENABLE_BEHAVIOUR_CLASSIFICATION),
        ('activity_classifications',        ENABLE_ACTIVITY_CLASSIFICATION),
        ('body_display_classifications',    ENABLE_BODY_DISPLAY_CLASSIFICATION),
        ('location_classifications',        ENABLE_LOCATION_CLASSIFICATION),
        ('body_shape_classifications',      ENABLE_BODY_SHAPE_CLASSIFICATION),
        ('accessory_classifications',       ENABLE_ACCESSORY_CLASSIFICATION),
        ('social_distance_classifications', ENABLE_SOCIAL_DISTANCE),
    ]
    n_enabled = sum(1 for _, enabled in vlm_lists if enabled)
    expected  = n_persons * n_enabled
    missing   = 0
    for key, enabled in vlm_lists:
        if not enabled:
            continue
        items = stats.get(key, []) or []
        valid = 0
        for it in items:
            raw = (it.get("raw_response") or "")
            if raw.startswith("keypoints:") or raw.startswith("shortcut:") or raw == "no_keypoints" or raw == "no_waist_keypoints":
                valid += 1
                continue
            # "body_weight" reemplaza a "silhouette" (eliminada 2026-07-04); sin él,
            # body_shape contaba SIEMPRE como fallo y todo run salía "partial".
            val_candidates = [it.get(k) for k in ("gender", "age_group", "behaviour", "activity", "body_display", "location", "body_weight", "category") if k in it]
            # Accesorios: la entrada no tiene clave de valor única, sino una clave 0/1
            # por categoría; su presencia significa que el parseo fue correcto.
            if not val_candidates and any(cat in it for cat in ACCESSORY_CATEGORIES):
                valid += 1
                continue
            val = val_candidates[0] if val_candidates else None
            if val in (None, "no visible", "na", ""):
                continue
            valid += 1
        missing += max(0, n_persons - valid)

    vlm_failure_rate = (missing / expected) if expected > 0 else 0.0
    threshold = getattr(config, "VLM_IMAGE_FAILURE_THRESHOLD", 0.5)
    if expected == 0:
        vlm_status = "ok"
    elif vlm_failure_rate >= threshold:
        vlm_status = "failed"
    elif missing > 0:
        vlm_status = "partial"
    else:
        vlm_status = "ok"

    result_info["vlm_failures"]     = missing
    result_info["vlm_attempts"]     = expected
    result_info["vlm_failure_rate"] = round(vlm_failure_rate, 3)
    result_info["vlm_status"]       = vlm_status

    if vlm_status != "ok":
        warn(f"  VLM {vlm_status}: {missing}/{expected} fallos ({vlm_failure_rate:.0%})")

    return result_info


def rerender_annotated_with_beauty(image_path: Path, output_dir: Path,
                                   summary_result: dict, beauty_by_track: dict,
                                   model_detect, model_pose,
                                   social_distance_classifier=None) -> bool:
    """Re-dibuja `{stem}_annotated.jpg` inyectando la belleza del pase DIFERIDO.

    En el path de IMAGEN la belleza es un pase diferido (demand/*) que corre DESPUÉS de que
    `process_image` ya guardó la imagen anotada → el chip de belleza nunca aparecía. Esta
    función re-renderiza la imagen SIN re-llamar al VLM: re-ejecuta solo YOLO (determinista →
    reproduce los mismos track_ids/orden en una imagen fija) y reconstruye todos los `*_info`
    desde el `summary_result` vía los caches de `annotate_frame`, con `beauty_cache`
    pre-rellenado desde `beauty_by_track` (track_id → score numérico).

    Reusa exactamente los mismos primitivos de dibujo → salida pixel-idéntica + el chip de
    belleza. Degrada con elegancia: sin scores o si no se puede releer la imagen → no toca nada
    y devuelve False.
    """
    # Solo scores numéricos (descarta None / not_calculable).
    scored = {t: float(s) for t, s in (beauty_by_track or {}).items()
              if isinstance(s, (int, float))}
    if not scored:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    _use_pose_detector = getattr(config, "USE_POSE_AS_DETECTOR", False) and ENABLE_POSE_ESTIMATION
    _detect_kwargs = dict(source=img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                          classes=[PERSON_CLASS_ID], verbose=False)
    if _use_pose_detector:
        _detect_kwargs["imgsz"] = getattr(config, "YOLO_POSE_IMGSZ", 640)
    results_detect = model_detect.predict(**_detect_kwargs)
    if _use_pose_detector:
        results_pose = results_detect
    elif ENABLE_POSE_ESTIMATION:
        results_pose = model_pose.predict(source=img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    else:
        from ultralytics.engine.results import Results
        results_pose = [Results(orig_img=img, path=str(image_path), names={}, boxes=None, keypoints=None)]

    def _by_track(lst):
        return {c["track_id"]: c for c in (lst or []) if c.get("track_id") is not None}

    g   = _by_track(summary_result.get("gender_classifications"))
    a   = _by_track(summary_result.get("age_classifications"))
    b   = _by_track(summary_result.get("behaviour_classifications"))
    act = _by_track(summary_result.get("activity_classifications"))
    bd  = _by_track(summary_result.get("body_display_classifications"))
    loc = _by_track(summary_result.get("location_classifications"))
    bs  = _by_track(summary_result.get("body_shape_classifications"))
    acc = _by_track(summary_result.get("accessory_classifications"))

    gender_cache       = {t: {"gender": c.get("gender"), "last_frame": 0, "raw_response": ""} for t, c in g.items()}
    age_cache          = {t: {"age_group": c.get("age_group"), "last_frame": 0, "raw_response": ""} for t, c in a.items()}
    behaviour_cache    = {t: {"behaviour": c.get("behaviour"), "last_frame": 0, "raw_response": c.get("raw_response", "")} for t, c in b.items()}
    activity_cache     = {t: {"activity": c.get("activity"), "last_frame": 0, "raw_response": c.get("raw_response", "")} for t, c in act.items()}
    body_display_cache = {t: {"body_display": c.get("body_display"), "last_frame": 0, "raw_response": c.get("raw_response", "")} for t, c in bd.items()}
    location_cache     = {t: {"location": c.get("location"), "last_frame": 0, "raw_response": c.get("raw_response", "")} for t, c in loc.items()}
    body_shape_cache   = {t: {"body_weight": c.get("body_weight"), "muscle": c.get("muscle"),
                              "attire": c.get("attire"), "last_frame": 0,
                              "raw_response": c.get("raw_response", "")} for t, c in bs.items()}
    accessory_cache = {}
    for t, c in acc.items():
        entry = {"last_frame": 0, "raw_response": c.get("raw_response", "")}
        for cat in ACCESSORY_CATEGORIES:
            entry[cat] = c.get(cat, 0)
        accessory_cache[t] = entry
    beauty_cache = {t: {"score": s, "last_frame": 0, "raw_response": ""} for t, s in scored.items()}

    # classifiers=None + caches llenos → annotate_frame NO llama al VLM, reconstruye desde cache.
    # beauty_estimator=None + beauty_cache → dibuja el score cacheado (solo demand/*).
    # social_distance_classifier es determinista por keypoints (sin coste VLM) → recomputa igual.
    annotated, *_ = annotate_frame(
        img, results_detect, results_pose,
        with_tracking=False,
        beauty_estimator=None,
        social_distance_classifier=social_distance_classifier if ENABLE_SOCIAL_DISTANCE else None,
        gender_cache=gender_cache, age_cache=age_cache,
        behaviour_cache=behaviour_cache, activity_cache=activity_cache,
        body_display_cache=body_display_cache, location_cache=location_cache,
        body_shape_cache=body_shape_cache, accessory_cache=accessory_cache,
        beauty_cache=beauty_cache,
        frame_idx=0, output_dir=None, scene_number=1, total_scenes=1,
        output_basename=image_path.stem,
    )
    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated)
    return True
