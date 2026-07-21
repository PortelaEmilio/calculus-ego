"""
Procesamiento de video: detección de escenas, análisis VLM por escena y process_video().
"""

import cv2
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    YOLO = None
    Results = None

try:
    from scenedetect import detect, ContentDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False

from config import (
    ENABLE_TRACKING, ENABLE_POSE_ESTIMATION, ENABLE_BEHAVIOUR_CLASSIFICATION,
    ENABLE_ACTIVITY_CLASSIFICATION, ENABLE_BODY_DISPLAY_CLASSIFICATION,
    ENABLE_LOCATION_CLASSIFICATION, ENABLE_BODY_SHAPE_CLASSIFICATION,
    ENABLE_ACCESSORY_CLASSIFICATION, ENABLE_OCR, ENABLE_GENDER_CLASSIFICATION,
    ENABLE_AGE_CLASSIFICATION, ENABLE_BEAUTY_ESTIMATION, ENABLE_SOCIAL_DISTANCE,
    ENABLE_SCENE_DETECTION, ENABLE_FACE_KEYPOINT_SCENE_DETECTION,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, PERSON_CLASS_ID,
    SCENE_CONTENT_THRESHOLD, SCENE_MIN_LEN, TRACKER_CONFIG,
    FACE_KEYPOINT_DISPLACEMENT_THRESHOLD, FACE_KEYPOINT_MIN_SCENE_LEN,
    ENABLE_FACE_KEYPOINT_COUNT_SCENE_DETECTION,
    FACE_KEYPOINT_COUNT_FRONTAL_MIN, FACE_KEYPOINT_COUNT_PROFILE_MAX,
    MAX_SCENE_FRAMES_IN_MEMORY, COLLAGE_MIN_PANELS, COLLAGE_MIN_PANEL_SIZE_PERCENT,
    VISUALIZATION, BEAUTY_MIN_FACE_KEYPOINTS,
    BEAUTY_SHARP_TIEBREAK_K, BEAUTY_MIN_SHARPNESS, BEAUTY_MIN_HEAD_PX,
)


def _fade_alpha_for(frames_into_scene: int, fps: int) -> float:
    """Alpha de fade-in (0-1) según cuántos frames llevamos dentro de la (sub)escena."""
    if not VISUALIZATION.get("fade_enabled", True):
        return 1.0
    fade_frames = max(1, round(fps * VISUALIZATION.get("fade_seconds", 0.3)))
    return min(1.0, max(0.0, (frames_into_scene + 1) / fade_frames))
from utils.visualization import (
    has_five_face_keypoints_visible, extract_face_crop, is_frontal_pose_with_waist,
    count_visible_keypoints, get_face_keypoint_centroid, put_text_pil, face_sharpness,
)
from processing.image import annotate_frame, detect_collage_panels


def combine_panel_videos_with_borders(video_path: Path, panels: list, panel_results: list,
                                      output_dir: Path, fps: int, width: int, height: int,
                                      total_frames: int) -> Path:
    """
    Combina todos los videos de paneles procesados en un único video con bordes de colores.
    """
    panel_colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
    ]

    from ui import info, warn, success
    info(f"  Combinando paneles ([cyan]{len(panels)}[/], {total_frames} frames)")

    output_path = output_dir / f"{video_path.stem}_collage_combined.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    panel_captures = []
    for panel_data in panel_results:
        annotated_video_path = output_dir / f"panel_{panel_data['panel_id']}_video_annotated.mp4"

        if annotated_video_path.exists():
            cap = cv2.VideoCapture(str(annotated_video_path))
            if cap.isOpened():
                panel_captures.append({
                    "cap": cap,
                    "panel_id": panel_data['panel_id'],
                    "coordinates": panel_data['coordinates'],
                    "color": panel_colors[(panel_data['panel_id'] - 1) % len(panel_colors)]
                })
            else:
                warn(f"  No se pudo abrir el video del panel {panel_data['panel_id']}")
        else:
            warn(f"  No se encontró video del panel {panel_data['panel_id']}: {annotated_video_path}")

    if not panel_captures:
        warn("  No se encontraron videos de paneles para combinar")
        return None

    border_thickness = 5
    frame_idx = 0

    with tqdm(total=total_frames, desc="  Combinando frames", unit="frame") as pbar:
        while frame_idx < total_frames:
            combined_frame = np.zeros((height, width, 3), dtype=np.uint8)

            all_panels_ok = True
            for panel_info in panel_captures:
                ret, panel_frame = panel_info["cap"].read()

                if not ret:
                    all_panels_ok = False
                    break

                x = panel_info["coordinates"]["x"]
                y = panel_info["coordinates"]["y"]
                w = panel_info["coordinates"]["width"]
                h = panel_info["coordinates"]["height"]

                if panel_frame.shape[:2] != (h, w):
                    panel_frame = cv2.resize(panel_frame, (w, h))

                combined_frame[y:y+h, x:x+w] = panel_frame

                color = panel_info["color"]
                cv2.rectangle(combined_frame, (x, y), (x + w, y + h), color, border_thickness)

                label = f"Panel {panel_info['panel_id']}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                text_x = x + 10
                text_y = y + 30

                cv2.rectangle(
                    combined_frame,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + baseline),
                    color,
                    -1
                )

                put_text_pil(
                    img=combined_frame,
                    text=label,
                    org=(text_x, text_y),
                    font_size=26,
                    color=(255, 255, 255),
                    thickness=font_thickness
                )

            if not all_panels_ok:
                break

            writer.write(combined_frame)
            frame_idx += 1
            pbar.update(1)

    for panel_info in panel_captures:
        panel_info["cap"].release()
    writer.release()

    success(f"  Video combinado: [dim]{output_path.name}")

    return output_path


def create_video_with_colored_panels(video_path: Path, panels: list, output_dir: Path) -> Path:
    """
    Crea un video mostrando el video completo con los paneles marcados con bordes de colores.
    """
    from ui import info, warn, success
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warn("  Error al abrir video para visualización")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / f"{video_path.stem}_panels_visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    info(f"  Marcando paneles ([cyan]{len(panels)}[/], {total_frames} frames)")

    border_thickness = 5

    with tqdm(total=total_frames, desc="  Procesando frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for idx, (x, y, w, h) in enumerate(panels):
                color = colors[idx % len(colors)]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, border_thickness)

                label = f"Panel {idx + 1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                text_x = x + 10
                text_y = y + 30

                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + baseline),
                    color,
                    -1
                )

                put_text_pil(
                    img=frame,
                    text=label,
                    org=(text_x, text_y),
                    font_size=26,
                    color=(255, 255, 255),
                    thickness=font_thickness
                )

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()

    success(f"  Video con paneles: [dim]{output_path.name}")

    return output_path


def process_video_with_panels(video_path: Path, panels: list, model_detect, model_pose,
                              output_dir: Path, beauty_estimator=None,
                              gender_classifier=None, age_classifier=None,
                              behaviour_classifier=None, activity_classifier=None,
                              body_display_classifier=None, location_classifier=None,
                              body_shape_classifier=None, social_distance_classifier=None,
                              person_attributes_classifier=None, scene_context_classifier=None) -> dict:
    """
    Procesa un video con collages dividiendo primero por escenas y luego por paneles.
    """
    from ui import header, info, warn, success
    info(f"[bold]{video_path.name}[/]  [dim]→ video con collage ({len(panels)} paneles)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warn("  Error al abrir video")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    info(f"  [dim]{width}×{height}  {fps} fps  {duration_sec:.1f}s")

    header("Detección de escenas")

    scene_change_frames = detect_scene_changes(video_path)
    if not scene_change_frames:
        scene_change_frames = [0, total_frames // 2, max(0, total_frames - 1)]
        warn("  Sin cambios de escena, usando fallback")

    scenes = []
    for i in range(len(scene_change_frames)):
        start_frame = scene_change_frames[i]
        end_frame = scene_change_frames[i + 1] if i + 1 < len(scene_change_frames) else total_frames
        scenes.append({
            "scene_number": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": end_frame - start_frame
        })

    info(f"  Escenas detectadas: [cyan]{len(scenes)}")

    header("Procesamiento por escena y panel")

    scene_results = []

    for scene_info in scenes:
        scene_num = scene_info["scene_number"]
        start_frame = scene_info["start_frame"]
        end_frame = scene_info["end_frame"]

        info(f"\n  [bold]Escena {scene_num}/{len(scenes)}[/]  [dim]frames {start_frame}-{end_frame}")

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            warn(f"  No se pudo leer el primer frame de la escena {scene_num}")
            continue

        temp_frame_path = output_dir / f"temp_scene{scene_num}_frame{start_frame}.jpg"
        cv2.imwrite(str(temp_frame_path), first_frame)

        is_collage, scene_panels, method = detect_collage_panels(
            temp_frame_path,
            min_panels=COLLAGE_MIN_PANELS,
            min_panel_size_percent=COLLAGE_MIN_PANEL_SIZE_PERCENT,
            model_detect=model_detect
        )

        if temp_frame_path.exists():
            temp_frame_path.unlink()

        if not is_collage or len(scene_panels) == 0:
            scene_panels = panels
            method = "initial_detection"
            info(f"  [dim]paneles: {len(scene_panels)} (iniciales)")
        else:
            info(f"  [dim]paneles: {len(scene_panels)} ({method})")

        panel_results = []

        for panel_idx, (x, y, w, h) in enumerate(scene_panels, 1):
            info(f"\n  [dim]── Escena {scene_num} · Panel {panel_idx}/{len(scene_panels)}  ({x},{y} {w}×{h})")

            temp_panel_dir = output_dir / f"temp_scene{scene_num}_panel{panel_idx}_{video_path.stem}"
            temp_panel_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            panel_video_path = temp_panel_dir / f"scene{scene_num}_panel{panel_idx}_video.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            panel_writer = cv2.VideoWriter(str(panel_video_path), fourcc, fps, (w, h))

            frame_idx = start_frame
            frames_written = 0

            with tqdm(total=end_frame - start_frame, desc=f"     Extrayendo panel {panel_idx}", unit="frame") as pbar:
                while frame_idx < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    panel_frame = frame[y:y+h, x:x+w]
                    panel_writer.write(panel_frame)

                    frame_idx += 1
                    frames_written += 1
                    pbar.update(1)

            cap.release()
            panel_writer.release()

            info(f"     [dim]extraídos {frames_written} frames · procesando…")

            panel_result = process_video(
                panel_video_path, model_detect, model_pose, output_dir,
                beauty_estimator, gender_classifier, age_classifier,
                behaviour_classifier, activity_classifier, body_display_classifier,
                location_classifier, body_shape_classifier, None, None, social_distance_classifier
            )

            if panel_result:
                panel_results.append({
                    "panel_id": panel_idx,
                    "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "result": panel_result
                })


        scene_results.append({
            "scene_number": scene_num,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": end_frame - start_frame,
            "detection_method": method,
            "total_panels": len(scene_panels),
            "panels": panel_results
        })

    success(
        f"  Procesamiento completado: [cyan]{len(scenes)}[/] escenas · "
        f"[cyan]{sum(len(s['panels']) for s in scene_results)}[/] paneles"
    )

    return {
        "video_path": str(video_path),
        "collage_detected": True,
        "total_scenes": len(scenes),
        "scenes": scene_results,
        "duration_seconds": duration_sec,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": {"width": width, "height": height}
    }


def detect_face_keypoint_changes_in_scene(frames_data: list, scene_start: int = 0) -> list:
    """
    Detecta cortes de sub-escena dentro de una escena ya escaneada, a partir de los
    keypoints faciales por track. Dos criterios (unión), cada uno con su flag:

    1. DESPLAZAMIENTO del centroide facial (ENABLE_FACE_KEYPOINT_SCENE_DETECTION):
       corte cuando el centroide de la cara salta > umbral entre frames consecutivos.
    2. Nº de keypoints faciales visibles (ENABLE_FACE_KEYPOINT_COUNT_SCENE_DETECTION):
       corte cuando la cara cruza frontal(≥FRONTAL_MIN) ↔ perfil/espaldas(≤PROFILE_MAX).
       La persona gira la cara → probable cambio de comportamiento → re-clasificar.

    Devuelve índices LOCALES de corte (respetando FACE_KEYPOINT_MIN_SCENE_LEN).
    """
    if not ENABLE_POSE_ESTIMATION:
        return []
    if not (ENABLE_FACE_KEYPOINT_SCENE_DETECTION
            or ENABLE_FACE_KEYPOINT_COUNT_SCENE_DETECTION):
        return []
    if not frames_data:
        return []

    # Por track: lista de (local_idx, centroid|None, bbox_width, n_face). Se registra
    # también n_face=0 (perfil/espaldas): son justo las transiciones que interesan al
    # criterio 2, mientras que el centroide es None ahí (el criterio 1 las ignora).
    face_tracks = defaultdict(list)

    for local_idx, entry in enumerate(frames_data):
        if len(entry) == 4:
            _, results_detect, results_pose, keypoints_list = entry
        elif len(entry) == 3:
            results_detect, results_pose, keypoints_list = entry
        else:
            continue

        if results_detect[0].boxes is None or len(results_detect[0].boxes) == 0:
            continue

        boxes = results_detect[0].boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None and ENABLE_TRACKING else None

        person_idx = 0
        for idx, box in enumerate(boxes):
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_width = max(x2 - x1, 1)
            track_id = int(ids[idx]) if ids is not None else f"det_{local_idx}_{idx}"

            if person_idx < len(keypoints_list):
                kp = keypoints_list[person_idx]
                n_face = sum(1 for ki in range(5)
                             if ki < len(kp) and float(kp[ki][2]) > 0.5)
                centroid = get_face_keypoint_centroid(kp)
                face_tracks[track_id].append((local_idx, centroid, bbox_width, n_face))
            person_idx += 1

    change_frames = set()
    for track_id, track_data in face_tracks.items():
        if len(track_data) < 2:
            continue
        for i in range(1, len(track_data)):
            prev_idx, prev_c, prev_bw, prev_n = track_data[i - 1]
            curr_idx, curr_c, curr_bw, curr_n = track_data[i]
            if curr_idx - prev_idx > 5:
                continue

            # Criterio 1 — desplazamiento del centroide (necesita ambos centroides).
            if (ENABLE_FACE_KEYPOINT_SCENE_DETECTION
                    and prev_c is not None and curr_c is not None):
                avg_bw = (prev_bw + curr_bw) / 2
                disp = float(np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1]))
                if disp / avg_bw > FACE_KEYPOINT_DISPLACEMENT_THRESHOLD:
                    change_frames.add(curr_idx)

            # Criterio 2 — cruce frontal ↔ perfil por Nº de keypoints faciales.
            if ENABLE_FACE_KEYPOINT_COUNT_SCENE_DETECTION:
                prev_frontal = prev_n >= FACE_KEYPOINT_COUNT_FRONTAL_MIN
                curr_frontal = curr_n >= FACE_KEYPOINT_COUNT_FRONTAL_MIN
                prev_profile = prev_n <= FACE_KEYPOINT_COUNT_PROFILE_MAX
                curr_profile = curr_n <= FACE_KEYPOINT_COUNT_PROFILE_MAX
                if (prev_frontal and curr_profile) or (prev_profile and curr_frontal):
                    change_frames.add(curr_idx)

    if not change_frames:
        return []

    total_len = len(frames_data)
    sorted_changes = sorted(change_frames)
    filtered = []
    last_break = 0
    for cf in sorted_changes:
        if (cf - last_break >= FACE_KEYPOINT_MIN_SCENE_LEN and
                total_len - cf >= FACE_KEYPOINT_MIN_SCENE_LEN):
            filtered.append(cf)
            last_break = cf

    if filtered:
        from ui import info as _info
        global_breaks = [scene_start + cf for cf in filtered]
        _info(f"  [dim]+{len(filtered)} corte(s) de sub-escena por keypoints faciales: {global_breaks}")
    return filtered


def detect_scene_changes(video_path: Path) -> list:
    if not PYSCENEDETECT_AVAILABLE or not ENABLE_SCENE_DETECTION:
        return []

    from ui import info as _info, warn as _warn
    try:
        scene_list = detect(str(video_path), ContentDetector(threshold=SCENE_CONTENT_THRESHOLD, min_scene_len=SCENE_MIN_LEN))

        if not scene_list:
            _warn("  No se detectaron cambios de escena")
            return []

        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        cap.release()

        scene_change_frames = [0]

        for i, scene in enumerate(scene_list):
            if i > 0:
                frame_num = scene[0].get_frames()
                if frame_num > 0 and frame_num not in scene_change_frames:
                    scene_change_frames.append(frame_num)

        scene_change_frames = sorted(set(scene_change_frames))
        _info(f"  [dim]{len(scene_list)} escena(s) · {len(scene_change_frames)} cortes")
        return scene_change_frames

    except Exception as e:
        _warn(f"  Error en detección de escenas: {e}")
        return []


def _update_beauty_candidate(cand: dict, frame, kpts, bbox: tuple, local_idx: int):
    """Actualiza `cand['beauty']` con el mejor frame de cara para puntuar belleza.

    Criterio de selección (compartido por el pre-escaneo buffer y el streaming):
      score = n_face + sharp/(sharp + K)
    donde `n_face` = nº de keypoints faciales visibles (3-5) y `sharp` = nitidez
    (var. del Laplaciano del crop de cara). La FRONTALIDAD (n_face, entero) es
    primaria; la NITIDEZ entra squasheada a [0,1) como desempate FUERTE entre
    frames de igual frontalidad → evita elegir un frame con desenfoque de
    movimiento (los keypoints se detectan con alta confianza aunque la cara esté
    movida, así que antes el desempate por confianza media no distinguía el blur).
    Si `BEAUTY_MIN_SHARPNESS > 0`, descarta el crop cuando la nitidez cae por debajo.
    """
    face_confs = [float(kpts[ki][2]) for ki in range(5)
                  if ki < len(kpts) and float(kpts[ki][2]) > 0.5]
    n_face = len(face_confs)
    if n_face < BEAUTY_MIN_FACE_KEYPOINTS:
        return

    face_crop = extract_face_crop(frame, kpts, bbox,
                                  min_points=BEAUTY_MIN_FACE_KEYPOINTS)
    if face_crop is None or face_crop.size == 0:
        return

    # Gate de tamaño mínimo de cabeza (anclado a los datasets de belleza): por debajo
    # la cara es demasiado pequeña para el estimador (thumbnail 672 sin ampliar → ancla
    # la nota bajo) → se descarta. Ver config.BEAUTY_MIN_HEAD_PX.
    if min(face_crop.shape[0], face_crop.shape[1]) < BEAUTY_MIN_HEAD_PX:
        return

    sharp = face_sharpness(face_crop)
    if sharp < BEAUTY_MIN_SHARPNESS:
        return

    sel_score = n_face + sharp / (sharp + BEAUTY_SHARP_TIEBREAK_K)
    current = cand.get('beauty')
    if current is None or sel_score > current['score']:
        cand['beauty'] = {
            'face_crop': face_crop.copy(),
            'score': sel_score,
            'frame_local_idx': local_idx,
            'sharpness': round(sharp, 1),
            'n_face': n_face,
        }


def update_scene_best_candidates(best_candidates: dict, frame, results_detect, results_pose,
                                  keypoints_list: list, local_idx: int):
    """
    Actualiza best_candidates procesando un solo frame (versión streaming).
    """
    if results_detect[0].boxes is None or len(results_detect[0].boxes) == 0:
        return

    boxes = results_detect[0].boxes
    ids = boxes.id.cpu().numpy() if boxes.id is not None and ENABLE_TRACKING else None

    person_idx = 0
    for idx, box in enumerate(boxes):
        if int(box.cls) != PERSON_CLASS_ID:
            continue

        if ids is not None:
            track_id = int(ids[idx])
        else:
            track_id = f"det_{local_idx}_{idx}"

        if track_id not in best_candidates:
            best_candidates[track_id] = {'general': None, 'beauty': None, 'body_shape': None}

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        bbox_area = (x2 - x1) * (y2 - y1)

        general_score = confidence * bbox_area
        current_general = best_candidates[track_id]['general']
        if current_general is None or general_score > current_general['score']:
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size > 0:
                best_candidates[track_id]['general'] = {
                    'crop': person_crop.copy(),
                    'score': general_score,
                    'frame_local_idx': local_idx,
                    'bbox': (x1, y1, x2, y2),
                    'kpts': keypoints_list[person_idx] if person_idx < len(keypoints_list) else None,
                }

        if ENABLE_BEAUTY_ESTIMATION and person_idx < len(keypoints_list):
            _update_beauty_candidate(best_candidates[track_id], frame,
                                     keypoints_list[person_idx], (x1, y1, x2, y2),
                                     local_idx)

        if ENABLE_BODY_SHAPE_CLASSIFICATION and person_idx < len(keypoints_list):
            kpts = keypoints_list[person_idx]
            if is_frontal_pose_with_waist(kpts):
                num_visible_kpts = count_visible_keypoints(kpts)
                body_score = num_visible_kpts / bbox_area if bbox_area > 0 else 0

                current_body = best_candidates[track_id]['body_shape']
                if current_body is None or body_score > current_body['score']:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:
                        best_candidates[track_id]['body_shape'] = {
                            'crop': person_crop.copy(),
                            'score': body_score,
                            'frame_local_idx': local_idx,
                        }

        person_idx += 1


def prescan_scene_for_best_frames(scene_frames_data: list, scene_num: int) -> dict:
    """
    Analiza los frames almacenados de una escena para encontrar el mejor frame
    por persona por función VLM.
    """
    best_candidates = defaultdict(lambda: {
        'general': None,
        'beauty': None,
        'body_shape': None,
    })

    for local_idx, (frame, results_detect, results_pose, keypoints_list) in enumerate(scene_frames_data):
        if results_detect[0].boxes is None or len(results_detect[0].boxes) == 0:
            continue

        boxes = results_detect[0].boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None and ENABLE_TRACKING else None

        person_idx = 0
        for idx, box in enumerate(boxes):
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            if ids is not None:
                track_id = int(ids[idx])
            else:
                track_id = f"det_{local_idx}_{idx}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            bbox_area = (x2 - x1) * (y2 - y1)

            general_score = confidence * bbox_area
            current_general = best_candidates[track_id]['general']
            if current_general is None or general_score > current_general['score']:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    best_candidates[track_id]['general'] = {
                        'crop': person_crop.copy(),
                        'score': general_score,
                        'frame_local_idx': local_idx,
                        'bbox': (x1, y1, x2, y2),
                        'kpts': keypoints_list[person_idx] if person_idx < len(keypoints_list) else None,
                    }

            if ENABLE_BEAUTY_ESTIMATION and person_idx < len(keypoints_list):
                _update_beauty_candidate(best_candidates[track_id], frame,
                                         keypoints_list[person_idx], (x1, y1, x2, y2),
                                         local_idx)

            if ENABLE_BODY_SHAPE_CLASSIFICATION and person_idx < len(keypoints_list):
                kpts = keypoints_list[person_idx]
                if is_frontal_pose_with_waist(kpts):
                    num_visible_kpts = count_visible_keypoints(kpts)
                    body_score = num_visible_kpts / bbox_area if bbox_area > 0 else 0

                    current_body = best_candidates[track_id]['body_shape']
                    if current_body is None or body_score > current_body['score']:
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size > 0:
                            best_candidates[track_id]['body_shape'] = {
                                'crop': person_crop.copy(),
                                'score': body_score,
                                'frame_local_idx': local_idx,
                            }

            person_idx += 1

    return dict(best_candidates)


def analyze_scene_vlm(best_candidates: dict, first_frame, scene_num: int,
                      scene_start_frame: int,
                      behaviour_classifier=None, activity_classifier=None,
                      body_display_classifier=None, location_classifier=None,
                      body_shape_classifier=None, accessory_classifier=None,
                      ocr_classifier=None, beauty_estimator=None,
                      gender_classifier=None, age_classifier=None,
                      person_attributes_classifier=None,
                      scene_context_classifier=None) -> tuple:
    """
    Ejecuta el análisis VLM una sola vez por persona por función en la escena.
    """
    behaviour_cache = {}
    activity_cache = {}
    body_display_cache = {}
    location_cache = {}
    body_shape_cache = {}
    accessory_cache = {}
    beauty_cache = {}
    gender_cache = {}
    age_cache = {}

    classifications = {
        'behaviour': [],
        'activity': [],
        'body_display': [],
        'location': [],
        'body_shape': [],
        'accessory': [],
        'beauty': [],
        'gender': [],
        'age': [],
    }

    # OCR en el primer frame de la escena
    ocr_result = None
    if ENABLE_OCR and ocr_classifier is not None and first_frame is not None:
        ocr_result = ocr_classifier.classify(first_frame)
        if not (ocr_result and ocr_result.get("success")):
            from ui import warn as _warn
            _warn(f"    OCR falló (escena {scene_num}): {ocr_result.get('error', 'Unknown error') if ocr_result else 'None'}")

    # Recopilar crops generales para batch
    persons_to_analyze = list(best_candidates.keys())
    general_crops = []
    general_track_ids = []
    for track_id in persons_to_analyze:
        general = best_candidates[track_id].get('general')
        if general and general.get('crop') is not None:
            general_crops.append(general['crop'])
            general_track_ids.append(track_id)

    use_merged = (
        person_attributes_classifier is not None
        and person_attributes_classifier.is_loaded()
    )

    if general_crops and use_merged:
        # ============================================================
        # Camino fusionado (Merge A + Merge B). Reduce 6+2 llamadas a
        # 2 llamadas VLM por escena. body_shape se gestiona aparte
        # porque en video usa crops especializados (pose frontal con
        # cintura visible) que difieren de los general_crops.
        # ============================================================
        n = len(general_crops)
        (
            merged_gender,
            merged_age,
            merged_behaviour,
            merged_body_display,
            merged_body_shape,
            merged_accessory,
        ) = person_attributes_classifier.classify_batch(
            general_crops,
            [False] * n,  # gate de SILUETA desactivado: la silueta se hace aparte
            # sobre body_crops. Pero peso (FRS/IMC) y musculatura viajan SIEMPRE en
            # el JSON de person-attrs (no dependen del gate) → los capturamos aquí.
        )

        # Peso + musculatura del JSON de person-attrs (la silueta llega None por el
        # gate off; se sobrescribe luego con el clasificador de silueta especializado).
        for track_id, result in zip(general_track_ids, merged_body_shape):
            if result and result.get('success'):
                entry = body_shape_cache.setdefault(track_id, {})
                entry['body_weight'] = result.get('body_weight')
                entry['muscle'] = result.get('muscle')
                # Vestimenta: el VLM decide directamente (gate de hombros ELIMINADO
                # 2026-07-08 — ver image.py / CLAUDE.md).
                entry['attire'] = result.get('attire')

        for track_id, result in zip(general_track_ids, merged_behaviour):
            if ENABLE_BEHAVIOUR_CLASSIFICATION and result and result.get('success'):
                frame_local = best_candidates[track_id]['general']['frame_local_idx']
                classifications['behaviour'].append({
                    'track_id': track_id,
                    'frame': scene_start_frame + frame_local,
                    'behaviour': result.get('behaviour'),
                    'raw_response': result.get('raw_response'),
                })
                behaviour_cache[track_id] = {
                    'behaviour': result.get('behaviour'),
                    'last_frame': scene_start_frame + frame_local,
                    'raw_response': result.get('raw_response'),
                }

        for track_id, result in zip(general_track_ids, merged_body_display):
            if ENABLE_BODY_DISPLAY_CLASSIFICATION and result and result.get('success'):
                frame_local = best_candidates[track_id]['general']['frame_local_idx']
                classifications['body_display'].append({
                    'track_id': track_id,
                    'frame': scene_start_frame + frame_local,
                    'body_display': result.get('body_display'),
                    'raw_response': result.get('raw_response'),
                })
                body_display_cache[track_id] = {
                    'body_display': result.get('body_display'),
                    'last_frame': scene_start_frame + frame_local,
                    'raw_response': result.get('raw_response'),
                }

        for track_id, result in zip(general_track_ids, merged_accessory):
            if ENABLE_ACCESSORY_CLASSIFICATION and result and result.get('success'):
                frame_local = best_candidates[track_id]['general']['frame_local_idx']
                classifications['accessory'].append({
                    'track_id': track_id,
                    'frame': scene_start_frame + frame_local,
                    'accessories': result,
                    'raw_response': result.get('raw_response'),
                })
                accessory_cache[track_id] = {
                    'accessories': result,
                    'last_frame': scene_start_frame + frame_local,
                    'raw_response': result.get('raw_response'),
                }

        for track_id, g_result, a_result in zip(general_track_ids, merged_gender, merged_age):
            frame_local = best_candidates[track_id]['general']['frame_local_idx']
            if ENABLE_GENDER_CLASSIFICATION and g_result and g_result.get('success'):
                classifications['gender'].append({
                    'track_id': track_id,
                    'frame': scene_start_frame + frame_local,
                    'gender': g_result.get('gender'),
                    'raw_response': g_result.get('raw_response'),
                })
                gender_cache[track_id] = {
                    'gender': g_result.get('gender'),
                    'last_frame': scene_start_frame + frame_local,
                    'raw_response': g_result.get('raw_response'),
                }
            if ENABLE_AGE_CLASSIFICATION and a_result and a_result.get('success'):
                classifications['age'].append({
                    'track_id': track_id,
                    'frame': scene_start_frame + frame_local,
                    'age_group': a_result.get('age_group'),
                    'raw_response': a_result.get('raw_response'),
                })
                age_cache[track_id] = {
                    'age_group': a_result.get('age_group'),
                    'last_frame': scene_start_frame + frame_local,
                    'raw_response': a_result.get('raw_response'),
                }

        # Merge B: activity + location
        if scene_context_classifier is not None and scene_context_classifier.is_loaded():
            # classify_batch devuelve 3 listas (activity, location, social_distance).
            # En vídeo la distancia social se calcula en la anotación por keypoints,
            # así que el tercer valor se ignora aquí.
            scene_act_results, scene_loc_results, _scene_sd_results = (
                scene_context_classifier.classify_batch(general_crops)
            )
            for track_id, result in zip(general_track_ids, scene_act_results):
                if ENABLE_ACTIVITY_CLASSIFICATION and result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['activity'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'activity': result.get('activity'),
                        'raw_response': result.get('raw_response'),
                    })
                    activity_cache[track_id] = {
                        'activity': result.get('activity'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }
            for track_id, result in zip(general_track_ids, scene_loc_results):
                if ENABLE_LOCATION_CLASSIFICATION and result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['location'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'location': result.get('location'),
                        'raw_response': result.get('raw_response'),
                    })
                    location_cache[track_id] = {
                        'location': result.get('location'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

    elif general_crops:
        # Behaviour
        if ENABLE_BEHAVIOUR_CLASSIFICATION and behaviour_classifier is not None:
            results = behaviour_classifier.classify_batch(general_crops)
            for track_id, result in zip(general_track_ids, results):
                if result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['behaviour'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'behaviour': result.get('behaviour'),
                        'raw_response': result.get('raw_response'),
                    })
                    behaviour_cache[track_id] = {
                        'behaviour': result.get('behaviour'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

        # Activity
        if ENABLE_ACTIVITY_CLASSIFICATION and activity_classifier is not None:
            results = activity_classifier.classify_batch(general_crops)
            for track_id, result in zip(general_track_ids, results):
                if result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['activity'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'activity': result.get('activity'),
                        'raw_response': result.get('raw_response'),
                    })
                    activity_cache[track_id] = {
                        'activity': result.get('activity'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

        # Body display
        if ENABLE_BODY_DISPLAY_CLASSIFICATION and body_display_classifier is not None:
            results = body_display_classifier.classify_batch(general_crops)
            for track_id, result in zip(general_track_ids, results):
                if result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['body_display'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'body_display': result.get('body_display'),
                        'raw_response': result.get('raw_response'),
                    })
                    body_display_cache[track_id] = {
                        'body_display': result.get('body_display'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

        # Location
        if ENABLE_LOCATION_CLASSIFICATION and location_classifier is not None:
            results = location_classifier.classify_batch(general_crops)
            for track_id, result in zip(general_track_ids, results):
                if result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['location'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'location': result.get('location'),
                        'raw_response': result.get('raw_response'),
                    })
                    location_cache[track_id] = {
                        'location': result.get('location'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

        # Accessory
        if ENABLE_ACCESSORY_CLASSIFICATION and accessory_classifier is not None:
            results = accessory_classifier.classify_batch(general_crops)
            for track_id, result in zip(general_track_ids, results):
                if result and result.get('success'):
                    frame_local = best_candidates[track_id]['general']['frame_local_idx']
                    classifications['accessory'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'accessories': result.get('accessories'),
                        'raw_response': result.get('raw_response'),
                    })
                    accessory_cache[track_id] = {
                        'accessories': result.get('accessories'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': result.get('raw_response'),
                    }

        # Gender/Age
        if (ENABLE_GENDER_CLASSIFICATION or ENABLE_AGE_CLASSIFICATION) and gender_classifier is not None:
            gender_results, age_results = gender_classifier.classify_batch(general_crops)
            for track_id, g_result, a_result in zip(general_track_ids, gender_results, age_results):
                frame_local = best_candidates[track_id]['general']['frame_local_idx']
                if g_result and g_result.get('success'):
                    classifications['gender'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'gender': g_result.get('gender'),
                        'raw_response': g_result.get('raw_response'),
                    })
                    gender_cache[track_id] = {
                        'gender': g_result.get('gender'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': g_result.get('raw_response'),
                    }
                if a_result and a_result.get('success'):
                    classifications['age'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + frame_local,
                        'age_group': a_result.get('age_group'),
                        'raw_response': a_result.get('raw_response'),
                    })
                    age_cache[track_id] = {
                        'age_group': a_result.get('age_group'),
                        'last_frame': scene_start_frame + frame_local,
                        'raw_response': a_result.get('raw_response'),
                    }

    # Silueta ELIMINADA (2026-07-04): el clasificador standalone de silueta ya no
    # se usa. Peso + musculatura se cachean desde el JSON de person-attrs (merge A).

    # Beauty (crops especiales: cara con 5 keypoints faciales)
    if ENABLE_BEAUTY_ESTIMATION and beauty_estimator is not None:
        beauty_crops_count = 0
        beauty_success_count = 0
        for track_id in persons_to_analyze:
            beauty = best_candidates[track_id].get('beauty')
            if beauty and beauty.get('face_crop') is not None:
                beauty_crops_count += 1
                result = beauty_estimator.estimate(beauty['face_crop'])
                if result and result.get('success'):
                    beauty_success_count += 1
                    classifications['beauty'].append({
                        'track_id': track_id,
                        'frame': scene_start_frame + beauty['frame_local_idx'],
                        'score': result.get('score'),
                        'raw_response': result.get('raw_response'),
                    })
                    beauty_cache[track_id] = {
                        'score': result.get('score'),
                        'last_frame': scene_start_frame + beauty['frame_local_idx'],
                        'raw_response': result.get('raw_response'),
                    }
                else:
                    beauty_cache[track_id] = {'score': None, 'not_calculable': True}
            else:
                beauty_cache[track_id] = {'score': None, 'not_calculable': True}


    caches = {
        'behaviour': behaviour_cache,
        'activity': activity_cache,
        'body_display': body_display_cache,
        'location': location_cache,
        'body_shape': body_shape_cache,
        'accessory': accessory_cache,
        'beauty': beauty_cache,
        'gender': gender_cache,
        'age': age_cache,
    }

    return caches, classifications, ocr_result


def _dump_scene_validation(validation_scenes, video_stem, output_dir,
                           scene_label, start_frame, end_frame,
                           best_candidates, caches, full_frame):
    """Persiste, para una (sub)escena, el frame completo limpio + el crop limpio de
    cada persona, y acumula un registro escena→personas con sus etiquetas IA. Usado
    SOLO cuando `process_video(..., validation_dump=True)` (validación de vídeo). No
    altera las listas planas del resultado; añade la clave `scenes`. La distancia
    social se rellena post-bucle (se calcula en la anotación, no en analyze_scene_vlm)."""
    from models.accessory import ACCESSORY_CATEGORIES

    crops_dir = output_dir / "person_crops"
    frames_dir = output_dir / "full_frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    safe_label = str(scene_label).replace(".", "_")

    frame_path = None
    if full_frame is not None and getattr(full_frame, "size", 0) > 0:
        fp = frames_dir / f"{video_stem}_scene_{safe_label}.jpg"
        cv2.imwrite(str(fp), full_frame)
        frame_path = str(fp)

    def _get(cache_name, track_id, field):
        entry = caches.get(cache_name, {}).get(track_id)
        return entry.get(field) if isinstance(entry, dict) else None

    persons = []
    for pidx, track_id in enumerate(best_candidates.keys()):
        general = best_candidates[track_id].get('general')
        if not general or general.get('crop') is None:
            continue
        cp = crops_dir / f"{video_stem}_scene_{safe_label}_person_{pidx}_track_{track_id}.jpg"
        cv2.imwrite(str(cp), general['crop'])

        acc_entry = caches.get('accessory', {}).get(track_id)
        acc_data = acc_entry.get('accessories') if isinstance(acc_entry, dict) else None
        accessories = {
            cat: (int(acc_data.get(cat, 0)) if isinstance(acc_data, dict) else None)
            for cat in ACCESSORY_CATEGORIES
        }

        bbox = general.get('bbox')
        persons.append({
            'track_id': track_id,
            'crop_path': str(cp),
            'bbox': [int(v) for v in bbox] if bbox else None,
            'gender':            _get('gender', track_id, 'gender'),
            'age_group':         _get('age', track_id, 'age_group'),
            'behaviour':         _get('behaviour', track_id, 'behaviour'),
            'activity':          _get('activity', track_id, 'activity'),
            'body_display':      _get('body_display', track_id, 'body_display'),
            'location':          _get('location', track_id, 'location'),
            'weight':            _get('body_shape', track_id, 'body_weight'),
            'muscle':            _get('body_shape', track_id, 'muscle'),
            'attire':            _get('body_shape', track_id, 'attire'),
            'accessories':       accessories,
            'beauty':            _get('beauty', track_id, 'score'),
            'social_distance':      None,   # rellenado post-bucle por (track_id, frame∈escena)
        })

    validation_scenes.append({
        'scene_label': str(scene_label),
        'start_frame': int(start_frame),
        'end_frame': int(end_frame),
        'frame_path': frame_path,
        'persons': persons,
    })


def _collect_beauty_pending(pending: list, best_candidates: dict, caches: dict,
                            scene_label, scene_start_frame: int):
    """Recoge, tras clasificar una (sub)escena, los crops de cara para el pase de
    belleza DIFERIDO del vídeo (se puntúa al final del vídeo, antes de la fase de
    anotación). 2026-07-21: se eliminó el gating demand/* — se puntúa TODA persona con
    candidato de belleza válido (≥ BEAUTY_MIN_FACE_KEYPOINTS keypoints faciales y cabeza
    ≥ BEAUTY_MIN_HEAD_PX; ambos gates ya aplicados en _update_beauty_candidate). Guarda
    una referencia al `beauty_cache` de la escena para inyectar el score después (el chip
    se dibuja desde ese caché en la anotación)."""
    for track_id, cand in best_candidates.items():
        beauty = cand.get('beauty')
        if not beauty or beauty.get('face_crop') is None:
            continue
        pending.append({
            'scene_label': scene_label,
            'track_id': track_id,
            'frame': scene_start_frame + beauty.get('frame_local_idx', 0),
            'face_crop': beauty['face_crop'],
            'beauty_cache': caches['beauty'],
        })


def _score_beauty_pending(pending: list, beauty_estimator) -> list:
    """FASE DIFERIDA de belleza del vídeo: puntúa los crops demand/* recogidos
    durante la clasificación e inyecta el score en el `beauty_cache` de su
    (sub)escena — la fase de anotación posterior dibuja el chip desde ahí.
    Devuelve la lista de clasificaciones [{track_id, scene, frame, score, …}].
    Degrada con elegancia: sin estimador / sin pendientes → lista vacía."""
    out = []
    if not pending or beauty_estimator is None:
        return out
    from ui import info
    info(f"  Belleza diferida (demand/*): {len(pending)} rostro(s)…")
    for p in pending:
        try:
            result = beauty_estimator.estimate(p['face_crop'])
        except Exception:
            result = None
        if result and result.get('success'):
            p['beauty_cache'][p['track_id']] = {
                'score': result.get('score'),
                'last_frame': p['frame'],
                'raw_response': result.get('raw_response'),
            }
            out.append({
                'track_id': p['track_id'],
                'scene': str(p['scene_label']),
                'frame': p['frame'],
                'score': result.get('score'),
                'raw_response': result.get('raw_response'),
            })
        else:
            p['beauty_cache'][p['track_id']] = {'score': None, 'not_calculable': True}
    return out


def process_video(video_path: Path, model_detect, model_pose,
                  output_dir: Path, beauty_estimator=None, gender_classifier=None,
                  age_classifier=None, behaviour_classifier=None,
                  activity_classifier=None, body_display_classifier=None,
                  location_classifier=None, body_shape_classifier=None,
                  accessory_classifier=None, ocr_classifier=None, social_distance_classifier=None,
                  person_attributes_classifier=None, scene_context_classifier=None,
                  validation_dump: bool = False) -> dict:
    """
    Procesa un video con todas las funcionalidades activadas.
    """
    from ui import info, warn, success
    info(f"[bold]{video_path.name}[/]  [dim]→ video"
         + ("  [tracking on]" if ENABLE_TRACKING else ""))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warn(f"  Error al abrir video: {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    info(f"  [dim]{width}×{height}  {fps} fps  {duration_sec:.1f}s ({total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output_dir / f"{video_path.stem}_annotated.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    track_history = defaultdict(int)
    unique_tracks = set()
    frame_detections = []
    all_beauty_scores = []
    all_gender_classifications = []
    all_age_classifications = []
    all_behaviour_classifications = []
    all_activity_classifications = []
    all_body_display_classifications = []
    all_location_classifications = []
    all_body_shape_classifications = []
    all_accessory_classifications = []
    all_social_distance_classifications = []
    validation_scenes = []  # solo se rellena si validation_dump=True

    # Belleza DIFERIDA (2026-07-10): la clasificación ya no puntúa belleza inline.
    # Se recogen los crops demand/* por (sub)escena y se puntúan al final, ANTES de
    # la fase de anotación (así el chip sale en el mp4 sin re-encodear).
    beauty_pending = []
    # Trabajos de render: la anotación se difiere al final (tras la belleza). Cada
    # entrada guarda la metadata YOLO por frame (orig_img=None) y los caches por
    # sub-escena de UNA escena; los frames se re-leen por seek en la fase de render.
    render_jobs = []

    scene_change_frames = detect_scene_changes(video_path)
    if not scene_change_frames:
        scene_change_frames = [0]

    scenes = []
    for i, start in enumerate(scene_change_frames):
        end = scene_change_frames[i + 1] if i + 1 < len(scene_change_frames) else total_frames
        scenes.append((start, end))

    info(f"  Escenas: [cyan]{len(scenes)}")

    ocr_results_by_scene = {}

    frame_idx = 0

    with tqdm(total=total_frames, desc="  Procesando", unit="frame") as pbar:
        for scene_idx, (scene_start, scene_end) in enumerate(scenes):
            scene_num = scene_idx + 1
            total_scenes_count = len(scenes)
            scene_frame_count = scene_end - scene_start

            use_streaming = scene_frame_count > MAX_SCENE_FRAMES_IN_MEMORY

            if use_streaming:
                warn(f"  Escena {scene_num}: {scene_frame_count} > {MAX_SCENE_FRAMES_IN_MEMORY} frames → streaming")

                best_candidates = defaultdict(lambda: {
                    'general': None, 'beauty': None, 'body_shape': None
                })
                scene_metadata = []
                first_frame_copy = None
                frames_read = 0

                # FASE 1-S: PRE-ESCANEO STREAMING
                for fi in range(scene_frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_read += 1

                    if fi == 0:
                        first_frame_copy = frame.copy()

                    if ENABLE_TRACKING:
                        results_detect = model_detect.track(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD,
                            classes=[PERSON_CLASS_ID],
                            persist=True,
                            tracker=TRACKER_CONFIG,
                            verbose=False
                        )
                    else:
                        results_detect = model_detect.predict(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD,
                            classes=[PERSON_CLASS_ID],
                            verbose=False
                        )

                    if ENABLE_POSE_ESTIMATION:
                        results_pose = model_pose.predict(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            verbose=False
                        )
                    else:
                        from ultralytics.engine.results import Results
                        results_pose = [Results(orig_img=frame, path="", names={}, boxes=None, keypoints=None)]

                    keypoints_list = []
                    if ENABLE_POSE_ESTIMATION and results_pose[0].keypoints is not None:
                        keypoints_data = results_pose[0].keypoints.data
                        keypoints_list = [kpts.cpu().numpy() for kpts in keypoints_data]

                    update_scene_best_candidates(
                        best_candidates, frame, results_detect,
                        results_pose, keypoints_list, fi
                    )

                    results_detect[0].orig_img = None
                    results_pose[0].orig_img = None

                    scene_metadata.append((results_detect, results_pose, keypoints_list))

                    del frame
                    pbar.update(1)

                if frames_read == 0:
                    continue

                best_candidates = dict(best_candidates)

                if best_candidates:
                    info(f"  [dim]Escena {scene_num}: {len(best_candidates)} persona(s)")

                # Detección de sub-escenas por keypoints faciales (modo streaming)
                face_kpt_breaks_s = detect_face_keypoint_changes_in_scene(scene_metadata, scene_start)
                sub_boundaries_s = [0] + face_kpt_breaks_s + [frames_read]

                if len(sub_boundaries_s) > 2:
                    info(f"  [dim]Escena {scene_num} → {len(sub_boundaries_s) - 1} sub-escena(s) por keypoints")

                del first_frame_copy, best_candidates

                sub_caches_list = []
                sub_scene_labels = []

                for sub_i in range(len(sub_boundaries_s) - 1):
                    sub_start_local = sub_boundaries_s[sub_i]
                    sub_end_local   = sub_boundaries_s[sub_i + 1]
                    sub_frame_count = sub_end_local - sub_start_local
                    sub_start_global = scene_start + sub_start_local
                    sub_scene_label = (f"{scene_num}.{sub_i + 1}"
                                       if len(sub_boundaries_s) > 2 else scene_num)
                    sub_scene_labels.append(sub_scene_label)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, sub_start_global)
                    sub_best = defaultdict(lambda: {'general': None, 'beauty': None, 'body_shape': None})
                    sub_first_frame = None

                    for fi in range(sub_frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        meta_idx = sub_start_local + fi
                        rd_s, rp_s, kl_s = scene_metadata[meta_idx]
                        if fi == 0:
                            sub_first_frame = frame.copy()
                        update_scene_best_candidates(sub_best, frame, rd_s, rp_s, kl_s, fi)
                        del frame

                    sub_best = dict(sub_best)
                    if sub_best:
                        info(f"  [dim]Escena {sub_scene_label}: {len(sub_best)} persona(s)")

                    sub_caches, sub_vlm_classifications, sub_ocr_result = analyze_scene_vlm(
                        sub_best, sub_first_frame, sub_scene_label,
                        scene_start_frame=sub_start_global,
                        behaviour_classifier=behaviour_classifier if ENABLE_BEHAVIOUR_CLASSIFICATION else None,
                        activity_classifier=activity_classifier if ENABLE_ACTIVITY_CLASSIFICATION else None,
                        body_display_classifier=body_display_classifier if ENABLE_BODY_DISPLAY_CLASSIFICATION else None,
                        location_classifier=location_classifier if ENABLE_LOCATION_CLASSIFICATION else None,
                        body_shape_classifier=body_shape_classifier if ENABLE_BODY_SHAPE_CLASSIFICATION else None,
                        accessory_classifier=accessory_classifier if ENABLE_ACCESSORY_CLASSIFICATION else None,
                        ocr_classifier=ocr_classifier,
                        beauty_estimator=None,  # belleza DIFERIDA (demand/*), ver _score_beauty_pending
                        gender_classifier=gender_classifier if ENABLE_GENDER_CLASSIFICATION else None,
                        age_classifier=age_classifier if ENABLE_AGE_CLASSIFICATION else None,
                        person_attributes_classifier=person_attributes_classifier,
                        scene_context_classifier=scene_context_classifier,
                    )
                    sub_caches_list.append(sub_caches)

                    if ENABLE_BEAUTY_ESTIMATION:
                        _collect_beauty_pending(beauty_pending, sub_best, sub_caches,
                                                sub_scene_label, sub_start_global)

                    if validation_dump:
                        _dump_scene_validation(
                            validation_scenes, video_path.stem, output_dir,
                            sub_scene_label, sub_start_global,
                            sub_start_global + sub_frame_count,
                            sub_best, sub_caches, sub_first_frame,
                        )

                    del sub_first_frame, sub_best

                    if sub_ocr_result and sub_ocr_result.get("success"):
                        ocr_results_by_scene[sub_scene_label] = {
                            "scene": sub_scene_label,
                            "frame": sub_start_global,
                            "text": sub_ocr_result.get("text"),
                            "raw_response": sub_ocr_result.get("raw_response")
                        }

                    all_behaviour_classifications.extend(sub_vlm_classifications['behaviour'])
                    all_activity_classifications.extend(sub_vlm_classifications['activity'])
                    all_body_display_classifications.extend(sub_vlm_classifications['body_display'])
                    all_location_classifications.extend(sub_vlm_classifications['location'])
                    all_body_shape_classifications.extend(sub_vlm_classifications['body_shape'])
                    all_accessory_classifications.extend(sub_vlm_classifications['accessory'])
                    all_gender_classifications.extend(sub_vlm_classifications['gender'])
                    all_age_classifications.extend(sub_vlm_classifications['age'])

                # ANOTACIÓN DIFERIDA: se registra el trabajo de render (la anotación
                # corre al final del vídeo, tras el pase de belleza, re-leyendo los
                # frames por seek). scene_metadata ya tiene orig_img=None (ligero).
                render_jobs.append({
                    'scene_start': scene_start,
                    'n_frames': frames_read,
                    'metadata': scene_metadata,
                    'sub_boundaries': sub_boundaries_s,
                    'sub_caches_list': sub_caches_list,
                    'sub_labels': sub_scene_labels,
                })

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                frame_idx = scene_start + scene_frame_count

            else:
                # MODO BUFFER — escenas normales
                scene_frames = []

                for fi in range(scene_frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if ENABLE_TRACKING:
                        results_detect = model_detect.track(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD,
                            classes=[PERSON_CLASS_ID],
                            persist=True,
                            tracker=TRACKER_CONFIG,
                            verbose=False
                        )
                    else:
                        results_detect = model_detect.predict(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD,
                            classes=[PERSON_CLASS_ID],
                            verbose=False
                        )

                    if ENABLE_POSE_ESTIMATION:
                        results_pose = model_pose.predict(
                            source=frame,
                            conf=CONFIDENCE_THRESHOLD,
                            verbose=False
                        )
                    else:
                        from ultralytics.engine.results import Results
                        results_pose = [Results(orig_img=frame, path="", names={}, boxes=None, keypoints=None)]

                    keypoints_list = []
                    if ENABLE_POSE_ESTIMATION and results_pose[0].keypoints is not None:
                        keypoints_data = results_pose[0].keypoints.data
                        keypoints_list = [kpts.cpu().numpy() for kpts in keypoints_data]

                    scene_frames.append((frame, results_detect, results_pose, keypoints_list))
                    pbar.update(1)

                if not scene_frames:
                    continue

                # Detección de sub-escenas por keypoints faciales
                face_kpt_breaks = detect_face_keypoint_changes_in_scene(scene_frames, scene_start)
                sub_boundaries = [0] + face_kpt_breaks + [len(scene_frames)]
                sub_scenes_list = [
                    scene_frames[sub_boundaries[i]:sub_boundaries[i + 1]]
                    for i in range(len(sub_boundaries) - 1)
                ]

                if len(sub_scenes_list) > 1:
                    info(f"  [dim]Escena {scene_num} → {len(sub_scenes_list)} sub-escena(s) por keypoints")

                sub_caches_list_b = []
                sub_labels_b = []

                for sub_idx, sub_frames in enumerate(sub_scenes_list):
                    if not sub_frames:
                        # mantener el alineamiento sub_boundaries ↔ caches en el render
                        sub_caches_list_b.append(None)
                        sub_labels_b.append(scene_num)
                        continue

                    sub_scene_label = (f"{scene_num}.{sub_idx + 1}"
                                       if len(sub_scenes_list) > 1 else scene_num)
                    sub_start_global = scene_start + sub_boundaries[sub_idx]

                    sub_best = prescan_scene_for_best_frames(sub_frames, sub_scene_label)

                    if sub_best:
                        info(f"  [dim]Escena {sub_scene_label}: {len(sub_best)} persona(s)")

                    first_frame_sub = sub_frames[0][0]

                    caches, vlm_classifications, ocr_result = analyze_scene_vlm(
                        sub_best, first_frame_sub, sub_scene_label,
                        scene_start_frame=sub_start_global,
                        behaviour_classifier=behaviour_classifier if ENABLE_BEHAVIOUR_CLASSIFICATION else None,
                        activity_classifier=activity_classifier if ENABLE_ACTIVITY_CLASSIFICATION else None,
                        body_display_classifier=body_display_classifier if ENABLE_BODY_DISPLAY_CLASSIFICATION else None,
                        location_classifier=location_classifier if ENABLE_LOCATION_CLASSIFICATION else None,
                        body_shape_classifier=body_shape_classifier if ENABLE_BODY_SHAPE_CLASSIFICATION else None,
                        accessory_classifier=accessory_classifier if ENABLE_ACCESSORY_CLASSIFICATION else None,
                        ocr_classifier=ocr_classifier,
                        beauty_estimator=None,  # belleza DIFERIDA (demand/*), ver _score_beauty_pending
                        gender_classifier=gender_classifier if ENABLE_GENDER_CLASSIFICATION else None,
                        age_classifier=age_classifier if ENABLE_AGE_CLASSIFICATION else None,
                        person_attributes_classifier=person_attributes_classifier,
                        scene_context_classifier=scene_context_classifier,
                    )

                    sub_caches_list_b.append(caches)
                    sub_labels_b.append(sub_scene_label)

                    if ENABLE_BEAUTY_ESTIMATION:
                        _collect_beauty_pending(beauty_pending, sub_best, caches,
                                                sub_scene_label, sub_start_global)

                    if ocr_result and ocr_result.get("success"):
                        ocr_results_by_scene[sub_scene_label] = {
                            "scene": sub_scene_label,
                            "frame": sub_start_global,
                            "text": ocr_result.get("text"),
                            "raw_response": ocr_result.get("raw_response")
                        }

                    all_behaviour_classifications.extend(vlm_classifications['behaviour'])
                    all_activity_classifications.extend(vlm_classifications['activity'])
                    all_body_display_classifications.extend(vlm_classifications['body_display'])
                    all_location_classifications.extend(vlm_classifications['location'])
                    all_body_shape_classifications.extend(vlm_classifications['body_shape'])
                    all_accessory_classifications.extend(vlm_classifications['accessory'])
                    all_gender_classifications.extend(vlm_classifications['gender'])
                    all_age_classifications.extend(vlm_classifications['age'])

                    if validation_dump:
                        _dump_scene_validation(
                            validation_scenes, video_path.stem, output_dir,
                            sub_scene_label, sub_start_global,
                            sub_start_global + len(sub_frames),
                            sub_best, caches, first_frame_sub,
                        )

                # ANOTACIÓN DIFERIDA: registrar el trabajo de render y liberar los
                # frames (la metadata YOLO se conserva ligera con orig_img=None; los
                # frames se re-leen por seek en la fase de render).
                scene_metadata_b = []
                for (frame_b, results_detect, results_pose, keypoints_list) in scene_frames:
                    results_detect[0].orig_img = None
                    results_pose[0].orig_img = None
                    scene_metadata_b.append((results_detect, results_pose, keypoints_list))

                render_jobs.append({
                    'scene_start': scene_start,
                    'n_frames': len(scene_frames),
                    'metadata': scene_metadata_b,
                    'sub_boundaries': sub_boundaries,
                    'sub_caches_list': sub_caches_list_b,
                    'sub_labels': sub_labels_b,
                })

                del scene_frames
                frame_idx = scene_start + scene_frame_count

    # ========================================================================
    # FASE DIFERIDA DE BELLEZA (solo demand/*, 2026-07-10) — se puntúa al final
    # de la clasificación, ANTES de la anotación, para que el chip 'Beauty X.X'
    # salga en el mp4 en una sola pasada de encode. Gating idéntico a imágenes.
    # ========================================================================
    import config as _config
    beauty_classifications = []
    if (beauty_pending and beauty_estimator is not None
            and getattr(_config, "ENABLE_BEAUTY_PASS", True)):
        beauty_classifications = _score_beauty_pending(beauty_pending, beauty_estimator)
    beauty_pending = None

    # Reflejar los scores diferidos en las escenas de validación (se volcaron con
    # beauty=None porque el dump ocurre durante la clasificación).
    if validation_dump and validation_scenes and beauty_classifications:
        _smap = {(c['scene'], c['track_id']): c['score'] for c in beauty_classifications}
        for _scene in validation_scenes:
            for _person in _scene['persons']:
                _key = (str(_scene['scene_label']), _person['track_id'])
                if _key in _smap:
                    _person['beauty'] = _smap[_key]

    # ========================================================================
    # FASE DE ANOTACIÓN (diferida): re-lectura de frames por seek + render con
    # los caches por (sub)escena (belleza ya inyectada). Sin llamadas VLM.
    # ========================================================================
    _empty_caches = {k: {} for k in ('behaviour', 'activity', 'body_display',
                                     'location', 'body_shape', 'accessory',
                                     'beauty', 'gender', 'age')}
    total_render_frames = sum(j['n_frames'] for j in render_jobs)
    with tqdm(total=total_render_frames, desc="  Anotando", unit="frame") as pbar:
        for job in render_jobs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, job['scene_start'])
            sb = job['sub_boundaries']

            for local_idx in range(job['n_frames']):
                ret, frame = cap.read()
                if not ret:
                    break

                results_detect, results_pose, keypoints_list = job['metadata'][local_idx]
                global_frame_idx = job['scene_start'] + local_idx

                sub_idx_for_frame = 0
                for si in range(len(sb) - 1):
                    if sb[si] <= local_idx < sb[si + 1]:
                        sub_idx_for_frame = si
                        break
                active_caches = job['sub_caches_list'][sub_idx_for_frame] or _empty_caches
                active_scene_label = job['sub_labels'][sub_idx_for_frame]
                fade_alpha = _fade_alpha_for(
                    local_idx - sb[sub_idx_for_frame], fps)

                annotated, stats, *_ = annotate_frame(
                    frame, results_detect, results_pose,
                    with_tracking=ENABLE_TRACKING,
                    beauty_estimator=None,
                    gender_classifier=None,
                    age_classifier=None,
                    behaviour_classifier=None,
                    activity_classifier=None,
                    body_display_classifier=None,
                    location_classifier=None,
                    body_shape_classifier=None,
                    accessory_classifier=None,
                    social_distance_classifier=social_distance_classifier if ENABLE_SOCIAL_DISTANCE else None,
                    frame_idx=global_frame_idx,
                    output_dir=output_dir,
                    behaviour_cache=active_caches['behaviour'],
                    activity_cache=active_caches['activity'],
                    body_display_cache=active_caches['body_display'],
                    location_cache=active_caches['location'],
                    body_shape_cache=active_caches['body_shape'],
                    accessory_cache=active_caches['accessory'],
                    beauty_cache=active_caches['beauty'],
                    gender_cache=active_caches['gender'],
                    age_cache=active_caches['age'],
                    scene_number=active_scene_label,
                    total_scenes=len(scenes),
                    fade_alpha=fade_alpha,
                )

                out.write(annotated)

                if ENABLE_TRACKING:
                    for tid in stats['track_ids']:
                        unique_tracks.add(tid)
                        track_history[tid] += 1

                frame_detections.append({
                    "frame": global_frame_idx,
                    "persons": stats['num_persons'],
                    "persons_yolo": stats.get('num_persons_yolo', stats['num_persons'])
                })

                all_beauty_scores.extend(stats['beauty_scores'])
                all_social_distance_classifications.extend(stats['social_distance_classifications'])

                del frame, annotated
                pbar.update(1)

            job['metadata'] = None
            gc.collect()

    render_jobs = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cap.release()
    out.release()

    avg_persons = np.mean([d["persons"] for d in frame_detections]) if frame_detections else 0
    max_persons = max([d["persons"] for d in frame_detections]) if frame_detections else 0

    # Estadísticas de belleza
    beauty_stats = None
    if all_beauty_scores:
        valid_scores = [
            s["score"] for s in all_beauty_scores
            if s["score"] is not None and s["score"] != "NA" and isinstance(s["score"], (int, float))
        ]
        if valid_scores:
            beauty_stats = {
                "count": len(valid_scores),
                "mean": round(np.mean(valid_scores), 2),
                "median": round(np.median(valid_scores), 2),
                "min": round(min(valid_scores), 2),
                "max": round(max(valid_scores), 2),
                "std": round(np.std(valid_scores), 2)
            }

    def _make_stats(items, key, label='track_id'):
        if not items:
            return None
        counts = Counter([x[key] for x in items])
        return {
            "total_classifications": len(items),
            f"{key}_distribution": dict(counts),
            "unique_tracks_classified": len(set([x[label] for x in items if x[label] is not None]))
        }

    gender_stats = _make_stats(all_gender_classifications, 'gender')
    behaviour_stats = _make_stats(all_behaviour_classifications, 'behaviour')
    activity_stats = _make_stats(all_activity_classifications, 'activity')
    body_display_stats = _make_stats(all_body_display_classifications, 'body_display')
    location_stats = _make_stats(all_location_classifications, 'location')
    body_shape_stats = _make_stats(all_body_shape_classifications, 'body_weight')
    age_stats = _make_stats(all_age_classifications, 'age_group')
    social_distance_stats = _make_stats(all_social_distance_classifications, 'category')

    accessory_stats = None
    if all_accessory_classifications:
        all_accessories_flat = []
        for acc in all_accessory_classifications:
            if acc["accessories"]:
                all_accessories_flat.extend(acc["accessories"])
        accessory_counts = Counter(all_accessories_flat)
        accessory_stats = {
            "total_classifications": len(all_accessory_classifications),
            "accessory_distribution": dict(accessory_counts),
            "unique_tracks_classified": len(set([a["track_id"] for a in all_accessory_classifications if a["track_id"] is not None]))
        }

    # Rellenar distancia social por escena: se calcula durante la anotación (no en
    # analyze_scene_vlm), así que se asocia post-hoc al primer match por track_id con
    # frame dentro del rango [start, end) de cada (sub)escena.
    if validation_dump and validation_scenes and all_social_distance_classifications:
        for scene in validation_scenes:
            s0, s1 = scene['start_frame'], scene['end_frame']
            for person in scene['persons']:
                tid = person['track_id']
                for sd in all_social_distance_classifications:
                    if sd.get('track_id') == tid and s0 <= sd.get('frame', -1) < s1:
                        person['social_distance'] = sd.get('category')
                        break

    success(
        f"  Video [dim]{output_path.name}[/]  "
        + (f"tracks=[cyan]{len(unique_tracks)}[/]  " if ENABLE_TRACKING else "")
        + f"avg/frame=[cyan]{avg_persons:.1f}[/]  max=[cyan]{max_persons}"
    )

    result_info = {
        "input_path": str(video_path),
        "output_path": str(output_path),
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "duration_seconds": duration_sec,
        "total_frames": total_frames,
        "unique_persons_tracked": len(unique_tracks) if ENABLE_TRACKING else None,
        "track_ids": list(unique_tracks) if ENABLE_TRACKING else None,
        "track_frame_counts": dict(track_history) if ENABLE_TRACKING else None,
        "avg_persons_per_frame": round(avg_persons, 2),
        "max_persons_per_frame": max_persons,
        "beauty_scores": all_beauty_scores,
        "beauty_statistics": beauty_stats,
        # Pase DIFERIDO de belleza (demand/*, 2026-07-10): una entrada por
        # (sub)escena × persona demand puntuada.
        "beauty_classifications": beauty_classifications,
        "gender_classifications": all_gender_classifications,
        "gender_statistics": gender_stats,
        "age_classifications": all_age_classifications,
        "age_statistics": age_stats,
        "behaviour_classifications": all_behaviour_classifications,
        "behaviour_statistics": behaviour_stats,
        "activity_classifications": all_activity_classifications,
        "activity_statistics": activity_stats,
        "body_display_classifications": all_body_display_classifications,
        "body_display_statistics": body_display_stats,
        "location_classifications": all_location_classifications,
        "location_statistics": location_stats,
        "body_shape_classifications": all_body_shape_classifications,
        "body_shape_statistics": body_shape_stats,
        "accessory_classifications": all_accessory_classifications,
        "accessory_statistics": accessory_stats,
        "ocr_results": dict(ocr_results_by_scene) if ocr_results_by_scene else None,
        "social_distance_classifications": all_social_distance_classifications,
        "social_distance_statistics": social_distance_stats,
        "scenes": validation_scenes if validation_dump else None,
        "timestamp": datetime.now().isoformat()
    }

    return result_info
