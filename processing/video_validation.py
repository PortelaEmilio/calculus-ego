"""
Pipeline RÁPIDO de validación de vídeo en DOS FASES (reemplaza el path de vídeo
de `manual`, que era ~25-30 min/vídeo).

- FASE A (`detect_video_scenes`): YOLO en GPU. Detecta escenas (+ sub-escenas por
  keypoints faciales), elige el mejor crop por persona y por (sub)escena y persiste
  a disco los crops + el frame de la escena + metadata (bbox, keypoints) en un
  `phase1_<id>.json`. NO llama al VLM ni renderiza el .mp4 anotado.
- FASE B (`categorize_video_scenes`): con YOLO ya liberado de la GPU y gemma4 cargado,
  reconstruye `best_candidates` desde los .jpg y reutiliza `analyze_scene_vlm()` SIN
  cambios para las 14 categorías, + distancia social UNA vez por escena
  (keypoint-gate + fallback VLM sobre el crop de la persona).

El orquestador (en `main.py._process_manual`) corre la Fase A de una tanda de N vídeos,
libera la GPU, y corre la Fase B de la tanda → 1 swap de modelo por tanda, no por vídeo.

El `result` que devuelve la Fase B trae la clave `scenes` con el MISMO esquema que
`processing/video._dump_scene_validation`, así que `main.py.build_video_rows` lo consume
sin cambios.
"""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from config import (
    DETECTION_MODEL, POSE_MODEL,
    ENABLE_TRACKING, ENABLE_POSE_ESTIMATION, ENABLE_BODY_SHAPE_CLASSIFICATION,
    ENABLE_SOCIAL_DISTANCE,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, PERSON_CLASS_ID, TRACKER_CONFIG,
    MAX_SCENE_FRAMES_IN_MEMORY,
)
from processing.video import (
    detect_scene_changes, detect_face_keypoint_changes_in_scene, analyze_scene_vlm,
)
from processing import person_dedup
from utils.visualization import is_frontal_pose_with_waist, count_visible_keypoints
from models.accessory import ACCESSORY_CATEGORIES


# ───────────────────────── helpers de GPU / Ollama ─────────────────────────

def load_yolo_gpu(device: str = "cuda"):
    """Carga detect + pose en GPU (ignora config.YOLO_DEVICE='cpu'). Devuelve
    (model_detect, model_pose). El swap a GPU es seguro porque en Fase A gemma4
    está parado (VRAM libre)."""
    from ultralytics import YOLO
    from ui import info
    model_detect = YOLO(DETECTION_MODEL)
    model_pose = YOLO(POSE_MODEL) if ENABLE_POSE_ESTIMATION else None
    try:
        model_detect.to(device)
        if model_pose is not None:
            model_pose.to(device)
        info(f"  YOLO en [cyan]{device}[/] (fase de detección)")
    except Exception as e:
        from ui import warn
        warn(f"  No se pudo mover YOLO a {device}: {e}")
    return model_detect, model_pose


def free_yolo(*models):
    """Libera los modelos YOLO de la GPU antes de la fase de categorización."""
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stop_gemma4(host: str | None = None):
    """Descarga gemma4 de la instancia del pipeline (libera VRAM para YOLO en GPU).
    Patrón de main._stop_all_ollama: generate(keep_alive=0) ≈ `ollama stop`."""
    if host is None:
        hosts = list(getattr(config, "OLLAMA_HOSTS", ["http://localhost:11434"]))
        host = hosts[0] if hosts else "http://localhost:11434"
    try:
        import ollama
        client = ollama.Client(host=host)
        ps = client.ps()
        models = getattr(ps, "models", None)
        if models is None and isinstance(ps, dict):
            models = ps.get("models", [])
        for m in (models or []):
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                try:
                    client.generate(model=name, keep_alive=0)
                except Exception:
                    pass
    except Exception:
        pass


# ───────────────────────── FASE A: detección + prescan ─────────────────────

def _prescan_with_kpts(scene_frames_data: list) -> dict:
    """Como `prescan_scene_for_best_frames` pero además guarda los keypoints del
    mejor frame general por track (los necesita la distancia social en Fase B) y
    el crop body_shape cuando hay pose frontal con cintura visible.

    Devuelve {track_id: {'general': {crop,score,frame_local_idx,bbox,kpts},
                         'body_shape': {crop,score,frame_local_idx} | None}}."""
    best = {}
    for local_idx, (frame, results_detect, results_pose, keypoints_list) in enumerate(scene_frames_data):
        if results_detect[0].boxes is None or len(results_detect[0].boxes) == 0:
            continue
        boxes = results_detect[0].boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None and ENABLE_TRACKING else None

        person_idx = 0
        for idx, box in enumerate(boxes):
            if int(box.cls) != PERSON_CLASS_ID:
                continue
            track_id = int(ids[idx]) if ids is not None else f"det_{local_idx}_{idx}"
            if track_id not in best:
                best[track_id] = {'general': None, 'body_shape': None, 'frames': set()}
            # frames en los que aparece este track (para el guard de co-ocurrencia
            # del dedup: dos tracks que comparten frame son personas distintas).
            best[track_id].setdefault('frames', set()).add(local_idx)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            bbox_area = (x2 - x1) * (y2 - y1)
            kpts = keypoints_list[person_idx] if person_idx < len(keypoints_list) else None

            general_score = confidence * bbox_area
            cur = best[track_id]['general']
            if cur is None or general_score > cur['score']:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    best[track_id]['general'] = {
                        'crop': crop.copy(), 'score': general_score,
                        'frame_local_idx': local_idx, 'bbox': (x1, y1, x2, y2),
                        'kpts': None if kpts is None else np.asarray(kpts).tolist(),
                    }

            if ENABLE_BODY_SHAPE_CLASSIFICATION and kpts is not None and is_frontal_pose_with_waist(kpts):
                body_score = count_visible_keypoints(kpts) / bbox_area if bbox_area > 0 else 0
                curb = best[track_id]['body_shape']
                if curb is None or body_score > curb['score']:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        best[track_id]['body_shape'] = {
                            'crop': crop.copy(), 'score': body_score,
                            'frame_local_idx': local_idx,
                        }
            person_idx += 1
    return best


def _yolo_frame(model_detect, model_pose, frame):
    """Detección + pose de un frame (mismas llamadas que process_video)."""
    if ENABLE_TRACKING:
        results_detect = model_detect.track(
            source=frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS_ID], persist=True, tracker=TRACKER_CONFIG, verbose=False,
        )
    else:
        results_detect = model_detect.predict(
            source=frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS_ID], verbose=False,
        )
    if ENABLE_POSE_ESTIMATION and model_pose is not None:
        results_pose = model_pose.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    else:
        from ultralytics.engine.results import Results
        results_pose = [Results(orig_img=frame, path="", names={}, boxes=None, keypoints=None)]
    keypoints_list = []
    if ENABLE_POSE_ESTIMATION and results_pose[0].keypoints is not None:
        keypoints_list = [kp.cpu().numpy() for kp in results_pose[0].keypoints.data]
    return results_detect, results_pose, keypoints_list


def detect_video_scenes(video_path: Path, model_detect, model_pose,
                        phase1_dir: Path, person_crops_dir: Path,
                        full_frames_dir: Path) -> dict | None:
    """FASE A. Detecta escenas/sub-escenas y persiste el mejor crop por persona +
    el frame de la escena + metadata. Devuelve el dict de metadata (también escrito
    a `phase1_dir/phase1_<stem>.json`). YOLO debe estar ya en GPU. NO usa el VLM."""
    from ui import info, warn

    stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warn(f"  Error al abrir vídeo: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scene_change_frames = detect_scene_changes(video_path) or [0]
    scenes = []
    for i, start in enumerate(scene_change_frames):
        end = scene_change_frames[i + 1] if i + 1 < len(scene_change_frames) else total_frames
        scenes.append((start, end))

    person_crops_dir.mkdir(parents=True, exist_ok=True)
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    out_scenes = []

    def _emit_subscene(label, sub_start_global, sub_end_global, sub_best, first_frame):
        # Dedup de personas: fusiona fragmentos del tracker + det_* por-frame que son
        # la MISMA persona (apariencia + co-ocurrencia). Antes de emitir/guardar crops.
        if getattr(config, "ENABLE_PERSON_DEDUP", True) and len(sub_best) > 1:
            n_before = len(sub_best)
            sub_best = person_dedup.dedup_candidates(
                sub_best,
                sim_threshold=getattr(config, "PERSON_DEDUP_SIM_THRESHOLD", 0.86),
            )
            if len(sub_best) < n_before:
                info(f"    [dim]dedup escena {label}: {n_before} → {len(sub_best)} personas")
        safe = str(label).replace(".", "_")
        frame_file = full_frames_dir / f"{stem}_scene_{safe}.jpg"
        if first_frame is not None and first_frame.size > 0:
            cv2.imwrite(str(frame_file), first_frame)
        persons = []
        for pidx, (tid, cand) in enumerate(sub_best.items()):
            gen = cand.get('general')
            if not gen or gen.get('crop') is None:
                continue
            crop_file = person_crops_dir / f"{stem}_scene_{safe}_person_{pidx}_track_{tid}.jpg"
            cv2.imwrite(str(crop_file), gen['crop'])
            body_file = None
            if cand.get('body_shape') and cand['body_shape'].get('crop') is not None:
                body_file = phase1_dir / f"{stem}_scene_{safe}_person_{pidx}_track_{tid}_body.jpg"
                cv2.imwrite(str(body_file), cand['body_shape']['crop'])
            persons.append({
                'track_id': tid,
                'bbox': [int(v) for v in gen['bbox']],
                'frame_local_idx': gen['frame_local_idx'],
                'kpts': gen.get('kpts'),
                'general_crop_file': str(crop_file),
                'body_crop_file': str(body_file) if body_file else None,
            })
        if persons:
            out_scenes.append({
                'scene_label': str(label),
                'start_frame': int(sub_start_global),
                'end_frame': int(sub_end_global),
                'frame_file': str(frame_file),
                'num_persons': len(persons),
                'persons': persons,
            })

    info(f"  Escenas: [cyan]{len(scenes)}")
    for scene_idx, (scene_start, scene_end) in enumerate(scenes):
        scene_num = scene_idx + 1
        scene_frame_count = scene_end - scene_start

        if scene_frame_count <= MAX_SCENE_FRAMES_IN_MEMORY:
            # Buffered: conserva sub-escenas por keypoints faciales.
            scene_frames = []
            for _ in range(scene_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                rd, rp, kl = _yolo_frame(model_detect, model_pose, frame)
                scene_frames.append((frame, rd, rp, kl))
            if not scene_frames:
                continue
            breaks = detect_face_keypoint_changes_in_scene(scene_frames, scene_start)
            sub_bounds = [0] + breaks + [len(scene_frames)]
            multi = len(sub_bounds) > 2
            for si in range(len(sub_bounds) - 1):
                sub = scene_frames[sub_bounds[si]:sub_bounds[si + 1]]
                if not sub:
                    continue
                label = f"{scene_num}.{si + 1}" if multi else scene_num
                sub_start_global = scene_start + sub_bounds[si]
                sub_best = _prescan_with_kpts(sub)
                _emit_subscene(label, sub_start_global,
                               sub_start_global + len(sub), sub_best, sub[0][0])
            del scene_frames
        else:
            # Streaming (escena enorme): sin sub-split, frame a frame, sin bufferizar.
            warn(f"  Escena {scene_num}: {scene_frame_count} frames → streaming (sin sub-split)")
            first_frame = None
            window = []  # acumula solo lo mínimo para prescan en bloques
            sub_best_acc = {}
            for fi in range(scene_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if fi == 0:
                    first_frame = frame.copy()
                rd, rp, kl = _yolo_frame(model_detect, model_pose, frame)
                window.append((frame, rd, rp, kl))
                if len(window) >= 256:
                    _merge_best(sub_best_acc, _prescan_with_kpts(window))
                    window = []
            if window:
                _merge_best(sub_best_acc, _prescan_with_kpts(window))
            if sub_best_acc:
                _emit_subscene(scene_num, scene_start, scene_end, sub_best_acc, first_frame)

    cap.release()

    meta = {
        'video_path': str(video_path), 'stem': stem,
        'width': width, 'height': height, 'total_frames': total_frames,
        'scenes': out_scenes,
    }
    phase1_path = phase1_dir / f"phase1_{stem}.json"
    with open(phase1_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    info(f"  [dim]Fase A {stem}: {len(out_scenes)} (sub)escena(s) detectadas")
    return meta


def _merge_best(acc: dict, new: dict):
    """Funde best_candidates de un bloque streaming en el acumulador por mayor score."""
    for tid, cand in new.items():
        if tid not in acc:
            acc[tid] = cand
            continue
        for kind in ('general', 'body_shape'):
            n = cand.get(kind)
            if n is None:
                continue
            cur = acc[tid].get(kind)
            if cur is None or n['score'] > cur['score']:
                acc[tid][kind] = n
        # unir los frames vistos (para el guard de co-ocurrencia del dedup)
        acc[tid].setdefault('frames', set()).update(cand.get('frames') or set())


# ───────────────────────── FASE B: categorización VLM ──────────────────────

def categorize_video_scenes(meta: dict, models: dict) -> dict:
    """FASE B. Reconstruye best_candidates desde los .jpg de la Fase A y reutiliza
    `analyze_scene_vlm` para las 14 categorías + distancia social por escena.
    Devuelve un `result` con la clave `scenes` (esquema de _dump_scene_validation)."""
    width = meta.get('width'); height = meta.get('height')
    frame_shape = (height, width, 3)
    sd = models.get('social_distance_classifier') if ENABLE_SOCIAL_DISTANCE else None

    out_scenes = []
    all_tracks = set()

    for sc in meta.get('scenes', []):
        # Reconstruir best_candidates con los crops cargados de disco.
        best_candidates = {}
        for p in sc['persons']:
            tid = p['track_id']
            gen_crop = cv2.imread(p['general_crop_file'])
            if gen_crop is None:
                continue
            entry = {
                'general': {'crop': gen_crop, 'frame_local_idx': 0, 'bbox': tuple(p['bbox'])},
                'beauty': None, 'body_shape': None,
            }
            if p.get('body_crop_file'):
                bc = cv2.imread(p['body_crop_file'])
                if bc is not None:
                    entry['body_shape'] = {'crop': bc, 'frame_local_idx': 0}
            best_candidates[tid] = entry
            all_tracks.add(tid)

        if not best_candidates:
            continue

        first_frame = cv2.imread(sc['frame_file']) if sc.get('frame_file') else None

        caches, _classifications, _ocr = analyze_scene_vlm(
            best_candidates, first_frame, sc['scene_label'],
            scene_start_frame=sc['start_frame'],
            behaviour_classifier=models.get('behaviour_classifier'),
            activity_classifier=models.get('activity_classifier'),
            body_display_classifier=models.get('body_display_classifier'),
            location_classifier=models.get('location_classifier'),
            body_shape_classifier=models.get('body_shape_classifier'),
            accessory_classifier=models.get('accessory_classifier'),
            ocr_classifier=models.get('ocr_classifier'),
            beauty_estimator=models.get('beauty_estimator'),
            gender_classifier=models.get('gender_classifier'),
            age_classifier=models.get('age_classifier'),
            person_attributes_classifier=models.get('person_attributes_classifier'),
            scene_context_classifier=models.get('scene_context_classifier'),
        )

        # Distancia social UNA vez por escena: gate por keypoints + fallback VLM
        # sobre el crop de la persona (USE_LEGACY_SOCIAL_DISTANCE usa el crop).
        sd_by_track = {}
        if sd is not None:
            num_persons = len(sc['persons'])
            to_vlm_imgs, to_vlm_tids = [], []
            for p in sc['persons']:
                tid = p['track_id']
                kpts = np.asarray(p['kpts']) if p.get('kpts') is not None else None
                res = sd.classify_from_keypoints(kpts, tuple(p['bbox']), frame_shape, num_persons)
                if res is not None:
                    sd_by_track[tid] = res.get('category')
                else:
                    crop = best_candidates.get(tid, {}).get('general', {}).get('crop')
                    if crop is not None:
                        to_vlm_imgs.append(crop)
                        to_vlm_tids.append(tid)
            if to_vlm_imgs:
                results = sd.classify_batch(to_vlm_imgs)
                for tid, r in zip(to_vlm_tids, results):
                    sd_by_track[tid] = (r or {}).get('category')

        # Ensamblar personas (esquema de _dump_scene_validation).
        def _get(cache_name, tid, field):
            entry = caches.get(cache_name, {}).get(tid)
            return entry.get(field) if isinstance(entry, dict) else None

        persons = []
        for p in sc['persons']:
            tid = p['track_id']
            if tid not in best_candidates:
                continue
            acc_entry = caches.get('accessory', {}).get(tid)
            acc_data = acc_entry.get('accessories') if isinstance(acc_entry, dict) else None
            accessories = {
                cat: (int(acc_data.get(cat, 0)) if isinstance(acc_data, dict) else None)
                for cat in ACCESSORY_CATEGORIES
            }
            persons.append({
                'track_id': tid,
                'crop_path': p['general_crop_file'],
                'bbox': p['bbox'],
                'gender':            _get('gender', tid, 'gender'),
                'age_group':         _get('age', tid, 'age_group'),
                'behaviour':         _get('behaviour', tid, 'behaviour'),
                'activity':          _get('activity', tid, 'activity'),
                'body_display':      _get('body_display', tid, 'body_display'),
                'location':          _get('location', tid, 'location'),
                'weight':            _get('body_shape', tid, 'body_weight'),
                'muscle':            _get('body_shape', tid, 'muscle'),
                'accessories':       accessories,
                'beauty':            _get('beauty', tid, 'score'),
                'social_distance':      sd_by_track.get(tid),
            })

        out_scenes.append({
            'scene_label': sc['scene_label'],
            'start_frame': sc['start_frame'],
            'end_frame': sc['end_frame'],
            'frame_path': sc.get('frame_file'),
            'persons': persons,
        })

    return {
        'input_path': meta.get('video_path'),
        'resolution': {'width': width, 'height': height},
        'total_frames': meta.get('total_frames'),
        'unique_persons_tracked': len(all_tracks),
        'scenes': out_scenes,
    }


# ───────────────────────── Fase A como SUBPROCESO ──────────────────────────
# Se ejecuta en un proceso aparte (`python -m processing.video_validation <job.json>`)
# para que el contexto CUDA de YOLO se libere POR COMPLETO al salir. Si se corriera
# en el proceso principal, el contexto torch-CUDA residual (~1-2 GB) dejaría a gemma4
# sin VRAM suficiente y Ollama lo cargaría parcialmente en CPU (66% CPU observado).

def _detect_phase_entry():
    """Punto de entrada del subproceso de Fase A. Lee un job JSON con la lista de
    vídeos y los directorios de salida, carga YOLO en GPU y detecta/recorta cada
    vídeo (escribe phase1_<stem>.json + crops). Al salir, la GPU queda limpia."""
    import sys
    from ui import info, warn

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        job = json.load(f)

    device = job.get("device", "cuda")
    phase1_dir = Path(job["phase1_dir"])
    person_crops_dir = Path(job["person_crops_dir"])
    full_frames_dir = Path(job["full_frames_dir"])

    model_detect, model_pose = load_yolo_gpu(device)
    for vid_id, vpath in job["videos"]:
        vpath = Path(vpath)
        if not vpath.exists():
            warn(f"  [Fase A] No encontrado: {vpath}")
            continue
        try:
            detect_video_scenes(vpath, model_detect, model_pose,
                                 phase1_dir, person_crops_dir, full_frames_dir)
        except Exception as e:
            warn(f"  [Fase A] Error en {vpath.name}: {e}")
    info("  [Fase A] subproceso terminado (GPU liberada al salir)")


if __name__ == "__main__":
    _detect_phase_entry()
