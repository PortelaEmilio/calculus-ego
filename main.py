import os
import sys
import cv2
import json
import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

import config as _config
from config import (
    OUTPUT_DIR, DETECTION_MODEL, POSE_MODEL, BEHAVIOUR_MODEL_NAME,
    FINETUNED_LORA_PATH, USE_FINETUNED_MODEL,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG,
    ENABLE_POSE_ESTIMATION, ENABLE_TRACKING,
    ENABLE_GENDER_CLASSIFICATION, ENABLE_AGE_CLASSIFICATION,
    ENABLE_BEHAVIOUR_CLASSIFICATION, ENABLE_ACTIVITY_CLASSIFICATION,
    ENABLE_BODY_DISPLAY_CLASSIFICATION, ENABLE_LOCATION_CLASSIFICATION,
    ENABLE_BODY_SHAPE_CLASSIFICATION, ENABLE_ACCESSORY_CLASSIFICATION,
    ENABLE_OCR, ENABLE_SOCIAL_DISTANCE, ENABLE_BEAUTY_ESTIMATION,
    ENABLE_SCENE_DETECTION, ENABLE_COLLAGE_DETECTION,
    COLLAGE_MIN_PANELS, COLLAGE_MIN_PANEL_SIZE_PERCENT,
)
from models.loader import load_all_models
from processing.image import process_image, detect_collage_panels, extract_and_save_panels, visualize_collage_detection, create_panels_montage
from processing.video import process_video, process_video_with_panels, PYSCENEDETECT_AVAILABLE
from processing.batch import process_batch
from ui import console, header, info, success, warn, error, kv_table, dist_table


# ── Esquema de salida y constructores de filas (compartidos por `manual` y `carpeta`) ──
# Extraídos de _process_manual a nivel de módulo para que el modo `carpeta`
# (_process_directory) los reutilice sin duplicar. `run_id` pasa como parámetro.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Modo `carpeta`: cada cuántos archivos NUEVOS se vuelca el .xlsx maestro (constante,
# no parametrizable). El summary_<stem>.json por archivo es la seguridad real crash-safe.
DIRECTORY_CHECKPOINT = 10

ACC_TO_COL = {
    "makeup":   "IA Maquill.",
    "tattoos":  "IA Tattoos",
    "bags":     "IA Bolsos",
    "belts":    "IA Cints.",
    "jewelry":  "IA Joyas",
    "headwear": "IA Sombr.",
    "eyewear":  "IA Gafas",
}

FINAL_COLS = [
    "Img. orig.", "Img. anot.", "Estado",
    "IA Nº pers.", "IA Tamaño (%)", "IA BBox px", "IA BBox xyxy",
    "IA Gen.", "IA Edad", "IA Comport.", "IA Activ.",
    "IA Exp. Cp.", "IA Ubic.", "IA Adiposity", "IA Musculatura", "IA Vestimenta", "IA Dist. Soc.", "IA Belleza (1-10)",
    "IA Maquill.", "IA Tattoos", "IA Bolsos", "IA Cints.", "IA Joyas", "IA Sombr.", "IA Gafas",
    "Valid.", "Revisor", "Notas",
    "Run ID", "Join ID", "Img ID", "Archivo", "Pers. Idx", "Pers. Key",
    "Ruta Img.", "Ruta JSON", "Proc. At",
    # Solo vídeo (validation_dump): escena y rutas a imágenes embebibles.
    "Scene", "Crop Path", "Frame Path",
]


def find_by_track(lst, track_id):
    for item in lst:
        if item.get("track_id") == track_id:
            return item
    return None


def build_video_rows(run_id, img_id, img_path, result, json_path):
    """Una fila por (escena, persona) a partir de result['scenes'] (vídeo)."""
    rows = []
    scenes = result.get("scenes") or []
    n_tracks = result.get("unique_persons_tracked")
    if not scenes:
        rows.append({
            "Estado": "sin escenas", "IA Nº pers.": n_tracks,
            "Run ID": run_id, "Join ID": f"{img_id}__none",
            "Img ID": img_id, "Archivo": Path(img_path).name,
            "Pers. Idx": None, "Pers. Key": None,
            "Ruta Img.": str(img_path), "Ruta JSON": str(json_path),
            "Proc. At": datetime.now().isoformat(),
        })
        return rows

    pers_idx = 0
    for scene in scenes:
        label = scene.get("scene_label")
        frame_path = scene.get("frame_path")
        scene_persons = scene.get("persons") or []
        # Escena SIN personas (2026-08-31): la Fase B ya no la descarta, clasifica solo
        # la ubicación sobre el frame → una fila con `IA Ubic.` y las rutas de revisión.
        if not scene_persons:
            rows.append({
                "Estado":        "ok",
                "IA Nº pers.":   n_tracks,
                "IA Ubic.":      scene.get("location"),
                "Valid.": None, "Revisor": None, "Notas": None,
                "Run ID":    run_id,
                "Join ID":   f"{img_id}__{label}__none",
                "Img ID":    img_id,
                "Archivo":   Path(img_path).name,
                "Pers. Idx": None,
                "Pers. Key": None,
                "Ruta Img.": str(img_path),
                "Ruta JSON": str(json_path),
                "Proc. At":  datetime.now().isoformat(),
                "Scene":      label,
                "Crop Path":  None,
                "Frame Path": frame_path,
            })
            continue
        for p in scene_persons:
            acc = p.get("accessories") or {}
            row = {
                "Estado":        "ok",
                "IA Nº pers.":   n_tracks,
                "IA Tamaño (%)": p.get("occupancy"),
                "IA BBox px":    p.get("bbox_area"),
                "IA BBox xyxy":  json.dumps(p.get("bbox_xyxy")) if p.get("bbox_xyxy") else None,
                "IA Gen.":       p.get("gender"),
                "IA Edad":       p.get("age_group"),
                "IA Comport.":   p.get("behaviour"),
                "IA Activ.":     p.get("activity"),
                "IA Exp. Cp.":   p.get("body_display"),
                "IA Ubic.":      p.get("location"),
                "IA Adiposity":  p.get("weight"),
                "IA Musculatura": p.get("muscle"),
                "IA Vestimenta": p.get("attire"),
                "IA Dist. Soc.": p.get("social_distance"),
                "IA Belleza (1-10)": p.get("beauty"),
                "Valid.": None, "Revisor": None, "Notas": None,
                "Run ID":    run_id,
                "Join ID":   f"{img_id}__{label}__{pers_idx}",
                "Img ID":    img_id,
                "Archivo":   Path(img_path).name,
                "Pers. Idx": pers_idx,
                "Pers. Key": p.get("track_id"),
                "Ruta Img.": str(img_path),
                "Ruta JSON": str(json_path),
                "Proc. At":  datetime.now().isoformat(),
                "Scene":      label,
                "Crop Path":  p.get("crop_path"),
                "Frame Path": frame_path,
            }
            for cat, col in ACC_TO_COL.items():
                val = acc.get(cat)
                row[col] = "sí" if val == 1 else ("no" if val == 0 else None)
            rows.append(row)
            pers_idx += 1
    return rows


def build_rows(run_id, img_id, img_path, result, json_path):
    rows = []
    if result is None:
        return rows
    # Vídeo: result trae la estructura por escena (validation_dump). Una fila
    # por (escena, persona) en vez de la lógica de imagen (una por persona).
    if result.get("scenes") is not None:
        return build_video_rows(run_id, img_id, img_path, result, json_path)
    n = result.get("persons_detected", 0)
    # n_yolo: detecciones totales de YOLO26 (incluye las filtradas por
    # BBOX_MIN_FRAME_RATIO que no entran al VLM). Es el valor que se
    # reporta como `IA Nº pers.` — el recuento de personas por imagen no
    # debe depender de qué crops eran lo bastante grandes para el VLM.
    # Fallback al persons_detected antiguo para JSONs pre-cambio.
    n_yolo = result.get("persons_detected_yolo", n)
    vlm_status = result.get("vlm_status") or "ok"

    if n == 0:
        # Ubicación de ESCENA (2026-08-31): con 0 personas no hay entradas por track,
        # pero `process_image` sí anota una entrada de escena con track_id=None
        # (ENABLE_SCENE_LOCATION_NO_PERSON). Sin ella la celda queda vacía, como antes.
        scene_loc = next(
            (l for l in (result.get("location_classifications") or [])
             if l.get("track_id") is None), None)
        rows.append({
            "Img. orig.":  None,
            "Img. anot.":  None,
            "Estado":      vlm_status,
            "IA Nº pers.": n_yolo,
            "IA Gen.": None, "IA Edad": None, "IA Comport.": None, "IA Activ.": None,
            "IA Exp. Cp.": None,
            "IA Ubic.":    scene_loc.get("location") if scene_loc else None,
            "IA Dist. Soc.": None,
            "IA Maquill.": None, "IA Tattoos": None, "IA Bolsos": None, "IA Cints.": None,
            "IA Joyas": None, "IA Sombr.": None, "IA Gafas": None,
            "Valid.": None, "Revisor": None, "Notas": None,
            "Run ID":    run_id,
            "Join ID":   f"{img_id}__none",
            "Img ID":    img_id,
            "Archivo":   Path(img_path).name,
            "Pers. Idx": None,
            "Pers. Key": None,
            "Ruta Img.": str(img_path),
            "Ruta JSON": str(json_path),
            "Proc. At":  datetime.now().isoformat(),
        })
        return rows

    # Lista de track_ids reales preservando orden de detección. NO usar
    # f"det_0_{i}" porque el pipeline mezcla detecciones de YOLO detector
    # (`det_0_<idx>`) con las del modelo de pose cuando el detector las
    # pierde (`pose_0_<idx>`). Asumir solo `det_0_*` producía track_ids
    # inexistentes → todas las columnas IA en NaN para personas pose-only.
    gender_list = result.get("gender_classifications", [])
    ordered_track_ids = [g.get("track_id") for g in gender_list]
    while len(ordered_track_ids) < n:
        ordered_track_ids.append(f"det_0_{len(ordered_track_ids)}")

    for i in range(n):
        track_id = ordered_track_ids[i]
        join_id  = f"{img_id}__{i}"

        gen  = find_by_track(result.get("gender_classifications", []),         track_id)
        age  = find_by_track(result.get("age_classifications", []),            track_id)
        beh  = find_by_track(result.get("behaviour_classifications", []),      track_id)
        act  = find_by_track(result.get("activity_classifications", []),       track_id)
        bdis = find_by_track(result.get("body_display_classifications", []),   track_id)
        loc  = find_by_track(result.get("location_classifications", []),       track_id)
        bsh  = find_by_track(result.get("body_shape_classifications", []),     track_id)
        acc  = find_by_track(result.get("accessory_classifications", []),      track_id)
        soc  = find_by_track(result.get("social_distance_classifications", []), track_id)

        row = {
            "Img. orig.":     None,
            "Img. anot.":     None,
            "Estado":         vlm_status,
            "IA Nº pers.":    n_yolo,
            "IA Tamaño (%)":  gen.get("occupancy")     if gen  else None,
            "IA BBox px":     gen.get("bbox_area")     if gen  else None,
            "IA BBox xyxy":   json.dumps(gen.get("bbox")) if (gen and gen.get("bbox")) else None,
            "IA Gen.":        gen.get("gender")        if gen  else None,
            "IA Edad":        age.get("age_group")     if age  else None,
            "IA Comport.":    beh.get("behaviour")     if beh  else None,
            "IA Activ.":      act.get("activity")      if act  else None,
            "IA Exp. Cp.":    bdis.get("body_display") if bdis else None,
            "IA Ubic.":       loc.get("location")      if loc  else None,
            "IA Adiposity":   bsh.get("body_weight")   if bsh  else None,
            "IA Musculatura": bsh.get("muscle")        if bsh  else None,
            "IA Vestimenta":  bsh.get("attire")        if bsh  else None,
            "IA Dist. Soc.":  soc.get("category")      if soc  else None,
            "Valid.":  None,
            "Revisor": None,
            "Notas":   None,
            "Run ID":    run_id,
            "Join ID":   join_id,
            "Img ID":    img_id,
            "Archivo":   Path(img_path).name,
            "Pers. Idx": i,
            "Pers. Key": track_id,
            "Ruta Img.": str(img_path),
            "Ruta JSON": str(json_path),
            "Proc. At":  datetime.now().isoformat(),
        }

        for cat, col in ACC_TO_COL.items():
            if acc is not None:
                row[col] = "sí" if acc.get(cat) == 1 else "no"
            else:
                row[col] = None

        rows.append(row)
    return rows


def _process_manual(args, models):
    """Procesa imágenes únicas de un Excel manual y emite un qwen_*.xlsx."""
    from ui import console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )

    manual_path = Path(args.manual_xlsx).expanduser().resolve()
    if not manual_path.exists():
        console.print(f"[red]Manual no encontrado:[/] {manual_path}")
        return

    output_path = (
        Path(args.output).expanduser().resolve() if args.output
        else manual_path.with_name(f"qwen_{manual_path.stem}.xlsx")
    )
    runs_root = (
        Path(args.runs_dir).expanduser().resolve() if args.runs_dir
        else manual_path.parent / "runs"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id   = f"{args.run_prefix}_{timestamp}"
    run_dir  = runs_root / run_id
    # --resume: reutiliza el run más reciente con el mismo prefijo en vez de crear uno
    # nuevo, para continuar donde se cortó (salta vídeos con summary/phase1 ya escrito).
    resume = bool(getattr(args, "resume", False))
    if resume:
        existing = sorted(
            [d for d in runs_root.glob(f"{args.run_prefix}_*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        if existing:
            run_dir = existing[-1]
            run_id = run_dir.name
            console.print(f"  [cyan]↻ resume:[/] reutilizando run [bold]{run_id}")
        else:
            console.print(f"  [yellow]↻ resume: no hay run previo con prefijo {args.run_prefix}_*, empiezo uno nuevo")
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dir = run_dir / "json"
    annot_dir = run_dir / "annotated"
    json_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)

    # --- Validar el path de --output ANTES de procesar -----------------------
    # El guardado real ocurre al final (tras horas de cómputo). Si --output
    # apunta a un directorio o a una ruta no escribible, abortar YA en vez de
    # petar en to_excel() al terminar. (Los JSON por imagen se guardan igual en
    # run_dir/json/, así que un fallo tardío es recuperable, pero mejor evitarlo.)
    if output_path.is_dir():
        console.print(
            f"[red]--output es un directorio, no un .xlsx:[/] {output_path}\n"
            f"  [yellow]Usa un archivo, p.ej.[/] {output_path / 'qwen.xlsx'}")
        return
    if output_path.suffix.lower() != ".xlsx":
        console.print(f"[red]--output debe terminar en .xlsx:[/] {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _probe = output_path.parent / f".{run_id}_writetest.tmp"
        _probe.write_bytes(b""); _probe.unlink()
    except OSError as e:
        console.print(f"[red]No se puede escribir en {output_path.parent}:[/] {e}")
        return

    console.rule(f"[bold]Cargando {manual_path.name}")
    df_man = pd.read_excel(manual_path)
    unique_images = (
        df_man.drop_duplicates(subset="Img ID")[["Img ID", "Ruta Img."]]
              .reset_index(drop=True)
    )

    if args.only_img_ids:
        token = args.only_img_ids
        token_path = manual_path.parent / token
        if token_path.exists() and token_path.is_file():
            ids = [int(line.strip()) for line in token_path.read_text().splitlines() if line.strip()]
        else:
            ids = [int(x.strip()) for x in token.split(",") if x.strip()]
        unique_images = (
            unique_images[unique_images["Img ID"].astype(int).isin(ids)]
                        .reset_index(drop=True)
        )
        console.print(f"  [dim]filtrado --only-img-ids: {len(unique_images)}/{len(ids)}")

    console.print(f"  [dim]{len(unique_images)} imágenes únicas")

    def _worker(img_id, img_path):
        json_path = json_dir / f"summary_{img_id}.json"
        if not img_path.exists():
            return img_id, img_path, json_path, "missing", None, None
        try:
            result = process_image(
                img_path,
                models['model_detect'], models['model_pose'],
                annot_dir,
                beauty_estimator=models['beauty_estimator'],
                gender_classifier=models['gender_classifier'],
                age_classifier=models['age_classifier'],
                behaviour_classifier=models['behaviour_classifier'],
                activity_classifier=models['activity_classifier'],
                body_display_classifier=models['body_display_classifier'],
                location_classifier=models['location_classifier'],
                body_shape_classifier=models['body_shape_classifier'],
                accessory_classifier=models['accessory_classifier'],
                social_distance_classifier=models['social_distance_classifier'],
                person_attributes_classifier=models.get('person_attributes_classifier'),
                scene_context_classifier=models.get('scene_context_classifier'),
            )
            return img_id, img_path, json_path, "ok", result, None
        except Exception as e:
            return img_id, img_path, json_path, "error", None, e

    all_rows, errors = [], []

    # ── Vídeos: pipeline RÁPIDO en 2 fases (YOLO-GPU detecta tandas → libera GPU →
    #    gemma4 categoriza). Reemplaza el process_video por-vídeo (lento). Las imágenes
    #    siguen por el threadpool de abajo. ────────────────────────────────────────
    is_video = unique_images["Ruta Img."].apply(
        lambda p: Path(str(p)).suffix.lower() in VIDEO_EXTS
    )
    df_videos = unique_images[is_video].reset_index(drop=True)
    df_images = unique_images[~is_video].reset_index(drop=True)

    if len(df_videos):
        import subprocess
        import tempfile
        from processing.video_validation import categorize_video_scenes, stop_gemma4

        chunk = max(1, int(getattr(args, "chunk", 100)))
        yolo_device = getattr(args, "yolo_device", "cuda")
        phase1_root = run_dir / "phase1"
        phase1_root.mkdir(parents=True, exist_ok=True)
        person_crops_dir = annot_dir / "person_crops"
        full_frames_dir = annot_dir / "full_frames"
        pkg_dir = Path(__file__).resolve().parent  # Analisis_identidad/

        # El id se deriva del STEM del fichero ({id}.mp4), no de la columna Img ID:
        # Excel guarda los enteros como float64 y los ids de >16 dígitos (>2^53)
        # pierden precisión al round-trip (p.ej. ...326 → ...328). El stem es exacto.
        vids = [(Path(str(r["Ruta Img."])).stem, Path(str(r["Ruta Img."]))) for _, r in df_videos.iterrows()]
        n_chunks = (len(vids) + chunk - 1) // chunk
        console.print(f"  [cyan]{len(vids)} vídeo(s)[/] en {n_chunks} tanda(s) de {chunk} (Fase A YOLO@{yolo_device} → Fase B gemma4)")

        # Barra de progreso global sobre TODOS los vídeos. Avanza 1 por vídeo en la
        # Fase B (nuevo, saltado-ya-hecho o error), así que con --resume refleja el
        # total real. Se PAUSA durante la Fase A para no corromper el live display de
        # rich con la salida a stdout del subproceso YOLO.
        vprogress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        vtask = vprogress.add_task("[cyan]Vídeos", total=len(vids))
        vprogress.start()

        for ci in range(n_chunks):
            tanda = vids[ci * chunk:(ci + 1) * chunk]

            # --resume: salta detección de los que ya tienen summary (hechos del todo)
            # o phase1 (Fase A ya hecha, solo falta Fase B). Sin --resume, detecta todos.
            if resume:
                to_detect = [
                    (vid_id, vpath) for vid_id, vpath in tanda
                    if not (json_dir / f"summary_{vid_id}.json").exists()
                    and not (phase1_root / f"phase1_{vpath.stem}.json").exists()
                ]
            else:
                to_detect = list(tanda)

            console.rule(f"[bold]Tanda {ci + 1}/{n_chunks} — Fase A (detección YOLO@{yolo_device}, subproceso)")
            if to_detect:
                # Fase A en SUBPROCESO: libera gemma4, lanza YOLO@GPU en proceso aparte
                # (al salir libera TODO el contexto CUDA → el padre queda GPU-limpio y
                # gemma4 puede cargar 100% en GPU en la Fase B).
                vprogress.stop()  # pausa la barra: el subproceso YOLO escribe a stdout
                stop_gemma4()
                job = {
                    "device": yolo_device,
                    "phase1_dir": str(phase1_root),
                    "person_crops_dir": str(person_crops_dir),
                    "full_frames_dir": str(full_frames_dir),
                    "videos": [[vid_id, str(vpath)] for vid_id, vpath in to_detect],
                }
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jf:
                    json.dump(job, jf)
                    job_path = jf.name
                try:
                    subprocess.run(
                        [sys.executable, "-m", "processing.video_validation", job_path],
                        cwd=str(pkg_dir), check=False,
                    )
                finally:
                    try:
                        os.unlink(job_path)
                    except OSError:
                        pass
                    vprogress.start()  # reanuda la barra tras la Fase A
            else:
                console.print("  [dim]↻ resume: nada que detectar en esta tanda")

            console.rule(f"[bold]Tanda {ci + 1}/{n_chunks} — Fase B (categorización gemma4)")
            # Fase B: GPU limpia; gemma4 recarga en la 1ª llamada (100% GPU). Reusa classifiers.
            for vid_id, vpath in tanda:
                vprogress.update(vtask, description=f"[cyan]{vpath.name}")
                json_path = json_dir / f"summary_{vid_id}.json"
                if resume and json_path.exists():
                    # Ya categorizado en una corrida previa: carga sus filas, no reprocesa.
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            prev = json.load(f)
                        all_rows.extend(build_rows(run_id, vid_id, vpath, prev.get("result"), json_path))
                        console.print(f"  [dim]↻ saltado (ya hecho): {vpath.name}")
                    except Exception as e:
                        console.print(f"  [red]·[/] Error leyendo summary de {vpath.name}: [dim]{e}")
                        errors.append(vid_id)
                    vprogress.advance(vtask)
                    continue
                phase1_file = phase1_root / f"phase1_{vpath.stem}.json"
                if not phase1_file.exists():
                    console.print(f"  [yellow]·[/] Sin Fase A: [dim]{vpath.name}")
                    errors.append(vid_id)
                    vprogress.advance(vtask)
                    continue
                try:
                    with open(phase1_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    result = categorize_video_scenes(meta, models)
                    summary = {
                        "run_id": run_id, "img_id": vid_id,
                        "input_path": str(vpath), "result": result,
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    all_rows.extend(build_rows(run_id, vid_id, vpath, result, json_path))
                    n_sc = len(result.get("scenes") or [])
                    console.print(f"  [green]✓[/] {vpath.name} [dim]({n_sc} escena(s))")
                except Exception as e:
                    console.print(f"  [red]·[/] Fase B error en {vpath.name}: [dim]{e}")
                    errors.append(vid_id)
                vprogress.advance(vtask)

        vprogress.stop()

    unique_images = df_images  # el threadpool de abajo procesa solo imágenes

    file_workers = max(1, getattr(_config, "OLLAMA_FILE_PARALLEL", 1))
    # Transformers comparte un único modelo en GPU; paralelizar archivos
    # duplica las activaciones → OOM en GPUs ≤12 GiB. Forzar 1 worker.
    if getattr(_config, "VLM_BACKEND", "ollama") == "transformers":
        file_workers = 1
    state_lock = threading.Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Procesando imágenes", total=len(unique_images))
        with ThreadPoolExecutor(max_workers=file_workers) as pool:
            futures = [
                pool.submit(_worker, int(row["Img ID"]), Path(row["Ruta Img."]))
                for _, row in unique_images.iterrows()
            ]
            for fut in as_completed(futures):
                img_id, img_path, json_path, status, result, exc = fut.result()
                with state_lock:
                    progress.update(task, description=f"[cyan]{img_path.name}")
                    if status == "missing":
                        console.print(f"  [yellow]·[/] No encontrada: [dim]{img_path}")
                        errors.append(img_id)
                    elif status == "error":
                        console.print(f"  [red]·[/] Error en {img_path.name}: [dim]{exc}")
                        errors.append(img_id)
                    else:
                        summary = {
                            "run_id": run_id,
                            "img_id": img_id,
                            "input_path": str(img_path),
                            "result": result,
                        }
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(summary, f, indent=2, ensure_ascii=False)
                        all_rows.extend(build_rows(run_id, img_id, img_path, result, json_path))
                    progress.advance(task)

    console.rule(f"[bold]Guardando {output_path.name}")

    if not all_rows:
        console.print("[red]No hay filas que guardar. Abortando.")
        return

    df_out = pd.DataFrame(all_rows)
    for col in FINAL_COLS:
        if col not in df_out.columns:
            df_out[col] = None
    df_out = df_out[FINAL_COLS]

    # Pase de belleza diferido (Qwen3-VL @ :11435, solo demand/*). Tras clasificar con
    # gemma4, hace `ollama stop` y puntúa los rostros demand/*; fusiona la columna por
    # (Img ID, Pers. Key==track_id). Degrada con elegancia si está off / no hay demand /
    # :11435 caído (devuelve None → df_out se guarda sin la columna rellena).
    from processing.beauty_phase import run_beauty_phase, shared_beauty_backend_from_models
    df_beauty, beauty_col = run_beauty_phase(
        json_dir, models['model_pose'],
        shared_backend=shared_beauty_backend_from_models(models))
    if df_beauty is not None and not df_beauty.empty and beauty_col in df_beauty.columns:
        bmap = {
            (str(r["Img ID"]), str(r["track_id"])): r[beauty_col]
            for _, r in df_beauty.iterrows()
            if r.get("track_id") is not None
        }
        # merge SIN pisar: las filas de VÍDEO ya traen su belleza (pase diferido de
        # video_validation) y no están en df_beauty (que es solo de imágenes).
        df_out["IA Belleza (1-10)"] = [
            bmap.get((str(iid), str(pk)), cur)
            for iid, pk, cur in zip(df_out["Img ID"], df_out["Pers. Key"],
                                    df_out["IA Belleza (1-10)"])
        ]
        # Sidecar de auditoría (por_rostro) junto al output.
        try:
            sidecar = output_path.with_name(output_path.stem + "_belleza.xlsx")
            df_beauty.to_excel(sidecar, index=False, engine="openpyxl")
            console.print(f"  [dim]belleza (auditoría) → {sidecar.name}")
        except Exception as e:
            console.print(f"  [yellow]no se pudo escribir el sidecar de belleza: {e}")

    if args.append and output_path.exists():
        df_prev = pd.read_excel(output_path, engine="openpyxl")
        new_ids = set(df_out["Img ID"].astype(int))
        before = len(df_prev)
        df_prev = df_prev[~df_prev["Img ID"].astype(int).isin(new_ids)]
        console.print(f"  [dim]append: -{before - len(df_prev)} filas previas")
        df_out = pd.concat([df_prev, df_out], ignore_index=True)[FINAL_COLS]

    try:
        df_out.to_excel(output_path, index=False, engine="openpyxl")
    except Exception as e:
        # Red de seguridad: nunca perder el cómputo por un fallo de escritura
        # tardío. Volcar a una copia dentro del run_dir.
        fallback = run_dir / "qwen_recovered.xlsx"
        console.print(f"  [red]✗ Fallo al guardar en {output_path}: {e}")
        console.print(f"  [yellow]→ Guardando copia de seguridad en:[/] {fallback}")
        df_out.to_excel(fallback, index=False, engine="openpyxl")
        output_path = fallback
    console.print(f"  [green]✓[/] {output_path}")
    console.print(f"  [dim]filas={len(df_out)}  errores={len(errors)}  run={run_id}")
    if errors:
        sample = errors[:10]
        suffix = "..." if len(errors) > 10 else ""
        console.print(f"  [yellow]fallidas ({len(errors)}): {sample}{suffix}")


def _flush_directory_excel(all_rows, meta_resolver, output_path, run_dir):
    """Vuelca `all_rows` (con las columnas de metadatos ya anexadas por fila) a
    `output_path` en orden `FINAL_COLS + meta.columns()`. Red de seguridad a
    run_dir/recovered.xlsx si la escritura falla (nunca perder el cómputo)."""
    if not all_rows:
        return
    extra_cols = meta_resolver.columns()
    df = pd.DataFrame(all_rows)
    for col in (FINAL_COLS + extra_cols):
        if col not in df.columns:
            df[col] = None
    df = df[FINAL_COLS + extra_cols]
    try:
        df.to_excel(output_path, index=False, engine="openpyxl")
    except Exception as e:
        fallback = run_dir / "recovered.xlsx"
        console.print(f"  [red]✗ fallo guardando {output_path}: {e}")
        console.print(f"  [yellow]→ copia de seguridad:[/] {fallback}")
        try:
            df.to_excel(fallback, index=False, engine="openpyxl")
        except Exception as e2:
            console.print(f"  [red]✗ también falló el fallback: {e2}")


def _video_duration_s(path):
    """Duración de un vídeo en segundos leyendo SOLO la cabecera (mismo patrón que
    processing/video.py: CAP_PROP_FRAME_COUNT / CAP_PROP_FPS, sin decodificar
    frames — unos pocos ms por fichero). Devuelve None si no se puede leer
    (fichero corrupto, códec raro…); nunca lanza."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps <= 0 or frames <= 0:
                return None
            return frames / fps
        finally:
            cap.release()
    except Exception:
        return None


def _defer_long_videos(vids, threshold_s, cache_csv, json_dir):
    """Modo `carpeta`: reordena la lista de vídeos para que los de duración
    ≥ `threshold_s` queden AL FINAL, ascendente por duración (el más largo, el
    último); los cortos conservan el orden original. Motivación: un solo vídeo
    muy largo (hasta ~106 min en el corpus) puede tardar ~1 h y bloquear el
    barrido. La reanudación no se ve afectada (skip por summary_<stem>.json,
    independiente del orden).

    Duraciones desde la caché `cache_csv` (media_id,duracion_s — generada por
    validacion_imagenes/examen_duracion_videos.py); solo se sondean con cv2 los
    PENDIENTES (sin summary) que falten en la caché, que se re-escribe con lo
    nuevo. Duración ilegible → se trata como corto (nunca crashea).

    Devuelve (vids_reordenados, n_diferidos, max_duracion_s, n_sondeados)."""
    durations = {}
    if cache_csv.exists():
        try:
            with open(cache_csv, "r", encoding="utf-8") as f:
                next(f, None)  # cabecera
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) >= 2 and parts[1]:
                        try:
                            durations[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
        except OSError:
            pass
    missing = [(stem, path) for stem, path in vids
               if stem not in durations
               and not (json_dir / f"summary_{stem}.json").exists()]
    probed = 0
    for stem, path in missing:
        d = _video_duration_s(path)
        if d is not None:
            durations[stem] = d
            probed += 1
    if probed:
        try:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_csv, "w", encoding="utf-8") as f:
                f.write("media_id,duracion_s\n")
                for k in sorted(durations):
                    f.write(f"{k},{durations[k]:.6f}\n")
        except OSError:
            pass
    cortos = [(s, p) for s, p in vids if durations.get(s, 0.0) < threshold_s]
    largos = sorted(((s, p) for s, p in vids if durations.get(s, 0.0) >= threshold_s),
                    key=lambda sp: durations[sp[0]])
    max_dur = durations[largos[-1][0]] if largos else 0.0
    return cortos + largos, len(largos), max_dur, probed


def _process_directory(args, models):
    """Modo `carpeta`: procesa TODO `multimedia_downloads/` (imágenes + vídeos),
    reanuda solo (salta lo ya hecho por summary_<stem>.json), guarda el .xlsx por lotes
    fijos, une metadatos de los CSV, corre belleza siempre y NO persiste crops/anotadas."""
    import subprocess
    import tempfile
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from processing.video_validation import categorize_video_scenes, stop_gemma4
    from processing.metadata_join import build_default_index, build_metadata_index, DEFAULT_SOURCES

    pkg_dir = Path(__file__).resolve().parent      # Analisis_identidad/
    repo = pkg_dir.parent

    mdir = Path(args.multimedia_dir).expanduser().resolve() if args.multimedia_dir \
        else repo / "multimedia_downloads"
    if not mdir.is_dir():
        console.print(f"[red]Directorio multimedia no encontrado:[/] {mdir}")
        return

    output_path = Path(args.output).expanduser().resolve() if args.output \
        else repo / "multimedia_downloads_analizado.xlsx"
    if output_path.suffix.lower() != ".xlsx":
        console.print(f"[red]--output debe terminar en .xlsx:[/] {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_dir = Path(args.metadata_dir).expanduser().resolve() if args.metadata_dir else repo
    runs_root = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else mdir.parent / "runs"

    # ── Run dir: reanudación AUTOMÁTICA (reutiliza el run más reciente del prefijo) ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_prefix}_{timestamp}"
    run_dir = runs_root / run_id
    existing = sorted(
        [d for d in runs_root.glob(f"{args.run_prefix}_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    ) if runs_root.exists() else []
    if existing:
        run_dir = existing[-1]
        run_id = run_dir.name
        console.print(f"  [cyan]↻ reanudando run [bold]{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dir = run_dir / "json"
    annot_dir = run_dir / "annotated"
    phase1_root = run_dir / "phase1"
    person_crops_dir = annot_dir / "person_crops"
    full_frames_dir = annot_dir / "full_frames"
    for d in (json_dir, annot_dir, phase1_root):
        d.mkdir(parents=True, exist_ok=True)

    # ── Índice de metadatos (una vez): cadena RAW_7→…→limpio (ver DEFAULT_SOURCES) ──
    console.rule("[bold]Cargando índice de metadatos")
    meta = build_default_index(meta_dir)
    present = [n for n, _t, _m in DEFAULT_SOURCES if (meta_dir / n).exists()]
    console.print(f"  [dim]{len(present)} fuente(s): {', '.join(present)}")
    console.print(f"  [dim]metadatos: {len(meta.columns())} columnas anexadas por post")

    # ── Scan del directorio → (stem, path). Los archivos SIN match en ninguna fuente
    #    de metadatos NO se procesan (decisión del usuario 2026-07-09). ──
    console.rule(f"[bold]Escaneando {mdir.name}")
    entries, skipped = [], []
    for p in sorted(mdir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue
        if not meta.has_match(p.stem):
            skipped.append(p.name)
            continue
        entries.append((p.stem, p, ext in VIDEO_EXTS))
    if skipped:
        # Persistir la lista de saltados para auditoría (append-safe entre reanudaciones).
        sm = run_dir / "sin_match.txt"
        try:
            with open(sm, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(set(skipped))) + "\n")
        except OSError:
            pass
        console.print(f"  [yellow]⊘ {len(skipped)} archivo(s) SIN metadatos → NO se procesan[/] "
                      f"[dim](lista en {sm.name})")
    vids = [(stem, path) for stem, path, is_v in entries if is_v]
    imgs = [(stem, path) for stem, path, is_v in entries if not is_v]
    console.print(f"  [dim]{len(entries)} archivos: {len(imgs)} imagen(es) + {len(vids)} vídeo(s)")

    # ── Vídeos largos AL FINAL (≥ CARPETA_LONG_VIDEO_THRESHOLD_S; examen 2026-07-11:
    #    ≥60s = 17,6% de los vídeos pero 54,2% del tiempo). getattr → tolerante a
    #    configs de rollback sin la constante. ──
    _long_thr = getattr(_config, "CARPETA_LONG_VIDEO_THRESHOLD_S", 0) or 0
    if vids and _long_thr > 0:
        _dur_cache = repo / "validacion_imagenes" / "duraciones_videos.csv"
        vids, _n_long, _max_dur, _n_probed = _defer_long_videos(
            vids, _long_thr, _dur_cache, json_dir)
        if _n_long:
            msg = (f"  [cyan]⏩ {_n_long} vídeo(s) ≥{_long_thr:g}s diferidos al final, "
                   f"ascendente (el más largo: {_max_dur / 60:.1f} min)")
            if _n_probed:
                msg += f" [dim]· {_n_probed} duración(es) sondeada(s) → caché"
            console.print(msg)

    all_rows, errors = [], []
    new_done = [0]  # contador de archivos NUEVOS procesados (para el checkpoint)

    def _emit(stem, path, result, json_path):
        rws = build_rows(run_id, stem, path, result, json_path)
        for r in rws:
            if r.get("Scene") is not None:      # vídeo: no persistimos crops → sin rutas
                r["Crop Path"] = None
                r["Frame Path"] = None
            meta.attach(r, stem)
        all_rows.extend(rws)
        return len(rws)

    def _maybe_flush():
        if new_done[0] and new_done[0] % DIRECTORY_CHECKPOINT == 0:
            _flush_directory_excel(all_rows, meta, output_path, run_dir)
            console.print(f"  [dim]checkpoint → {output_path.name} ({new_done[0]} nuevos, {len(all_rows)} filas)")

    def _empty_dir(d):
        if not d.is_dir():
            return
        for f in d.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass

    def _phase1_is_stale(phase1_file):
        """Un `phase1` es HUÉRFANO si referencia crops que ya no están en disco (los
        crops son efímeros: `_empty_dir(person_crops_dir)` los borra al final de cada
        tanda). Un phase1 huérfano NO debe saltar la re-detección: si Fase B lo usa,
        `cv2.imread` falla y el vídeo sale a 0 escenas EN SILENCIO (pérdida de datos:
        el summary se escribe vacío y el vídeo queda marcado como hecho). Devuelve
        True → hay que re-detectar (regenerar phase1 + crops con Fase A)."""
        try:
            with open(phase1_file, "r", encoding="utf-8") as f:
                mp = json.load(f)
        except (OSError, ValueError):
            return True  # ilegible → re-detectar
        for sc in mp.get("scenes", []):
            persons = sc.get("persons") or []
            if not persons:
                # Escena SIN personas (2026-08-31): su único input para la Fase B es el
                # frame de escena, y `full_frames_dir` también se vacía al final de cada
                # tanda. Si ya no está, re-detectar o se perdería la ubicación EN SILENCIO
                # (mismo modo de fallo que los crops huérfanos).
                ff = sc.get("frame_file")
                if ff and not os.path.exists(ff):
                    return True
                continue
            for p in persons:
                cf = p.get("general_crop_file")
                if cf and not os.path.exists(cf):
                    return True
        return False

    # ── VÍDEOS: pipeline rápido en 2 fases (idéntico a `manual`) + limpieza de crops ──
    if vids:
        chunk = max(1, int(getattr(args, "chunk", 100)))
        yolo_device = getattr(args, "yolo_device", "cuda")
        n_chunks = (len(vids) + chunk - 1) // chunk
        console.print(f"  [cyan]{len(vids)} vídeo(s)[/] en {n_chunks} tanda(s) de {chunk}")
        vprogress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
            console=console)
        # Contador PERSISTENTE entre reanudaciones: la barra arranca en el nº de vídeos
        # que YA tienen summary (hechos en runs previos), no en 0. Al re-emitir esos
        # summaries NO se avanza (ya están contados); solo avanza el trabajo NUEVO.
        done_count = sum(1 for stem, _ in vids
                         if (json_dir / f"summary_{stem}.json").exists())
        vtask = vprogress.add_task("[cyan]Vídeos", total=len(vids), completed=done_count)
        if done_count:
            console.print(f"  [dim]↻ {done_count}/{len(vids)} vídeo(s) ya hechos (contador reanudado)")
        vprogress.start()
        for ci in range(n_chunks):
            tanda = vids[ci * chunk:(ci + 1) * chunk]
            # Reanudación siempre activa: salta lo que ya tiene summary; usa el phase1
            # SOLO si sus crops siguen en disco (un phase1 huérfano fuerza re-detección
            # y se borra para que Fase A lo regenere limpio — evita el 0-escenas falso).
            to_detect = []
            for vid_id, vpath in tanda:
                if (json_dir / f"summary_{vid_id}.json").exists():
                    continue
                p1 = phase1_root / f"phase1_{vpath.stem}.json"
                if p1.exists() and not _phase1_is_stale(p1):
                    continue
                if p1.exists():
                    try:
                        p1.unlink()  # huérfano → borrar para redetección limpia
                    except OSError:
                        pass
                to_detect.append((vid_id, vpath))
            console.rule(f"[bold]Tanda {ci + 1}/{n_chunks} — Fase A (YOLO@{yolo_device}, subproceso)")
            if to_detect:
                vprogress.stop()
                stop_gemma4()
                job = {
                    "device": yolo_device,
                    "phase1_dir": str(phase1_root),
                    "person_crops_dir": str(person_crops_dir),
                    "full_frames_dir": str(full_frames_dir),
                    "videos": [[vid_id, str(vpath)] for vid_id, vpath in to_detect],
                }
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jf:
                    json.dump(job, jf)
                    job_path = jf.name
                try:
                    subprocess.run(
                        [sys.executable, "-m", "processing.video_validation", job_path],
                        cwd=str(pkg_dir), check=False)
                finally:
                    try:
                        os.unlink(job_path)
                    except OSError:
                        pass
                    vprogress.start()
            else:
                console.print("  [dim]↻ nada que detectar en esta tanda")

            console.rule(f"[bold]Tanda {ci + 1}/{n_chunks} — Fase B (categorización)")
            for vid_id, vpath in tanda:
                vprogress.update(vtask, description=f"[cyan]{vpath.name}")
                json_path = json_dir / f"summary_{vid_id}.json"
                if json_path.exists():
                    # Vídeo ya hecho (contado en done_count) → re-emite filas para el
                    # xlsx pero NO avanza la barra (evita doble conteo al reanudar).
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            prev = json.load(f)
                        _emit(vid_id, vpath, prev.get("result"), json_path)
                        console.print(f"  [dim]↻ ya hecho: {vpath.name}")
                    except Exception as e:
                        console.print(f"  [red]·[/] Error leyendo summary de {vpath.name}: [dim]{e}")
                        errors.append(vid_id)
                    continue
                phase1_file = phase1_root / f"phase1_{vpath.stem}.json"
                if not phase1_file.exists():
                    console.print(f"  [yellow]·[/] Sin Fase A: [dim]{vpath.name}")
                    errors.append(vid_id)
                    vprogress.advance(vtask)
                    continue
                if _phase1_is_stale(phase1_file):
                    # phase1 con crops borrados y Fase A no lo regeneró (¿crash del
                    # subproceso?). NO escribir un summary de 0 escenas: borrar el
                    # phase1 huérfano y marcar error para redetección en el próximo run.
                    try:
                        phase1_file.unlink()
                    except OSError:
                        pass
                    console.print(f"  [yellow]·[/] phase1 huérfano (crops ausentes), pendiente: [dim]{vpath.name}")
                    errors.append(vid_id)
                    vprogress.advance(vtask)
                    continue
                try:
                    with open(phase1_file, "r", encoding="utf-8") as f:
                        vmeta = json.load(f)
                    result = categorize_video_scenes(vmeta, models)
                    summary = {"run_id": run_id, "img_id": vid_id,
                               "input_path": str(vpath), "result": result}
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    _emit(vid_id, vpath, result, json_path)
                    n_sc = len(result.get("scenes") or [])
                    console.print(f"  [green]✓[/] {vpath.name} [dim]({n_sc} escena(s))")
                    new_done[0] += 1
                    _maybe_flush()
                except Exception as e:
                    console.print(f"  [red]·[/] Fase B error en {vpath.name}: [dim]{e}")
                    errors.append(vid_id)
                vprogress.advance(vtask)
            # Crops/frames de esta tanda ya consumidos por la Fase B → efímeros, borrar.
            # Se borran AL FINAL de la tanda (no por vídeo): así un corte a mitad de
            # tanda conserva los crops de sus vídeos → al reanudar, los que ya tienen
            # phase1 pero no summary re-usan sus crops en Fase B. Los phase1 de la tanda
            # también se borran (sus summaries ya están escritos); si algún vídeo falló
            # la Fase B, quitar su phase1 fuerza re-detección limpia al reanudar.
            _empty_dir(person_crops_dir)
            _empty_dir(full_frames_dir)
            for vid_id, vpath in tanda:
                # phase1_<stem>.json + los crops _body.jpg/_face.jpg de este vídeo
                # (belleza/silueta) que viven en phase1_dir → todos efímeros.
                for f in list(phase1_root.glob(f"{vpath.stem}_*")) + [phase1_root / f"phase1_{vpath.stem}.json"]:
                    try:
                        if f.exists():
                            f.unlink()
                    except OSError:
                        pass
        vprogress.stop()

    # ── IMÁGENES: reanudación (salta las que ya tienen summary) + threadpool ──
    done_imgs = [(s, p) for s, p in imgs if (json_dir / f"summary_{s}.json").exists()]
    pending_imgs = [(s, p) for s, p in imgs if not (json_dir / f"summary_{s}.json").exists()]
    for stem, path in done_imgs:
        json_path = json_dir / f"summary_{stem}.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            _emit(stem, path, prev.get("result"), json_path)
        except Exception as e:
            console.print(f"  [red]·[/] Error leyendo summary de {path.name}: [dim]{e}")
            errors.append(stem)
    if done_imgs:
        console.print(f"  [dim]↻ {len(done_imgs)} imagen(es) ya hechas recargadas")

    # Flush inicial: refleja YA en el .xlsx todo lo reanudado (vídeo + imágenes hechas).
    _flush_directory_excel(all_rows, meta, output_path, run_dir)

    def _img_worker(stem, path):
        json_path = json_dir / f"summary_{stem}.json"
        if not path.exists():
            return stem, path, json_path, "missing", None, None
        try:
            result = process_image(
                path, models['model_detect'], models['model_pose'], annot_dir,
                beauty_estimator=models['beauty_estimator'],
                gender_classifier=models['gender_classifier'],
                age_classifier=models['age_classifier'],
                behaviour_classifier=models['behaviour_classifier'],
                activity_classifier=models['activity_classifier'],
                body_display_classifier=models['body_display_classifier'],
                location_classifier=models['location_classifier'],
                body_shape_classifier=models['body_shape_classifier'],
                accessory_classifier=models['accessory_classifier'],
                social_distance_classifier=models['social_distance_classifier'],
                person_attributes_classifier=models.get('person_attributes_classifier'),
                scene_context_classifier=models.get('scene_context_classifier'),
            )
            return stem, path, json_path, "ok", result, None
        except Exception as e:
            return stem, path, json_path, "error", None, e

    if pending_imgs:
        file_workers = max(1, getattr(_config, "OLLAMA_FILE_PARALLEL", 1))
        if getattr(_config, "VLM_BACKEND", "ollama") == "transformers":
            file_workers = 1
        state_lock = threading.Lock()
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
            console=console) as progress:
            task = progress.add_task("Procesando imágenes", total=len(pending_imgs))
            with ThreadPoolExecutor(max_workers=file_workers) as pool:
                futures = [pool.submit(_img_worker, s, p) for s, p in pending_imgs]
                for fut in as_completed(futures):
                    stem, path, json_path, status, result, exc = fut.result()
                    with state_lock:
                        progress.update(task, description=f"[cyan]{path.name}")
                        if status == "missing":
                            console.print(f"  [yellow]·[/] No encontrada: [dim]{path}")
                            errors.append(stem)
                        elif status == "error":
                            console.print(f"  [red]·[/] Error en {path.name}: [dim]{exc}")
                            errors.append(stem)
                        else:
                            summary = {"run_id": run_id, "img_id": stem,
                                       "input_path": str(path), "result": result}
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(summary, f, indent=2, ensure_ascii=False)
                            _emit(stem, path, result, json_path)
                            new_done[0] += 1
                            _maybe_flush()
                        # No persistir NINGÚN contenido de imagen (ahorro de disco):
                        # process_image escribe {stem}_annotated.jpg + person_crops/
                        # {stem}_person_*.jpg + person_crops_annotated/. La belleza
                        # diferida re-lee la imagen ORIGINAL (beauty_pass.py:179), no
                        # estos crops → se pueden borrar todos.
                        for f in annot_dir.rglob(f"{path.stem}_*"):
                            if f.is_file():
                                try:
                                    f.unlink()
                                except OSError:
                                    pass
                        progress.advance(task)

    # ── Belleza diferida en IMÁGENES (rostros demand/*, crop de cara, modelo especializado) ──
    console.rule("[bold]Pase de belleza (imágenes demand/*)")
    try:
        from processing.beauty_phase import run_beauty_phase, shared_beauty_backend_from_models
        df_beauty, beauty_col = run_beauty_phase(
            json_dir, models['model_pose'],
            shared_backend=shared_beauty_backend_from_models(models))
        if df_beauty is not None and not df_beauty.empty and beauty_col in df_beauty.columns:
            bmap = {
                (str(r["Img ID"]), str(r["track_id"])): r[beauty_col]
                for _, r in df_beauty.iterrows() if r.get("track_id") is not None
            }
            for r in all_rows:
                key = (str(r.get("Img ID")), str(r.get("Pers. Key")))
                if r.get("Pers. Key") is not None and key in bmap:
                    r["IA Belleza (1-10)"] = bmap[key]
            try:
                sidecar = output_path.with_name(output_path.stem + "_belleza.xlsx")
                df_beauty.to_excel(sidecar, index=False, engine="openpyxl")
                console.print(f"  [dim]belleza (auditoría) → {sidecar.name}")
            except Exception as e:
                console.print(f"  [yellow]no se pudo escribir el sidecar de belleza: {e}")
    except Exception as e:
        console.print(f"  [yellow]pase de belleza omitido: {e}")

    # ── Guardado final ──
    console.rule(f"[bold]Guardando {output_path.name}")
    _flush_directory_excel(all_rows, meta, output_path, run_dir)
    console.print(f"  [green]✓[/] {output_path}")
    console.print(f"  [dim]filas={len(all_rows)}  nuevos={new_done[0]}  errores={len(errors)}  run={run_id}")


def _stop_all_ollama():
    """Descarga TODOS los modelos cargados en las instancias Ollama del setup dual
    (pipeline :11434 + belleza :11435) al terminar main.py → libera la VRAM. NO detiene
    los servidores systemd (siguen escuchando); solo mata los runners de cada modelo
    cargado (`generate(keep_alive=0)`, equivalente a `ollama stop`). Silencioso si un
    servidor no responde. Se desactiva con `--keep-ollama`."""
    hosts = list(getattr(_config, "OLLAMA_HOSTS", ["http://localhost:11434"]))
    bh = getattr(_config, "BEAUTY_OLLAMA_HOST", None)
    if bh and bh not in hosts:
        hosts.append(bh)
    try:
        import ollama
    except ImportError:
        return
    for host in hosts:
        try:
            client = ollama.Client(host=host)
            ps = client.ps()
            models = getattr(ps, "models", None)
            if models is None and isinstance(ps, dict):
                models = ps.get("models", [])
            names = []
            for m in (models or []):
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name is None and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    names.append(name)
            for name in names:
                try:
                    client.generate(model=name, keep_alive=0)
                except Exception:
                    pass
            if names:
                console.print(f"  🛑 Ollama {host}: descargado(s) {', '.join(names)} (VRAM liberada)")
        except Exception:
            pass


def main():
    """Función principal del script."""

    parser = argparse.ArgumentParser(
        description="Análisis de participación en multimedia con YOLO26, género, edad y comportamiento"
    )

    subparsers = parser.add_subparsers(dest='mode', help='Modo de ejecución')

    # Opt-out del pase de belleza diferido (activado por defecto, ENABLE_BEAUTY_PASS).
    nobeauty_help = ("No ejecutar el pase de belleza diferido (Qwen3-VL en :11435 sobre "
                     "rostros demand/*). Por defecto SÍ se ejecuta tras la clasificación.")
    # Opt-out del apagado de modelos Ollama al terminar (por defecto SÍ se descargan).
    keepollama_help = ("No descargar los modelos de Ollama al terminar (por defecto se hace "
                       "`ollama stop` de todo en :11434 y :11435 para liberar VRAM).")

    parser_single = subparsers.add_parser('single', help='Analizar un solo archivo multimedia')
    parser_single.add_argument(
        "input_file",
        type=str,
        help="Ruta al archivo multimedia (imagen o video) a analizar"
    )
    parser_single.add_argument("--no-beauty", action="store_true", help=nobeauty_help)
    parser_single.add_argument("--keep-ollama", action="store_true", help=keepollama_help)

    parser_batch = subparsers.add_parser('batch', help='Analizar batch desde CSV')
    parser_batch.add_argument(
        "csv_file",
        type=str,
        help="Ruta al CSV (masculinity-IG-limpio.csv)"
    )
    parser_batch.add_argument(
        "--sample-size", "-n",
        type=int,
        default=400,
        help="Número de archivos a muestrear (default: 400)"
    )
    parser_batch.add_argument(
        "--multimedia-dir", "-m",
        type=str,
        default=None,
        help="Directorio con archivos multimedia descargados (default: multimedia_downloads/ junto al CSV)"
    )
    parser_batch.add_argument(
        "--output-csv", "-o",
        type=str,
        default=None,
        help="Ruta al CSV de salida (default: <nombre>_analizado.csv)"
    )
    parser_batch.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del muestreo (default: 42)"
    )
    parser_batch.add_argument("--no-beauty", action="store_true", help=nobeauty_help)
    parser_batch.add_argument("--keep-ollama", action="store_true", help=keepollama_help)

    parser_manual = subparsers.add_parser(
        'manual',
        help='Procesar imágenes únicas de un Excel manual → Excel IA (qwen_*.xlsx) '
             'consumible por validacion_imagenes/alinear_categorias.py y comparar_validacion.py'
    )
    parser_manual.add_argument(
        "manual_xlsx",
        type=str,
        help="Ruta al Excel manual (debe tener columnas 'Img ID' y 'Ruta Img.')"
    )
    parser_manual.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta al Excel IA de salida (default: qwen_<stem_del_manual>.xlsx junto al manual)"
    )
    parser_manual.add_argument(
        "--run-prefix",
        type=str,
        default="qwen",
        help="Prefijo de Run ID y de la carpeta runs/<prefijo>_<timestamp>/ (default: 'qwen')"
    )
    parser_manual.add_argument(
        "--only-img-ids",
        type=str,
        default=None,
        help="Procesar solo estos Img ID (lista CSV o ruta a un .txt con un id por línea)"
    )
    parser_manual.add_argument(
        "--append",
        action="store_true",
        help="Si --output existe, sustituye las filas con los mismos Img IDs y conserva el resto"
    )
    parser_manual.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directorio raíz para runs/<run_id>/ (default: junto al Excel manual)"
    )
    parser_manual.add_argument("--no-beauty", action="store_true", help=nobeauty_help)
    parser_manual.add_argument("--keep-ollama", action="store_true", help=keepollama_help)
    parser_manual.add_argument(
        "--chunk", type=int, default=100,
        help="Vídeo: tamaño de tanda del pipeline en 2 fases (Fase A YOLO-GPU + Fase B gemma4). Default 100."
    )
    parser_manual.add_argument(
        "--yolo-device", type=str, default="cuda",
        help="Vídeo: dispositivo para YOLO en la Fase A de detección (default: cuda)."
    )
    parser_manual.add_argument(
        "--resume", action="store_true",
        help="Reanuda el run más reciente con el mismo --run-prefix: salta vídeos cuyo summary "
             "ya existe (y su detección si phase1 ya existe). Reconstruye el xlsx completo al final."
    )

    parser_dir = subparsers.add_parser(
        'carpeta',
        help='Procesar TODO un directorio de multimedia (multimedia_downloads/) → un Excel único '
             'con metadatos de los CSV. Reanuda solo, guarda por lotes, belleza siempre.'
    )
    parser_dir.add_argument(
        "--multimedia-dir", "-m", type=str, default=None,
        help="Directorio a procesar (default: <repo>/multimedia_downloads)"
    )
    parser_dir.add_argument(
        "--output", "-o", type=str, default=None,
        help="Excel de salida .xlsx (default: <repo>/multimedia_downloads_analizado.xlsx)"
    )
    parser_dir.add_argument(
        "--run-prefix", type=str, default="carpeta",
        help="Prefijo de Run ID y de runs/<prefijo>_<timestamp>/ (default: 'carpeta')"
    )
    parser_dir.add_argument(
        "--runs-dir", type=str, default=None,
        help="Directorio raíz para runs/<run_id>/ (default: junto al directorio multimedia)"
    )
    parser_dir.add_argument(
        "--metadata-dir", type=str, default=None,
        help="Directorio con los CSV de metadatos (cadena RAW_7→…→RAW_2→IG-Raw→nuevas→limpio; "
             "ver DEFAULT_SOURCES). Default: <repo>. Los archivos sin match en NINGUNA fuente NO se procesan."
    )
    parser_dir.add_argument(
        "--chunk", type=int, default=100,
        help="Vídeo: tamaño de tanda del pipeline en 2 fases. Default 100."
    )
    parser_dir.add_argument(
        "--yolo-device", type=str, default="cuda",
        help="Vídeo: dispositivo para YOLO en la Fase A (default: cuda)."
    )
    parser_dir.add_argument("--keep-ollama", action="store_true", help=keepollama_help)

    parser_compare = subparsers.add_parser(
        'compare',
        help='Generar XLSX de comparación IA vs anotación manual'
    )
    parser_compare.add_argument(
        "manual_xlsx",
        type=str,
        help="Ruta al Excel de anotaciones manuales (validacion_manual_solo.xlsx)"
    )
    parser_compare.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta del XLSX de salida (default: comparacion_ia_manual.xlsx junto al manual)"
    )
    parser_compare.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="No incrustar imágenes en el XLSX (más rápido, archivo más pequeño)"
    )

    args = parser.parse_args()

    # Determinismo por defecto: el cliente va SIEMPRE en serie (FILE/OUTER/INNER=1,
    # ya son los defaults de config.py) → 1 sola request en vuelo, el servidor Ollama
    # nunca batchea peticiones concurrentes, que es la única fuente de variación
    # run-a-run con greedy+seed=42. También exporta OLLAMA_NUM_PARALLEL=1 (efectivo
    # solo si un server se lanza desde este proceso; con el systemd ya activo, la
    # serialización del cliente basta). Subir el paralelismo degrada el κ → no se expone.
    os.environ["OLLAMA_NUM_PARALLEL"] = "1"
    _config.OLLAMA_FILE_PARALLEL = 1
    _config.OLLAMA_OUTER_PARALLEL = 1
    _config.OLLAMA_INNER_PARALLEL = 1
    print("  🔒 Cliente en serie (FILE/OUTER/INNER=1) + OLLAMA_NUM_PARALLEL=1")

    # Opt-out del pase de belleza diferido (por defecto activo, ver config.ENABLE_BEAUTY_PASS).
    if getattr(args, "no_beauty", False):
        _config.ENABLE_BEAUTY_PASS = False
        print("  💄 Pase de belleza desactivado (--no-beauty)")

    # Al terminar (cualquier salida: éxito, error o excepción), descargar TODOS los
    # modelos de Ollama de ambas instancias para liberar VRAM. Opt-out con --keep-ollama.
    # atexit cubre todas las rutas de return de main() (single/batch/manual).
    if args.mode in ("single", "batch", "manual", "carpeta") and not getattr(args, "keep_ollama", False):
        import atexit
        atexit.register(_stop_all_ollama)

    if args.mode == 'compare':
        from processing.generar_comparacion import generar_xlsx_comparacion
        generar_xlsx_comparacion(
            args.manual_xlsx,
            args.output,
            include_images=not args.no_images,
        )
        return

    if args.mode == 'batch':
        process_batch(
            csv_path=args.csv_file,
            sample_size=args.sample_size,
            multimedia_dir=args.multimedia_dir,
            output_csv=args.output_csv,
            seed=args.seed
        )
        return

    if args.mode == 'manual':
        models = load_all_models()
        if models is None:
            return
        _process_manual(args, models)
        return

    if args.mode == 'carpeta':
        # Belleza SIEMPRE en este modo (imágenes: pase diferido demand/*, crop de cara;
        # vídeo: inline por escena). Forzar los flags ANTES de cargar los modelos para
        # que el beauty_estimator inline de vídeo se cargue.
        _config.ENABLE_BEAUTY_PASS = True
        _config.ENABLE_BEAUTY_ESTIMATION = True
        # Este modo NO persiste imágenes → no perder tiempo escribiéndolas para
        # borrarlas después (anotadas + person_crops de imagen; los crops de VÍDEO
        # sí se escriben: son el input de la Fase B, efímeros por tanda).
        _config.SAVE_ANNOTATED_OUTPUTS = False
        print("  💄 Belleza SIEMPRE activa (imágenes demand/* + vídeo por escena)")
        print("  🚫 Sin salidas anotadas (no se persisten en este modo)")
        models = load_all_models()
        if models is None:
            return
        _process_directory(args, models)
        return

    elif args.mode == 'single':
        input_path = Path(args.input_file)
    else:
        # Compatibilidad con uso anterior: python main.py <archivo>
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ('single', 'batch'):
            input_path = Path(sys.argv[1])
        else:
            parser.print_help()
            return

    if not input_path.exists():
        error(f"El archivo no existe: {input_path}")
        return

    header("Análisis de participación")
    _on, _off = "[green]on[/]", "[dim]off[/]"
    _scene = (
        f"  ([dim]escena: {'on' if (ENABLE_SCENE_DETECTION and PYSCENEDETECT_AVAILABLE) else 'off'}[/])"
        if (ENABLE_BEHAVIOUR_CLASSIFICATION or ENABLE_ACTIVITY_CLASSIFICATION) else ""
    )
    features = [
        ("Archivo", input_path.name),
        ("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("", ""),
        ("Personas", _on),
        ("Pose", _on if ENABLE_POSE_ESTIMATION else _off),
        ("Tracking", _on if ENABLE_TRACKING else _off),
        ("Género", _on if ENABLE_GENDER_CLASSIFICATION else _off),
        ("Edad", _on if ENABLE_AGE_CLASSIFICATION else _off),
        ("Comportamiento", _on if ENABLE_BEHAVIOUR_CLASSIFICATION else _off),
        ("Actividades", (_on if ENABLE_ACTIVITY_CLASSIFICATION else _off) + _scene),
        ("Tipo de cuerpo", _on if ENABLE_BODY_SHAPE_CLASSIFICATION else _off),
        ("Accesorios", _on if ENABLE_ACCESSORY_CLASSIFICATION else _off),
        ("OCR", _on if ENABLE_OCR else _off),
        ("Distancia social", _on if ENABLE_SOCIAL_DISTANCE else _off),
        ("Belleza", _on if ENABLE_BEAUTY_ESTIMATION else _off),
    ]
    kv_table("", features)

    # ========================================================================
    # CARGAR TODOS LOS MODELOS (YOLO + backend VLM compartido)
    # ========================================================================
    models = load_all_models()
    if models is None:
        return

    model_detect = models['model_detect']
    model_pose = models['model_pose']
    behaviour_classifier = models['behaviour_classifier']
    activity_classifier = models['activity_classifier']
    body_display_classifier = models['body_display_classifier']
    location_classifier = models['location_classifier']
    body_shape_classifier = models['body_shape_classifier']
    accessory_classifier = models['accessory_classifier']
    ocr_classifier = models['ocr_classifier']
    gender_classifier = models['gender_classifier']
    age_classifier = models['age_classifier']
    social_distance_classifier = models['social_distance_classifier']
    beauty_estimator = models['beauty_estimator']
    person_attributes_classifier = models.get('person_attributes_classifier')
    scene_context_classifier = models.get('scene_context_classifier')

    header("Procesando multimedia")

    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    file_ext = input_path.suffix.lower()

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "configuration": {
            "detection_model": DETECTION_MODEL,
            "pose_model": POSE_MODEL if ENABLE_POSE_ESTIMATION else None,
            "gender_model": BEHAVIOUR_MODEL_NAME if (ENABLE_GENDER_CLASSIFICATION or ENABLE_AGE_CLASSIFICATION) else None,
            "age_model": BEHAVIOUR_MODEL_NAME if (ENABLE_GENDER_CLASSIFICATION or ENABLE_AGE_CLASSIFICATION) else None,
            "behaviour_model": BEHAVIOUR_MODEL_NAME if ENABLE_BEHAVIOUR_CLASSIFICATION else None,
            "beauty_model": f"{BEHAVIOUR_MODEL_NAME} + LoRA" if ENABLE_BEAUTY_ESTIMATION and USE_FINETUNED_MODEL else BEHAVIOUR_MODEL_NAME if ENABLE_BEAUTY_ESTIMATION else None,
            "finetuned_lora_path": FINETUNED_LORA_PATH if USE_FINETUNED_MODEL else None,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "tracker": TRACKER_CONFIG if ENABLE_TRACKING else None,
            "enabled_features": {
                "pose_estimation": ENABLE_POSE_ESTIMATION,
                "tracking": ENABLE_TRACKING,
                "gender_classification": ENABLE_GENDER_CLASSIFICATION,
                "age_classification": ENABLE_AGE_CLASSIFICATION,
                "behaviour_classification": ENABLE_BEHAVIOUR_CLASSIFICATION,
                "beauty_estimation": ENABLE_BEAUTY_ESTIMATION
            }
        },
        "result": None
    }

    if file_ext in image_extensions:
        info("  Tipo: [cyan]imagen")

        if ENABLE_COLLAGE_DETECTION:
            is_collage, panels, method = detect_collage_panels(
                input_path,
                min_panels=COLLAGE_MIN_PANELS,
                min_panel_size_percent=COLLAGE_MIN_PANEL_SIZE_PERCENT,
                model_detect=model_detect
            )

            if is_collage:
                info(f"  Collage detectado: [cyan]{len(panels)}[/] paneles ([dim]{method}[/])")

                temp_dir = OUTPUT_DIR / f"temp_panels_{input_path.stem}"
                temp_dir.mkdir(parents=True, exist_ok=True)

                panel_paths = extract_and_save_panels(input_path, panels, temp_dir)

                panel_results = []
                for idx, panel_path in enumerate(panel_paths, 1):
                    info(f"\n  [bold]Panel {idx}/{len(panel_paths)}[/]  [dim]{panel_path.name}")

                    panel_result = process_image(
                        panel_path, model_detect, model_pose, OUTPUT_DIR,
                        beauty_estimator, gender_classifier, age_classifier,
                        behaviour_classifier, activity_classifier, body_display_classifier,
                        location_classifier, body_shape_classifier, accessory_classifier, social_distance_classifier,
                        person_attributes_classifier=person_attributes_classifier,
                        scene_context_classifier=scene_context_classifier,
                    )

                    panel_results.append({
                        "panel_id": idx,
                        "panel_name": panel_path.name,
                        "coordinates": {
                            "x": int(panels[idx-1][0]),
                            "y": int(panels[idx-1][1]),
                            "width": int(panels[idx-1][2]),
                            "height": int(panels[idx-1][3])
                        },
                        "result": panel_result
                    })

                result = {
                    "collage_detected": True,
                    "detection_method": method,
                    "total_panels": len(panels),
                    "sub_images": panel_results
                }
                summary["result"] = result

                success(f"  Collage: [cyan]{len(panels)}[/] paneles procesados")

                detection_viz_path = visualize_collage_detection(
                    input_path, panels, method, OUTPUT_DIR
                )
                if detection_viz_path:
                    info(f"  [dim]detección: {detection_viz_path.name}")

                montage_path = create_panels_montage(
                    input_path, panel_paths, panels, OUTPUT_DIR
                )
                if montage_path:
                    info(f"  [dim]montaje: {montage_path.name}")

                result["visualizations"] = {
                    "detection": str(detection_viz_path) if detection_viz_path else None,
                    "montage": str(montage_path) if montage_path else None
                }

            else:
                result = process_image(
                    input_path, model_detect, model_pose, OUTPUT_DIR,
                    beauty_estimator, gender_classifier, age_classifier,
                    behaviour_classifier, activity_classifier, body_display_classifier,
                    location_classifier, body_shape_classifier, accessory_classifier, social_distance_classifier,
                    person_attributes_classifier=person_attributes_classifier,
                    scene_context_classifier=scene_context_classifier,
                )
                summary["result"] = result
        else:
            result = process_image(
                input_path, model_detect, model_pose, OUTPUT_DIR,
                beauty_estimator, gender_classifier, age_classifier,
                behaviour_classifier, activity_classifier, body_display_classifier,
                location_classifier, body_shape_classifier, accessory_classifier, social_distance_classifier,
                person_attributes_classifier=person_attributes_classifier,
                scene_context_classifier=scene_context_classifier,
            )
            summary["result"] = result

    elif file_ext in video_extensions:
        info("  Tipo: [cyan]video")

        if ENABLE_COLLAGE_DETECTION:
            cap_test = cv2.VideoCapture(str(input_path))
            ret, first_frame = cap_test.read()
            cap_test.release()

            if ret:
                temp_frame_path = OUTPUT_DIR / f"temp_first_frame_{input_path.stem}.jpg"
                cv2.imwrite(str(temp_frame_path), first_frame)

                is_collage, panels, method = detect_collage_panels(
                    temp_frame_path,
                    min_panels=COLLAGE_MIN_PANELS,
                    min_panel_size_percent=COLLAGE_MIN_PANEL_SIZE_PERCENT,
                    model_detect=model_detect
                )

                if temp_frame_path.exists():
                    temp_frame_path.unlink()

                if is_collage:
                    info(f"  Collage en video: [cyan]{len(panels)}[/] paneles ([dim]{method}[/])")

                    result = process_video_with_panels(
                        input_path, panels, model_detect, model_pose, OUTPUT_DIR,
                        beauty_estimator, gender_classifier, age_classifier,
                        behaviour_classifier, activity_classifier, body_display_classifier,
                        location_classifier, body_shape_classifier, social_distance_classifier,
                        person_attributes_classifier=person_attributes_classifier,
                        scene_context_classifier=scene_context_classifier,
                    )

                    if result:
                        result["collage_detected"] = True
                        result["detection_method"] = method
                        result["total_panels"] = len(panels)
                        result["panels_coordinates"] = [
                            {"panel_id": i+1, "x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                            for i, (x, y, w, h) in enumerate(panels)
                        ]

                    summary["result"] = result
                else:
                    result = process_video(
                        input_path, model_detect, model_pose, OUTPUT_DIR,
                        beauty_estimator, gender_classifier, age_classifier,
                        behaviour_classifier, activity_classifier, body_display_classifier,
                        location_classifier, body_shape_classifier, accessory_classifier, ocr_classifier, social_distance_classifier,
                        person_attributes_classifier=person_attributes_classifier,
                        scene_context_classifier=scene_context_classifier,
                    )
                    summary["result"] = result
            else:
                warn("  No se pudo leer el primer frame, procesando sin detección de collages")
                result = process_video(
                    input_path, model_detect, model_pose, OUTPUT_DIR,
                    beauty_estimator, gender_classifier, age_classifier,
                    behaviour_classifier, activity_classifier, body_display_classifier,
                    location_classifier, body_shape_classifier, accessory_classifier, ocr_classifier, social_distance_classifier,
                    person_attributes_classifier=person_attributes_classifier,
                    scene_context_classifier=scene_context_classifier,
                )
                summary["result"] = result
        else:
            result = process_video(
                input_path, model_detect, model_pose, OUTPUT_DIR,
                beauty_estimator, gender_classifier, age_classifier,
                behaviour_classifier, activity_classifier, body_display_classifier,
                location_classifier, body_shape_classifier, accessory_classifier, ocr_classifier, social_distance_classifier,
                person_attributes_classifier=person_attributes_classifier,
                scene_context_classifier=scene_context_classifier,
            )
            summary["result"] = result

    else:
        error(f"Formato no soportado: {file_ext}")
        info(f"  [dim]imágenes: {', '.join(image_extensions)}")
        info(f"  [dim]videos: {', '.join(video_extensions)}")
        return

    # ========================================================================
    # GUARDAR RESUMEN
    # ========================================================================
    if result is not None:
        summary_path = OUTPUT_DIR / f"summary_{input_path.stem}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        header("Resumen del análisis")
        info(f"  [dim]Salida:  {OUTPUT_DIR}")
        info(f"  [dim]Resumen: {summary_path.name}\n")

        _render_result_summary(result, file_ext, image_extensions)

        if file_ext in image_extensions:
            # Pase de belleza diferido de IMAGEN (2026-07-21: ≥4 keypoints, sin gating
            # demand/*). Lee el summary recién escrito (filtrado a este stem) y puntúa
            # toda cara con keypoints/tamaño suficientes. Degrada con elegancia.
            from processing.beauty_phase import run_beauty_phase, shared_beauty_backend_from_models
            df_beauty, beauty_col = run_beauty_phase(
                OUTPUT_DIR, model_pose, only_stem=input_path.stem,
                shared_backend=shared_beauty_backend_from_models(models))
            if df_beauty is not None and not df_beauty.empty and beauty_col in df_beauty.columns:
                kv_table("Belleza (≥4 kpts)", [
                    (f"persona {r.get('track_id') or r['person_idx']}", str(r[beauty_col]))
                    for _, r in df_beauty.iterrows()
                ])

                # Re-dibujar el _annotated.jpg con el chip de belleza: el pase diferido
                # calcula la belleza DESPUÉS de guardar la imagen anotada, así que sin esto
                # el chip no aparece. Reusa el summary + caches, sin re-llamar al VLM.
                from processing.image import rerender_annotated_with_beauty
                beauty_by_track = {
                    r.get("track_id"): r[beauty_col]
                    for _, r in df_beauty.iterrows()
                    if r.get("track_id") is not None
                }
                if rerender_annotated_with_beauty(
                        input_path, OUTPUT_DIR, result, beauty_by_track,
                        model_detect, model_pose,
                        social_distance_classifier=social_distance_classifier):
                    info("  ↻ imagen anotada re-dibujada con chip de belleza")
        else:
            # VÍDEO (2026-07-21: ≥4 keypoints, sin gating demand/*): la belleza va por el
            # pase DIFERIDO interno de process_video (chip ya renderizado en el mp4). Aquí
            # solo se reporta; NO se lanza run_beauty_phase (los summaries de vídeo no
            # llevan bbox por persona).
            vb = (result or {}).get("beauty_classifications") or []
            if vb:
                kv_table("Belleza (≥4 kpts)", [
                    (f"escena {c.get('scene')} · track {c.get('track_id')}",
                     str(c.get('score')))
                    for c in vb
                ])

        success("\nAnálisis completado.\n")

    else:
        error("\nError durante el procesamiento\n")


def _render_result_summary(result, file_ext, image_extensions):
    """Imprime tablas compactas con métricas y distribuciones del resultado."""

    def _dist_from_items(items, key):
        c = Counter(it.get(key) for it in items if it.get(key))
        return [(k, v) for k, v in c.most_common()]

    def _accessory_dist(items):
        from models.accessory import ACCESSORY_CATEGORIES
        counts = Counter()
        for it in items:
            for cat in ACCESSORY_CATEGORIES:
                if it.get(cat) == 1:
                    counts[cat] += 1
        return [(k, v) for k, v in counts.most_common()]

    def _render_image_categories(persons, gen, age, beh, act, bdis, loc, bsh, acc, soc, beauty):
        kv_table("Detecciones", [("Personas", persons)])
        occs = [g.get('occupancy') for g in gen if g.get('occupancy') is not None]
        if occs:
            kv_table("Tamaño (% frame)", [
                ("Media", f"{sum(occs)/len(occs):.1f}%"),
                ("Máx",   f"{max(occs):.1f}%"),
            ])
        image_pairs = [
            (ENABLE_GENDER_CLASSIFICATION,       "Género",            gen,  'gender'),
            (ENABLE_AGE_CLASSIFICATION,          "Edad",              age,  'age_group'),
            (ENABLE_BEHAVIOUR_CLASSIFICATION,    "Comportamiento",    beh,  'behaviour'),
            (ENABLE_ACTIVITY_CLASSIFICATION,     "Actividad",         act,  'activity'),
            (ENABLE_BODY_DISPLAY_CLASSIFICATION, "Exposición cuerpo", bdis, 'body_display'),
            (ENABLE_LOCATION_CLASSIFICATION,     "Ubicación",         loc,  'location'),
            (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Adiposity",         bsh,  'body_weight'),
            (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Musculatura",       bsh,  'muscle'),
            (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Vestimenta",        bsh,  'attire'),
            (ENABLE_SOCIAL_DISTANCE,             "Distancia social",  soc,  'category'),
        ]
        for enabled, title, items, key in image_pairs:
            if enabled and items:
                dist_table(title, _dist_from_items(items, key))
        if ENABLE_ACCESSORY_CLASSIFICATION and acc:
            dist = _accessory_dist(acc)
            if dist:
                dist_table("Accesorios", dist)
        if ENABLE_BEAUTY_ESTIMATION and beauty:
            scores = [b.get('score') for b in beauty if isinstance(b.get('score'), (int, float))]
            if scores:
                kv_table("Belleza", [
                    ("Evaluaciones", len(scores)),
                    ("Media", f"{sum(scores)/len(scores):.1f}"),
                    ("Min/Max", f"{min(scores):.0f} / {max(scores):.0f}"),
                ])
            else:
                info(f"  Belleza: [cyan]{len(beauty)}[/] evaluaciones")

    if file_ext in image_extensions:
        if result.get('collage_detected'):
            kv_table("Collage", [
                ("Método", result['detection_method']),
                ("Paneles", result['total_panels']),
            ])
            total_persons = sum(p['result'].get('persons_detected', 0) for p in result['sub_images'])
            agg = {k: [] for k in (
                'gender_classifications', 'age_classifications', 'behaviour_classifications',
                'activity_classifications', 'body_display_classifications',
                'location_classifications', 'body_shape_classifications',
                'accessory_classifications', 'social_distance_classifications', 'beauty_scores',
            )}
            for panel in result['sub_images']:
                pr = panel['result']
                for k in agg:
                    agg[k].extend(pr.get(k, []) or [])
            _render_image_categories(
                total_persons, agg['gender_classifications'], agg['age_classifications'],
                agg['behaviour_classifications'], agg['activity_classifications'],
                agg['body_display_classifications'], agg['location_classifications'],
                agg['body_shape_classifications'], agg['accessory_classifications'],
                agg['social_distance_classifications'], agg['beauty_scores'],
            )
        else:
            _render_image_categories(
                result.get('persons_detected', 0),
                result.get('gender_classifications', []),
                result.get('age_classifications', []),
                result.get('behaviour_classifications', []),
                result.get('activity_classifications', []),
                result.get('body_display_classifications', []),
                result.get('location_classifications', []),
                result.get('body_shape_classifications', []),
                result.get('accessory_classifications', []),
                result.get('social_distance_classifications', []),
                result.get('beauty_scores', []),
            )
        return

    # Video
    if result.get('collage_detected'):
        kv_table("Collage en video", [
            ("Paneles", result['total_panels']),
            ("Duración", f"{result['duration_seconds']:.1f}s ({result['total_frames']} frames)"),
        ])
        if result.get('combined_video'):
            info(f"  [dim]Video combinado: {Path(result['combined_video']).name}")
        for panel_data in result.get('panels', []):
            pr = panel_data['result']
            rows = []
            if ENABLE_TRACKING and pr.get('unique_persons_tracked'):
                rows.append(("Personas únicas", pr['unique_persons_tracked']))
            if pr.get('avg_persons_per_frame') is not None:
                rows.append(("Avg p/frame", f"{pr['avg_persons_per_frame']:.2f}"))
            if rows:
                kv_table(f"Panel {panel_data['panel_id']}", rows)
        return

    rows = [("Duración", f"{result['duration_seconds']:.1f}s ({result['total_frames']} frames)")]
    if ENABLE_TRACKING and 'unique_persons_tracked' in result:
        rows.append(("Personas únicas", result['unique_persons_tracked']))
    if 'avg_persons_per_frame' in result:
        rows.append(("Avg p/frame", result['avg_persons_per_frame']))
    _occ = result.get('occupancy_statistics') or {}
    if _occ.get('mean') is not None:
        rows.append(("Tamaño medio", f"{_occ['mean']:.1f}%"))
        rows.append(("Tamaño máx",   f"{_occ['max']:.1f}%"))
    kv_table("Video", rows)

    def _from_stats(stats_key, dist_key):
        stats = result.get(stats_key)
        if not stats:
            return None
        dist = stats.get(dist_key)
        if dist is None:
            # El path de vídeo nombra la distribución distinto que el de imagen
            # (p.ej. body_shape → 'silhouette_distribution'); tolerar cualquiera.
            dist = next((v for k, v in stats.items() if k.endswith("_distribution")), None)
        if not dist:
            return None
        return list(dist.items())

    pairs = [
        (ENABLE_GENDER_CLASSIFICATION,       "Género",            "gender_statistics",         "gender_distribution"),
        (ENABLE_AGE_CLASSIFICATION,          "Edad",              "age_statistics",            "age_group_distribution"),
        (ENABLE_BEHAVIOUR_CLASSIFICATION,    "Comportamiento",    "behaviour_statistics",      "behaviour_distribution"),
        (ENABLE_ACTIVITY_CLASSIFICATION,     "Actividad",         "activity_statistics",       "activity_distribution"),
        (ENABLE_BODY_DISPLAY_CLASSIFICATION, "Exposición cuerpo", "body_display_statistics",   "body_display_distribution"),
        (ENABLE_LOCATION_CLASSIFICATION,     "Ubicación",         "location_statistics",       "location_distribution"),
        (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Adiposity",         "body_shape_statistics",     "body_shape_distribution"),
        (ENABLE_ACCESSORY_CLASSIFICATION,    "Accesorios",        "accessory_statistics",      "accessory_distribution"),
        (ENABLE_SOCIAL_DISTANCE,             "Distancia social",  "social_distance_statistics", "category_distribution"),
    ]
    for enabled, title, sk, dk in pairs:
        if not enabled:
            continue
        d = _from_stats(sk, dk)
        if d:
            dist_table(title, d)

    if ENABLE_OCR and result.get('ocr_results'):
        info(f"\n  OCR: [cyan]{len(result['ocr_results'])}[/] escenas")
        for scene_num, ocr_data in sorted(result['ocr_results'].items(), key=lambda x: float(x[0])):
            text_preview = (ocr_data.get('text') or '').strip()
            if len(text_preview) > 80:
                text_preview = text_preview[:77] + "..."
            info(f"    [dim]escena {scene_num}:[/] {text_preview}")

    if ENABLE_BEAUTY_ESTIMATION and result.get('beauty_statistics'):
        s = result['beauty_statistics']
        kv_table("Belleza", [
            ("Media", f"{s['mean']:.1f}"),
            ("Mediana", f"{s['median']:.1f}"),
            ("Mín", f"{s['min']:.1f}"),
            ("Máx", f"{s['max']:.1f}"),
            ("σ", f"{s['std']:.1f}"),
        ])


if __name__ == "__main__":
    main()
