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
        "IA Nº pers.", "IA Gen.", "IA Edad", "IA Comport.", "IA Activ.",
        "IA Exp. Cp.", "IA Ubic.", "IA Peso", "IA Musculatura", "IA Dist. Soc.", "IA Belleza (1-10)",
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

    def build_video_rows(img_id, img_path, result, json_path):
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
            for p in scene.get("persons", []):
                acc = p.get("accessories") or {}
                row = {
                    "Estado":        "ok",
                    "IA Nº pers.":   n_tracks,
                    "IA Gen.":       p.get("gender"),
                    "IA Edad":       p.get("age_group"),
                    "IA Comport.":   p.get("behaviour"),
                    "IA Activ.":     p.get("activity"),
                    "IA Exp. Cp.":   p.get("body_display"),
                    "IA Ubic.":      p.get("location"),
                    "IA Peso":       p.get("weight"),
                    "IA Musculatura": p.get("muscle"),
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

    def build_rows(img_id, img_path, result, json_path):
        rows = []
        if result is None:
            return rows
        # Vídeo: result trae la estructura por escena (validation_dump). Una fila
        # por (escena, persona) en vez de la lógica de imagen (una por persona).
        if result.get("scenes") is not None:
            return build_video_rows(img_id, img_path, result, json_path)
        n = result.get("persons_detected", 0)
        # n_yolo: detecciones totales de YOLO26 (incluye las filtradas por
        # BBOX_MIN_FRAME_RATIO que no entran al VLM). Es el valor que se
        # reporta como `IA Nº pers.` — el recuento de personas por imagen no
        # debe depender de qué crops eran lo bastante grandes para el VLM.
        # Fallback al persons_detected antiguo para JSONs pre-cambio.
        n_yolo = result.get("persons_detected_yolo", n)
        vlm_status = result.get("vlm_status") or "ok"

        if n == 0:
            rows.append({
                "Img. orig.":  None,
                "Img. anot.":  None,
                "Estado":      vlm_status,
                "IA Nº pers.": n_yolo,
                "IA Gen.": None, "IA Edad": None, "IA Comport.": None, "IA Activ.": None,
                "IA Exp. Cp.": None, "IA Ubic.": None, "IA Dist. Soc.": None,
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
                "IA Gen.":        gen.get("gender")        if gen  else None,
                "IA Edad":        age.get("age_group")     if age  else None,
                "IA Comport.":    beh.get("behaviour")     if beh  else None,
                "IA Activ.":      act.get("activity")      if act  else None,
                "IA Exp. Cp.":    bdis.get("body_display") if bdis else None,
                "IA Ubic.":       loc.get("location")      if loc  else None,
                "IA Peso":        bsh.get("body_weight")   if bsh  else None,
                "IA Musculatura": bsh.get("muscle")        if bsh  else None,
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

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

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
                        all_rows.extend(build_rows(vid_id, vpath, prev.get("result"), json_path))
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
                    all_rows.extend(build_rows(vid_id, vpath, result, json_path))
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
                        all_rows.extend(build_rows(img_id, img_path, result, json_path))
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
        df_out["IA Belleza (1-10)"] = [
            bmap.get((str(iid), str(pk)))
            for iid, pk in zip(df_out["Img ID"], df_out["Pers. Key"])
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
    if args.mode in ("single", "batch", "manual") and not getattr(args, "keep_ollama", False):
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

        # Pase de belleza diferido (Qwen3-VL @ :11435, solo demand/*). Lee el
        # summary recién escrito (filtrado a este stem), hace `ollama stop` gemma4
        # y puntúa los rostros demand/* de esta imagen. Degrada con elegancia.
        from processing.beauty_phase import run_beauty_phase, shared_beauty_backend_from_models
        df_beauty, beauty_col = run_beauty_phase(
            OUTPUT_DIR, model_pose, only_stem=input_path.stem,
            shared_backend=shared_beauty_backend_from_models(models))
        if df_beauty is not None and not df_beauty.empty and beauty_col in df_beauty.columns:
            kv_table("Belleza (demand/*)", [
                (f"persona {r.get('track_id') or r['person_idx']}", str(r[beauty_col]))
                for _, r in df_beauty.iterrows()
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
        image_pairs = [
            (ENABLE_GENDER_CLASSIFICATION,       "Género",            gen,  'gender'),
            (ENABLE_AGE_CLASSIFICATION,          "Edad",              age,  'age_group'),
            (ENABLE_BEHAVIOUR_CLASSIFICATION,    "Comportamiento",    beh,  'behaviour'),
            (ENABLE_ACTIVITY_CLASSIFICATION,     "Actividad",         act,  'activity'),
            (ENABLE_BODY_DISPLAY_CLASSIFICATION, "Exposición cuerpo", bdis, 'body_display'),
            (ENABLE_LOCATION_CLASSIFICATION,     "Ubicación",         loc,  'location'),
            (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Peso",              bsh,  'body_weight'),
            (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Musculatura",       bsh,  'muscle'),
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
        (ENABLE_BODY_SHAPE_CLASSIFICATION,   "Peso",              "body_shape_statistics",     "body_shape_distribution"),
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
