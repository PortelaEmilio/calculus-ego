import ast
import io
import json
import threading
import time
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from datetime import datetime

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

import config
from config import OUTPUT_DIR
from models.loader import load_all_models
from models.accessory import ACCESSORY_CATEGORIES
from processing.image import process_image
from processing.video import process_video
from ui import console as _console, header, info, warn, error, kv_table


@contextmanager
def _noop():
    yield

ACCESSORY_CSV_COLUMNS = {
    "makeup": "ia_accesorio_maquillaje",
    "tattoos": "ia_accesorio_tatuajes",
    "bags": "ia_accesorio_bolsos",
    "belts": "ia_accesorio_cinturones",
    "jewelry": "ia_accesorio_joyeria",
    "headwear": "ia_accesorio_sombrero",
    "eyewear": "ia_accesorio_gafas",
}


def resolve_file_path(row, multimedia_dir):
    """
    Resuelve la ruta del archivo multimedia descargado para una fila del CSV.

    Args:
        row: Fila del DataFrame
        multimedia_dir: Directorio base de descargas

    Returns:
        Path al archivo o None si no se encuentra
    """
    sid = str(row['id'])
    ct = row['content_type']

    if ct == 'videos':
        path = Path(multimedia_dir) / f"{sid}.mp4"
    elif ct == 'photos':
        path = Path(multimedia_dir) / f"{sid}.jpg"
    elif ct == 'albums':
        try:
            mm = ast.literal_eval(row['multimedia'])
            if mm:
                mid = str(mm[0]['id'])
                ext = '.jpg' if mm[0].get('type') == 'photo' else '.mp4'
                path = Path(multimedia_dir) / f"{mid}{ext}"
            else:
                return None
        except Exception:
            return None
    else:
        return None

    return path if path.exists() else None


def _occupancy_stats_from_entries(entries):
    """Media/máx de la ocupación (% del frame) de una lista de entradas por-persona
    (las de gender_classifications, que llevan el campo `occupancy`). Devuelve
    `(media, max)` redondeados a 2 decimales, o `(None, None)` si no hay datos."""
    occs = [e.get('occupancy') for e in (entries or []) if e.get('occupancy') is not None]
    if not occs:
        return None, None
    return round(float(np.mean(occs)), 2), round(float(np.max(occs)), 2)


def extract_results_for_csv(result, content_type):
    """
    Extrae resultados de un análisis y los aplana en un diccionario para columnas CSV.

    Args:
        result: Diccionario de resultados del análisis (de process_image o process_video)
        content_type: Tipo de contenido ('photos', 'videos', 'albums')

    Returns:
        Diccionario con valores planos para añadir como columnas al CSV
    """
    if result is None:
        base = {
            'ia_n_personas': None,
            'ia_avg_personas_frame': None,
            'ia_genero': None,
            'ia_edad': None,
            'ia_comportamiento': None,
            'ia_actividad': None,
            'ia_exposicion_cuerpo': None,
            'ia_ubicacion': None,
            'ia_adiposity': None,
            'ia_distancia_social': None,
            'ia_belleza_media': None,
            'ia_ocupacion_media': None,
            'ia_ocupacion_max': None,
            'ia_ocr_texto': None,
            'ia_analisis_estado': 'error',
        }
        for col_name in ACCESSORY_CSV_COLUMNS.values():
            base[col_name] = None
        return base

    data = {}

    # --- Collage handling: agregar resultados de sub-imágenes ---
    if result.get('collage_detected'):
        total_persons = 0
        all_genders = []
        all_ages = []
        all_behaviours = []
        all_activities = []
        all_body_display = []
        all_locations = []
        all_body_shapes = []
        all_accessories = []
        all_beauty = []
        all_social = []

        panels = result.get('sub_images') or result.get('panels', [])
        for panel in panels:
            pr = panel.get('result', {}) or {}
            total_persons += pr.get('persons_detected', 0) or pr.get('unique_persons_tracked', 0)
            all_genders.extend(pr.get('gender_classifications', []))
            all_ages.extend(pr.get('age_classifications', []))
            all_behaviours.extend(pr.get('behaviour_classifications', []))
            all_activities.extend(pr.get('activity_classifications', []))
            all_body_display.extend(pr.get('body_display_classifications', []))
            all_locations.extend(pr.get('location_classifications', []))
            all_body_shapes.extend(pr.get('body_shape_classifications', []))
            all_accessories.extend(pr.get('accessory_classifications', []))
            all_beauty.extend(pr.get('beauty_scores', []))
            all_social.extend(pr.get('social_distance_classifications', []))

        data['ia_n_personas'] = total_persons
        data['ia_avg_personas_frame'] = None

        if all_genders:
            gc = Counter(g.get('gender', g.get('label', '')) for g in all_genders)
            data['ia_genero'] = json.dumps(dict(gc), ensure_ascii=False)
        else:
            data['ia_genero'] = None

        if all_ages:
            ac = Counter(a.get('age_group', a.get('label', '')) for a in all_ages)
            data['ia_edad'] = json.dumps(dict(ac), ensure_ascii=False)
        else:
            data['ia_edad'] = None

        if all_behaviours:
            bc = Counter(b.get('behaviour', '') for b in all_behaviours)
            data['ia_comportamiento'] = json.dumps(dict(bc), ensure_ascii=False)
        else:
            data['ia_comportamiento'] = None

        if all_activities:
            actc = Counter(a.get('activity', '') for a in all_activities)
            data['ia_actividad'] = json.dumps(dict(actc), ensure_ascii=False)
        else:
            data['ia_actividad'] = None

        if all_body_display:
            bdc = Counter(b.get('body_display', '') for b in all_body_display)
            data['ia_exposicion_cuerpo'] = json.dumps(dict(bdc), ensure_ascii=False)
        else:
            data['ia_exposicion_cuerpo'] = None

        if all_locations:
            lc = Counter(l.get('location', '') for l in all_locations)
            data['ia_ubicacion'] = json.dumps(dict(lc), ensure_ascii=False)
        else:
            data['ia_ubicacion'] = None

        if all_body_shapes:
            # silueta ELIMINADA 2026-07-04 → distribución de PESO corporal
            bsc = Counter(b.get('body_weight', '') for b in all_body_shapes)
            data['ia_adiposity'] = json.dumps(dict(bsc), ensure_ascii=False)
        else:
            data['ia_adiposity'] = None

        if all_accessories:
            for cat, col_name in ACCESSORY_CSV_COLUMNS.items():
                values = [a.get(cat) for a in all_accessories if a.get(cat) is not None]
                data[col_name] = 1 if any(v == 1 for v in values) else (0 if values else None)
        else:
            for col_name in ACCESSORY_CSV_COLUMNS.values():
                data[col_name] = None

        if all_social:
            sc = Counter(s.get('category', '') for s in all_social)
            data['ia_distancia_social'] = json.dumps(dict(sc), ensure_ascii=False)
        else:
            data['ia_distancia_social'] = None

        if all_beauty:
            scores = [b.get('score', b) for b in all_beauty if isinstance(b, (dict, int, float))]
            scores = [s for s in scores if s is not None]
            data['ia_belleza_media'] = round(np.mean(scores), 1) if scores else None
        else:
            data['ia_belleza_media'] = None

        # Ocupación: media/máx del % de frame por persona. En collage es relativa
        # al PANEL (cada panel se clasifica por separado), no a la imagen completa.
        data['ia_ocupacion_media'], data['ia_ocupacion_max'] = _occupancy_stats_from_entries(all_genders)

        data['ia_ocr_texto'] = None
        data['ia_analisis_estado'] = 'completado'
        return data

    # --- Imagen simple ---
    if 'persons_detected' in result:
        data['ia_n_personas'] = result.get('persons_detected', 0)
        data['ia_avg_personas_frame'] = None

        gc = result.get('gender_classifications', [])
        if gc:
            gender_counts = Counter(g.get('gender', g.get('label', '')) for g in gc)
            data['ia_genero'] = json.dumps(dict(gender_counts), ensure_ascii=False)
        else:
            data['ia_genero'] = None

        ac = result.get('age_classifications', [])
        if ac:
            age_counts = Counter(a.get('age_group', a.get('label', '')) for a in ac)
            data['ia_edad'] = json.dumps(dict(age_counts), ensure_ascii=False)
        else:
            data['ia_edad'] = None

        bc = result.get('behaviour_classifications', result.get('visual_interaction_classifications', []))
        if bc:
            beh_counts = Counter(b.get('behaviour', b.get('interaction_type', '')) for b in bc)
            data['ia_comportamiento'] = json.dumps(dict(beh_counts), ensure_ascii=False)
        else:
            data['ia_comportamiento'] = None

        actc = result.get('activity_classifications', [])
        if actc:
            act_counts = Counter(a.get('activity', '') for a in actc)
            data['ia_actividad'] = json.dumps(dict(act_counts), ensure_ascii=False)
        else:
            data['ia_actividad'] = None

        bdc = result.get('body_display_classifications', [])
        if bdc:
            bd_counts = Counter(b.get('body_display', '') for b in bdc)
            data['ia_exposicion_cuerpo'] = json.dumps(dict(bd_counts), ensure_ascii=False)
        else:
            data['ia_exposicion_cuerpo'] = None

        lc = result.get('location_classifications', [])
        if lc:
            loc_counts = Counter(l.get('location', '') for l in lc)
            data['ia_ubicacion'] = json.dumps(dict(loc_counts), ensure_ascii=False)
        else:
            data['ia_ubicacion'] = None

        bsc = result.get('body_shape_classifications', [])
        if bsc:
            bs_counts = Counter(b.get('body_weight', '') for b in bsc)
            data['ia_adiposity'] = json.dumps(dict(bs_counts), ensure_ascii=False)
        else:
            data['ia_adiposity'] = None

        acc = result.get('accessory_classifications', [])
        if acc:
            for cat, col_name in ACCESSORY_CSV_COLUMNS.items():
                values = [a.get(cat) for a in acc if a.get(cat) is not None]
                data[col_name] = 1 if any(v == 1 for v in values) else (0 if values else None)
        else:
            for col_name in ACCESSORY_CSV_COLUMNS.values():
                data[col_name] = None

        sdc = result.get('social_distance_classifications', [])
        if sdc:
            sd_counts = Counter(s.get('category', '') for s in sdc)
            data['ia_distancia_social'] = json.dumps(dict(sd_counts), ensure_ascii=False)
        else:
            data['ia_distancia_social'] = None

        bs = result.get('beauty_scores', [])
        if bs:
            scores = []
            for b in bs:
                if isinstance(b, dict):
                    score_val = b.get('score')
                else:
                    score_val = b
                if score_val is not None:
                    try:
                        scores.append(float(score_val))
                    except (ValueError, TypeError):
                        pass
            data['ia_belleza_media'] = round(np.mean(scores), 1) if scores else None
        else:
            data['ia_belleza_media'] = None

        # Ocupación: media/máx del % de frame que ocupa cada persona.
        data['ia_ocupacion_media'], data['ia_ocupacion_max'] = _occupancy_stats_from_entries(gc)

        data['ia_ocr_texto'] = None

    # --- Video ---
    elif 'unique_persons_tracked' in result:
        data['ia_n_personas'] = result.get('unique_persons_tracked', 0)
        data['ia_avg_personas_frame'] = result.get('avg_persons_per_frame', None)

        gs = result.get('gender_statistics', {})
        if gs and gs.get('gender_distribution'):
            data['ia_genero'] = json.dumps(gs['gender_distribution'], ensure_ascii=False)
        else:
            data['ia_genero'] = None

        ages = result.get('age_statistics', {})
        if ages and ages.get('age_group_distribution'):
            data['ia_edad'] = json.dumps(ages['age_group_distribution'], ensure_ascii=False)
        else:
            data['ia_edad'] = None

        behs = result.get('behaviour_statistics', {})
        if behs and behs.get('behaviour_distribution'):
            data['ia_comportamiento'] = json.dumps(behs['behaviour_distribution'], ensure_ascii=False)
        else:
            data['ia_comportamiento'] = None

        acts = result.get('activity_statistics', {})
        if acts and acts.get('activity_distribution'):
            data['ia_actividad'] = json.dumps(acts['activity_distribution'], ensure_ascii=False)
        else:
            data['ia_actividad'] = None

        bds = result.get('body_display_statistics', {})
        if bds and bds.get('body_display_distribution'):
            data['ia_exposicion_cuerpo'] = json.dumps(bds['body_display_distribution'], ensure_ascii=False)
        else:
            data['ia_exposicion_cuerpo'] = None

        locs = result.get('location_statistics', {})
        if locs and locs.get('location_distribution'):
            data['ia_ubicacion'] = json.dumps(locs['location_distribution'], ensure_ascii=False)
        else:
            data['ia_ubicacion'] = None

        bss = result.get('body_shape_statistics', {})
        if bss and bss.get('body_weight_distribution'):
            data['ia_adiposity'] = json.dumps(bss['body_weight_distribution'], ensure_ascii=False)
        else:
            data['ia_adiposity'] = None

        accs = result.get('accessory_statistics', {})
        acc_dist = accs.get('accessory_distribution', {}) if accs else {}
        for cat, col_name in ACCESSORY_CSV_COLUMNS.items():
            val = acc_dist.get(cat)
            data[col_name] = val if val is not None else None

        sds = result.get('social_distance_statistics', {})
        if sds and sds.get('category_distribution'):
            data['ia_distancia_social'] = json.dumps(sds['category_distribution'], ensure_ascii=False)
        else:
            data['ia_distancia_social'] = None

        bstats = result.get('beauty_statistics', {})
        if bstats and bstats.get('mean') is not None:
            data['ia_belleza_media'] = round(bstats['mean'], 1)
        else:
            data['ia_belleza_media'] = None

        occ_stats = result.get('occupancy_statistics', {}) or {}
        data['ia_ocupacion_media'] = occ_stats.get('mean')
        data['ia_ocupacion_max'] = occ_stats.get('max')

        ocr = result.get('ocr_results', {})
        if ocr:
            all_texts = []
            for scene_num in sorted(ocr.keys(), key=lambda x: int(x)):
                text = ocr[scene_num].get('text', '')
                if text:
                    all_texts.append(text.strip())
            data['ia_ocr_texto'] = ' | '.join(all_texts) if all_texts else None
        else:
            data['ia_ocr_texto'] = None

    else:
        data['ia_n_personas'] = None
        data['ia_avg_personas_frame'] = None
        data['ia_genero'] = None
        data['ia_edad'] = None
        data['ia_comportamiento'] = None
        data['ia_actividad'] = None
        data['ia_exposicion_cuerpo'] = None
        data['ia_ubicacion'] = None
        data['ia_adiposity'] = None
        data['ia_distancia_social'] = None
        data['ia_belleza_media'] = None
        data['ia_ocupacion_media'] = None
        data['ia_ocupacion_max'] = None
        data['ia_ocr_texto'] = None
        for col_name in ACCESSORY_CSV_COLUMNS.values():
            data[col_name] = None

    vlm_status = (result.get('vlm_status') if isinstance(result, dict) else None) or 'ok'
    data['ia_analisis_estado'] = {
        'ok':      'completado',
        'partial': 'parcial',
        'failed':  'vlm_failed',
    }.get(vlm_status, 'completado')
    return data


def process_single_file(input_path, models, verbose=True):
    """
    Procesa un único archivo multimedia y devuelve el resultado.

    Args:
        input_path: Path al archivo
        models: Diccionario de modelos cargados
        verbose: Si False, suprime los prints internos de los procesadores

    Returns:
        Diccionario de resultados o None
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    file_ext = input_path.suffix.lower()

    _null = io.StringIO() if not verbose else None

    with redirect_stdout(_null) if _null else _noop():
        if file_ext in image_extensions:
            result = process_image(
                input_path, models['model_detect'], models['model_pose'], OUTPUT_DIR,
                models['beauty_estimator'], models['gender_classifier'], models['age_classifier'],
                models['behaviour_classifier'], models['activity_classifier'], models['body_display_classifier'],
                models['location_classifier'], models['body_shape_classifier'], models['accessory_classifier'],
                models['social_distance_classifier'],
                person_attributes_classifier=models.get('person_attributes_classifier'),
                scene_context_classifier=models.get('scene_context_classifier'),
            )
        elif file_ext in video_extensions:
            result = process_video(
                input_path, models['model_detect'], models['model_pose'], OUTPUT_DIR,
                models['beauty_estimator'], models['gender_classifier'], models['age_classifier'],
                models['behaviour_classifier'], models['activity_classifier'], models['body_display_classifier'],
                models['location_classifier'], models['body_shape_classifier'], models['accessory_classifier'],
                models['ocr_classifier'], models['social_distance_classifier'],
                person_attributes_classifier=models.get('person_attributes_classifier'),
                scene_context_classifier=models.get('scene_context_classifier'),
            )
        else:
            if verbose:
                warn(f"  Formato no soportado: {file_ext}")
            result = None

    if result is not None:
        summary = {
            "run_timestamp": datetime.now().isoformat(),
            "input_file": str(input_path),
            "result": result
        }
        summary_path = OUTPUT_DIR / f"summary_{input_path.stem}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return result


def _load_pose_only():
    """Carga solo el YOLO pose (para el pase de belleza cuando todo el batch estaba
    en caché y no se cargaron los modelos del pipeline)."""
    try:
        from ultralytics import YOLO
        m = YOLO(config.POSE_MODEL)
        if config.YOLO_DEVICE:
            m.to(config.YOLO_DEVICE)
        return m
    except Exception as e:
        _console.print(f"  [yellow]no se pudo cargar YOLO pose para belleza: {e}")
        return None


def process_batch(csv_path, sample_size=400, multimedia_dir=None, output_csv=None, seed=42):
    """
    Procesa un batch de archivos multimedia desde un CSV y añade columnas con los resultados.

    Args:
        csv_path: Ruta al CSV de entrada (masculinity-IG-limpio.csv)
        sample_size: Número de archivos a muestrear
        multimedia_dir: Directorio con los archivos descargados
        output_csv: Ruta al CSV de salida (por defecto añade _analizado al nombre)
        seed: Semilla para reproducibilidad del muestreo
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        error(f"CSV no encontrado: {csv_path}")
        return

    if multimedia_dir is None:
        multimedia_dir = csv_path.parent / "multimedia_downloads"
    multimedia_dir = Path(multimedia_dir)

    if output_csv is None:
        output_csv = csv_path.parent / f"{csv_path.stem}_analizado.csv"
    output_csv = Path(output_csv)

    header("Batch — análisis masivo")
    kv_table("", [
        ("CSV entrada", csv_path.name),
        ("Multimedia", str(multimedia_dir)),
        ("Muestra", sample_size),
        ("Seed", seed),
    ])

    df = pd.read_csv(csv_path)
    df_downloaded = df[df['multimedia_descargado'] == True].copy()
    df_downloaded['_file_path'] = df_downloaded.apply(
        lambda row: resolve_file_path(row, multimedia_dir), axis=1
    )
    df_valid = df_downloaded[df_downloaded['_file_path'].notna()].copy()

    if len(df_valid) == 0:
        error("No se encontraron archivos válidos para procesar")
        return

    actual_sample_size = min(sample_size, len(df_valid))
    df_sample = df_valid.sample(n=actual_sample_size, random_state=seed)

    info(
        f"  Filas: [cyan]{len(df)}[/]  "
        f"descargadas=[cyan]{len(df_downloaded)}[/]  "
        f"válidas=[cyan]{len(df_valid)}[/]  "
        f"muestra=[cyan]{actual_sample_size}"
    )

    already_processed = 0
    to_process_indices = []
    pre_loaded_results = {}

    for idx, row in df_sample.iterrows():
        file_path = row['_file_path']
        summary_path = OUTPUT_DIR / f"summary_{file_path.stem}.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                if summary_data.get('result') is not None:
                    pre_loaded_results[idx] = summary_data['result']
                    already_processed += 1
                    continue
            except Exception:
                pass
        to_process_indices.append(idx)

    info(f"  Caché: [cyan]{already_processed}[/]  pendientes=[cyan]{len(to_process_indices)}")

    if len(to_process_indices) > 0:
        models = load_all_models()
        if models is None:
            error("Error al cargar modelos")
            return
    else:
        models = None
        info("  [dim]Todo en caché, no se cargan modelos")

    header("Procesando archivos")

    new_columns = [
        'ia_n_personas', 'ia_avg_personas_frame', 'ia_genero', 'ia_edad',
        'ia_comportamiento', 'ia_actividad', 'ia_exposicion_cuerpo',
        'ia_ubicacion', 'ia_adiposity',
        *list(ACCESSORY_CSV_COLUMNS.values()),
        'ia_distancia_social', 'ia_belleza_media',
        'ia_ocupacion_media', 'ia_ocupacion_max', 'ia_ocr_texto',
        'ia_analisis_estado'
    ]

    results_data = {}

    for idx, result in pre_loaded_results.items():
        row = df_sample.loc[idx]
        csv_data = extract_results_for_csv(result, row['content_type'])
        results_data[idx] = csv_data

    total_to_process = len(to_process_indices)
    successful = 0
    failed = 0
    failed_files = []
    batch_start = time.time()

    # Con backend Transformers, paralelizar archivos solo genera contención
    # CUDA (la GPU ya está saturada por el batching real dentro de cada
    # imagen). Forzamos a 1 worker en ese caso.
    if getattr(config, "VLM_BACKEND", "ollama") == "transformers":
        file_workers = 1
    else:
        file_workers = max(1, getattr(config, "OLLAMA_FILE_PARALLEL", 1))
    state_lock = threading.Lock()
    completed_count = [0]  # mutable holder accesible desde el worker

    def _worker(idx):
        row = df_sample.loc[idx]
        file_path = row['_file_path']
        try:
            result = process_single_file(file_path, models, verbose=False)
            if result is not None:
                csv_data = extract_results_for_csv(result, row['content_type'])
                return idx, file_path, csv_data, "ok", None
            csv_data = extract_results_for_csv(None, row['content_type'])
            return idx, file_path, csv_data, "empty", None
        except Exception as e:
            csv_data = extract_results_for_csv(None, row['content_type'])
            csv_data['ia_analisis_estado'] = f'error: {str(e)[:100]}'
            return idx, file_path, csv_data, "error", e

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[filename]:<42}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]✓{task.fields[ok]}[/] [red]✗{task.fields[err]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]:.1f} arch/min"),
        console=_console,
    ) as progress:
        task = progress.add_task(
            "", total=total_to_process, filename="", ok=0, err=0, speed=0.0
        )

        with ThreadPoolExecutor(max_workers=file_workers) as pool:
            futures = [pool.submit(_worker, idx) for idx in to_process_indices]
            for fut in as_completed(futures):
                idx, file_path, csv_data, status, exc = fut.result()

                with state_lock:
                    results_data[idx] = csv_data
                    if status == "ok":
                        successful += 1
                        n = csv_data.get('ia_n_personas', 0)
                        _console.log(f"[green]·[/] {file_path.name}  [dim]{n}p")
                    elif status == "empty":
                        failed += 1
                        _console.log(f"[yellow]·[/] {file_path.name}  [dim]sin resultado")
                    else:
                        failed += 1
                        failed_files.append((file_path.name, str(exc)[:120]))
                        _console.log(f"[red]·[/] {file_path.name}  [dim]{str(exc)[:80]}")

                    completed_count[0] += 1
                    i = completed_count[0]
                    elapsed = time.time() - batch_start
                    speed = i / (elapsed / 60) if elapsed > 0 else 0.0
                    progress.update(
                        task, advance=1, ok=successful, err=failed,
                        speed=speed, filename=file_path.name[:42],
                    )

                    if i % 25 == 0:
                        _console.log(f"[dim]checkpoint ({i}/{total_to_process})")
                        _save_batch_csv(df, df_sample, results_data, new_columns, output_csv)

    header("Guardando resultados")
    _save_batch_csv(df, df_sample, results_data, new_columns, output_csv)

    # ========================================================================
    # PASE DE BELLEZA DIFERIDO (Qwen3-VL @ :11435, solo demand/*)
    # ========================================================================
    # Tras clasificar con gemma4, puntúa los rostros demand/* leyendo los
    # summary_*.json de OUTPUT_DIR. Sidecar <output_csv>_belleza.xlsx. Degrada con
    # elegancia (off / sin demand / :11435 caído → no escribe nada). Reutiliza el
    # YOLO pose ya cargado; si todo estaba en caché, lo carga al vuelo.
    if getattr(config, "ENABLE_BEAUTY_PASS", True):
        try:
            from processing.beauty_phase import run_beauty_phase, shared_beauty_backend_from_models
            pose = models['model_pose'] if models else _load_pose_only()
            if pose is not None:
                df_beauty, beauty_col = run_beauty_phase(
                    OUTPUT_DIR, pose,
                    shared_backend=shared_beauty_backend_from_models(models))
                if df_beauty is not None and not df_beauty.empty:
                    sidecar = output_csv.with_name(output_csv.stem + "_belleza.xlsx")
                    df_beauty.to_excel(sidecar, index=False, engine="openpyxl")
                    _console.print(f"  [dim]belleza → {sidecar}")

                    # Re-dibujar cada _annotated.jpg con el chip de belleza: el pase
                    # diferido puntúa DESPUÉS de guardar las imágenes anotadas, así que
                    # sin esto el chip no aparece. Reusa el summary + caches, sin VLM.
                    # Solo si el detector está cargado (models=None ⇒ todo en caché ⇒
                    # las imágenes ya vienen de un run previo).
                    if (models and models.get('model_detect') is not None
                            and beauty_col in df_beauty.columns
                            and 'track_id' in df_beauty.columns):
                        from processing.image import rerender_annotated_with_beauty
                        redrawn = 0
                        for img_id, grp in df_beauty.groupby('Img ID'):
                            beauty_by_track = {
                                r['track_id']: r[beauty_col]
                                for _, r in grp.iterrows()
                                if r.get('track_id') is not None
                            }
                            if not beauty_by_track:
                                continue
                            summ_path = OUTPUT_DIR / f"summary_{img_id}.json"
                            if not summ_path.exists():
                                continue
                            try:
                                with open(summ_path, 'r', encoding='utf-8') as f:
                                    res = json.load(f).get('result')
                                if not res or not res.get('input_path'):
                                    continue
                                if rerender_annotated_with_beauty(
                                        Path(res['input_path']), OUTPUT_DIR, res,
                                        beauty_by_track, models['model_detect'],
                                        models['model_pose'],
                                        social_distance_classifier=models.get('social_distance_classifier')):
                                    redrawn += 1
                            except Exception:
                                continue
                        if redrawn:
                            _console.print(f"  [dim]↻ {redrawn} imágenes re-dibujadas con chip de belleza")
        except Exception as e:
            _console.print(f"  [yellow]pase de belleza omitido por error: {e}")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    total_elapsed = time.time() - batch_start
    total_in_results = len(results_data)
    completed = sum(1 for r in results_data.values() if r.get('ia_analisis_estado') == 'completado')

    summary_table = Table(title="Resumen del batch", title_style="bold", show_header=False, box=None, pad_edge=False)
    summary_table.add_column(style="dim")
    summary_table.add_column(justify="right")
    summary_table.add_row("Muestra", str(actual_sample_size))
    summary_table.add_row("Caché reutilizada", str(already_processed))
    summary_table.add_row("Procesados", str(total_to_process))
    summary_table.add_row("  exitosos", f"[green]{successful}")
    summary_table.add_row("  fallidos", f"[red]{failed}" if failed else "0")
    summary_table.add_row("Completados (total)", str(completed))
    summary_table.add_row("Filas CSV", str(len(df)))
    summary_table.add_row("Filas con análisis", str(total_in_results))
    if total_to_process > 0:
        summary_table.add_row("Tiempo", f"{total_elapsed/60:.1f} min")
        summary_table.add_row("Velocidad", f"{total_to_process/(total_elapsed/60):.1f} arch/min")
    _console.print(summary_table)

    if failed_files:
        err_table = Table(title="Archivos con errores", title_style="bold red", show_header=False, box=None)
        err_table.add_column(style="red")
        err_table.add_column(style="dim")
        for fname, err in failed_files:
            err_table.add_row(fname, err)
        _console.print(err_table)

    _console.print(f"\n[green]✓[/] CSV guardado en: [dim]{output_csv}\n")


def _save_batch_csv(df_full, df_sample, results_data, new_columns, output_path):
    """
    Guarda el CSV con las columnas de análisis añadidas.
    Solo las filas de la muestra tendrán datos de análisis.
    """
    import pandas as pd

    df_out = df_full.copy()

    for col in new_columns:
        if col not in df_out.columns:
            df_out[col] = None

    for idx, data in results_data.items():
        for col, val in data.items():
            df_out.at[idx, col] = val

    df_out.to_csv(output_path, index=False)
