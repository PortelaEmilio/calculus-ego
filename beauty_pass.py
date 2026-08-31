"""
Pase DIFERIDO de belleza facial, fase 2 del flujo integrado.

Se ejecuta APARTE del pipeline principal de 15 clasificadores: en 12 GB no caben a la vez
el backend Ollama de gemma4 y el modelo de belleza. Flujo:
  1) Pipeline normal (`main.py batch`, gemma4) → 15 categorías + summary_*.json por imagen
     (cada uno con behaviour_classifications que ahora incluyen el bbox de la persona).
  2) Este pase (gemma4 parado → Ollama hace UN swap al modelo de belleza) → puntúa la
     belleza SOLO de los rostros cuyo comportamiento es demand/* (con --solo-demand).

Backend de belleza: Qwen3.5-9B continuo (1-10 decimal) por Transformers/PEFT
(`config.BEAUTY_BACKEND="qwen35_cont"`).

Reutiliza la MISMA detección/gate/crop que processing/image.py:
  - YOLO pose (config.POSE_MODEL) para keypoints + bboxes
  - filtro de bbox minúsculo (config.BBOX_MIN_FRAME_RATIO)
  - gate de rostro frontal (has_five_face_keypoints_visible)
  - extract_face_crop para el recorte facial
Con --solo-demand, un rostro solo se puntúa si su bbox detectado hace IoU>=0.5 con un bbox
demand/* de la fase 1 (matching geométrico, inmune a drift de orden de detección).

Entrada: --phase1-json (dir con summary_*.json o un .json) deriva la lista de imágenes y los
bboxes demand. Sin --phase1-json, --input acepta (a) .xlsx ('Img ID'/'Ruta Img.'), (b) .csv
(--img-col), o (c) dir/glob, y puntúa TODOS los rostros frontales (modo legacy).
Salida: --output .xlsx/.csv con una fila por rostro puntuado + media por imagen.

Uso (desde Analisis_identidad/, venv principal):
    ollama stop gemma4:e4b
    python beauty_pass.py --phase1-json <output_dir_fase1> --solo-demand \
        --output ../validacion_imagenes/belleza_qwen_demand.xlsx
"""
import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from ui import header, info, success, warn
from utils.visualization import (has_five_face_keypoints_visible, extract_face_crop,
                                 count_face_keypoints_visible)
from processing.image import _iou_xyxy

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEMAND_IOU_THRESHOLD = 0.5


def load_phase1_demand(phase1_path: str, demand_only: bool = True) -> dict:
    """Lee summary_*.json de la fase 1 y devuelve, por imagen, los bboxes de sus personas.

    Returns dict keyed por (a) ruta absoluta del input_file y (b) stem, cada uno →
    {"path": Path, "bboxes": [[x1,y1,x2,y2], ...], "demand": [{"bbox":[...],
    "track_id": str}, ...]}. `demand` conserva el track_id de cada bbox para poder enlazar
    la puntuación de belleza con la fila por-persona del run (IoU del rostro detectado →
    entry → track_id).

    2026-07-21: `demand_only` (default True para no romper el CLI standalone `--solo-demand`).
    Con demand_only=False la lista `demand` incluye TODAS las personas (behaviour cualquiera),
    de modo que el pase de belleza puntúa toda cara y recupera igualmente el track_id por IoU.
    """
    p = Path(phase1_path)
    files = sorted(p.glob("summary_*.json")) if p.is_dir() else [p]
    by_key: dict = {}
    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            warn(f"  no se pudo leer {jf.name}: {e}")
            continue
        result = data.get("result") or {}
        img_path = data.get("input_file") or result.get("input_path")
        if not img_path:
            continue
        demand = [
            {"bbox": bc["bbox"], "track_id": bc.get("track_id")}
            for bc in result.get("behaviour_classifications", [])
            if bc.get("bbox") and (not demand_only
                                   or str(bc.get("behaviour", "")).startswith("demand"))
        ]
        entry = {"path": Path(img_path),
                 "bboxes": [d["bbox"] for d in demand],
                 "demand": demand}
        by_key[str(Path(img_path).resolve())] = entry
        by_key.setdefault(Path(img_path).stem, entry)
    return by_key


def resolve_path(token: str, multimedia_dir: str | None) -> Path | None:
    p = Path(str(token)).expanduser()
    if p.exists():
        return p
    if multimedia_dir:
        cand = Path(multimedia_dir) / str(token)
        if cand.exists():
            return cand
    return None


def collect_inputs(args) -> list[tuple[str, Path]]:
    """Devuelve lista de (img_id, path)."""
    inp = Path(args.input)
    out = []
    if inp.is_dir():
        for f in sorted(inp.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                out.append((f.stem, f))
    elif inp.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(inp)
        id_col = "Img ID" if "Img ID" in df.columns else df.columns[0]
        path_col = "Ruta Img." if "Ruta Img." in df.columns else (args.img_col or df.columns[-1])
        for _, row in df.iterrows():
            p = resolve_path(row[path_col], args.multimedia_dir)
            if p:
                out.append((str(row[id_col]), p))
            else:
                warn(f"  no se encontró imagen: {row[path_col]}")
    elif inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
        col = args.img_col or next((c for c in ("Ruta Img.", "path", "image", "img_path") if c in df.columns), df.columns[-1])
        id_col = "Img ID" if "Img ID" in df.columns else None
        for i, row in df.iterrows():
            p = resolve_path(row[col], args.multimedia_dir)
            if p:
                out.append((str(row[id_col]) if id_col else p.stem, p))
    else:  # glob
        for f in sorted(_glob.glob(args.input)):
            fp = Path(f)
            if fp.suffix.lower() in IMG_EXTS:
                out.append((fp.stem, fp))
    # Dedup por img_id (los xlsx manuales traen una fila por persona → Img ID repetido).
    seen, dedup = set(), []
    for img_id, p in out:
        if img_id in seen:
            continue
        seen.add(img_id)
        dedup.append((img_id, p))
    out = dedup
    if args.limit:
        out = out[: args.limit]
    return out


def build_items_from_phase1(demand_map: dict, solo_demand: bool) -> list:
    """Deriva la lista ordenada de (img_id, path) de los summaries de la fase 1.
    Con solo_demand, solo incluye imágenes que tengan al menos un bbox demand/*."""
    items, seen = [], set()
    for entry in demand_map.values():
        path = entry["path"]
        rp = str(path.resolve())
        if rp in seen:
            continue
        if solo_demand and not entry["bboxes"]:
            continue
        seen.add(rp)
        items.append((path.stem, path))
    return items


def score_demand_beauty(items, demand_map, model_pose, estimator, *, solo_demand=True,
                        min_head_px=None):
    """Puntúa belleza de los rostros de `items` y devuelve (df_rows, df_img).

    Con `solo_demand`, un rostro solo se puntúa si su bbox detectado hace IoU>=0.5 con
    un bbox demand/* de la fase 1 (`demand_map`); además `df_rows` lleva el `track_id`
    de la persona demand con la que casa (mayor IoU) → permite enlazar la puntuación con
    la fila por-persona del run. Lógica idéntica a la del pase standalone (DRY).

    `min_head_px`: override del gate de tamaño de cabeza (`min(alto,ancho)` del recorte);
    si es None usa `config.BEAUTY_MIN_HEAD_PX`. Para dominios de cara pequeña que llena el
    frame (p.ej. avatares 168px) se pasa 0 para desactivar el gate. `df_rows` incluye la
    columna diagnóstica `head_px` para post-filtrar sin re-correr.
    """
    score_hi = int(estimator.scale[1])
    score_col = f"IA Belleza (1-{score_hi})"
    bbox_min = getattr(config, "BBOX_MIN_FRAME_RATIO", 0.01)
    head_px_floor = config.BEAUTY_MIN_HEAD_PX if min_head_px is None else min_head_px
    rows, per_image = [], []

    for k, (img_id, path) in enumerate(items):
        # entrada demand/* de la fase 1 para esta imagen (por ruta resuelta o stem).
        d_entry = demand_map.get(str(Path(path).resolve())) or demand_map.get(Path(path).stem)
        demand_list = d_entry["demand"] if d_entry else []
        if solo_demand and not demand_list:
            continue

        frame = cv2.imread(str(path))
        if frame is None:
            warn(f"  no se pudo leer {path}"); continue
        H, W = frame.shape[:2]
        frame_area = float(H * W)
        try:
            results = model_pose(frame, verbose=False)
        except Exception as e:
            warn(f"  pose falló en {path}: {e}"); continue

        scores_img = []
        person_idx = 0
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            kpts_all = r.keypoints.data.cpu().numpy()       # (N,17,3)
            boxes = r.boxes.xyxy.cpu().numpy()              # (N,4)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                if (x2 - x1) * (y2 - y1) / frame_area < bbox_min:
                    continue
                # Recuperar el track_id de la persona de la fase 1 con mayor IoU (para
                # enlazar la belleza con la fila por-persona en el merge). Con solo_demand
                # se descarta la cara si no casa con una persona demand/*; sin él (2026-07-21,
                # belleza para todas) se asigna el track_id de la mejor coincidencia y NO se
                # descarta por comportamiento.
                best_iou, best_tid = 0.0, None
                for d in demand_list:
                    iou = _iou_xyxy((x1, y1, x2, y2), tuple(d["bbox"]))
                    if iou > best_iou:
                        best_iou, best_tid = iou, d.get("track_id")
                if solo_demand and best_iou < DEMAND_IOU_THRESHOLD:
                    continue
                track_id = best_tid if best_iou >= DEMAND_IOU_THRESHOLD else None
                kp = kpts_all[i]
                if count_face_keypoints_visible(kp) < config.BEAUTY_MIN_FACE_KEYPOINTS:
                    continue
                face = extract_face_crop(frame, kp, (x1, y1, x2, y2),
                                         min_points=config.BEAUTY_MIN_FACE_KEYPOINTS)
                if face is None or face.size == 0:
                    continue
                # Gate de tamaño mínimo de cabeza (anclado a los datasets de belleza):
                # por debajo se descarta (fuera de distribución). Ver config.BEAUTY_MIN_HEAD_PX.
                head_px = int(min(face.shape[0], face.shape[1]))
                if head_px < head_px_floor:
                    continue
                res = estimator.estimate(face)
                score = res.get("score")
                rows.append({
                    "Img ID": img_id, "Ruta Img.": str(path), "person_idx": person_idx,
                    "track_id": track_id,
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "head_px": head_px,
                    score_col: score, "raw": res.get("raw_response"),
                })
                if isinstance(score, (int, float)):
                    scores_img.append(float(score))
                person_idx += 1

        mean_b = round(sum(scores_img) / len(scores_img), 2) if scores_img else "NA"
        per_image.append({"Img ID": img_id, "Ruta Img.": str(path),
                          "n_faces": person_idx, f"{score_col} (media)": mean_b})
        if (k + 1) % 25 == 0:
            info(f"  {k+1}/{len(items)}")

    return pd.DataFrame(rows), pd.DataFrame(per_image)


def _norm_img_id(v) -> str:
    """Normaliza Img ID a string para el join (tolera float64 de Excel: 12345.0 → '12345').
    Los stems textuales (p.ej. '2025-01_user') pasan tal cual."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def merge_beauty_into_master(master_path: str, df_rows: pd.DataFrame, score_col: str,
                             sheet: str = "Sheet1", backup: bool = True) -> None:
    """Rellena `score_col` en el xlsx maestro por (Img ID, Pers. Key)==(Img ID, track_id).

    Conserva todas las columnas/hojas del maestro; crea la columna de belleza si faltara.
    Respalda a `<maestro>.bak` antes de sobrescribir. Las caras sin fila coincidente
    (track_id no recuperado / persona extra) no se fusionan; las filas del maestro sin cara
    puntuada quedan como están (NaN).
    """
    import shutil
    mp = Path(master_path)
    if not mp.exists():
        warn(f"  --merge-into: no existe el maestro {mp}"); return

    # Mapa (img_id_norm, track_id) → score, solo de rostros con score numérico y track_id.
    score_by_key: dict = {}
    for r in df_rows.to_dict("records"):
        tid = r.get("track_id")
        sc = r.get(score_col)
        if tid is None or not isinstance(sc, (int, float)):
            continue
        score_by_key[(_norm_img_id(r.get("Img ID")), str(tid))] = sc

    if not score_by_key:
        warn("  --merge-into: no hay rostros puntuados con track_id → nada que fusionar."); return

    xls = pd.ExcelFile(mp, engine="openpyxl")
    if sheet not in xls.sheet_names:
        warn(f"  --merge-into: la hoja '{sheet}' no está en {mp.name} ({xls.sheet_names})."); return
    sheets = {sn: xls.parse(sn) for sn in xls.sheet_names}
    df = sheets[sheet]
    for req in ("Img ID", "Pers. Key"):
        if req not in df.columns:
            warn(f"  --merge-into: falta la columna '{req}' en la hoja '{sheet}'."); return
    if score_col not in df.columns:
        df[score_col] = np.nan

    def _lookup(row):
        key = (_norm_img_id(row["Img ID"]), str(row["Pers. Key"]))
        return score_by_key.get(key, row[score_col])

    n_before = int(df[score_col].notna().sum())
    df[score_col] = df.apply(_lookup, axis=1)
    n_after = int(df[score_col].notna().sum())
    sheets[sheet] = df

    if backup:
        bak = mp.with_suffix(mp.suffix + ".bak")
        shutil.copy2(mp, bak)
        info(f"  backup → {bak.name}")
    with pd.ExcelWriter(mp, engine="openpyxl") as xw:
        for sn, sdf in sheets.items():
            sdf.to_excel(xw, sheet_name=sn, index=False)
    success(f"  maestro fusionado: {score_col} pobladas {n_before} → {n_after} "
            f"(+{n_after - n_before}) en {mp.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help="xlsx (Img ID/Ruta Img.), csv (--img-col), dir o glob")
    ap.add_argument("--output", required=True, help="salida .xlsx o .csv")
    ap.add_argument("--img-col", default=None, help="columna de ruta en csv/xlsx")
    ap.add_argument("--multimedia-dir", default=None, help="raíz para resolver rutas relativas")
    ap.add_argument("--phase1-json", default=None,
                    help="dir con summary_*.json (o un .json) de la fase 1 → bboxes demand/*")
    ap.add_argument("--solo-demand", action="store_true",
                    help="puntuar solo rostros cuyo bbox hace IoU>=0.5 con un bbox demand/* (requiere --phase1-json)")
    ap.add_argument("--min-head-px", type=int, default=None,
                    help="override del gate de tamaño de cabeza (min(alto,ancho) del recorte); "
                         "default = config.BEAUTY_MIN_HEAD_PX. Usar 0 para desactivarlo (p.ej. avatares 168px).")
    ap.add_argument("--merge-into", default=None,
                    help="xlsx maestro (carpeta/manual) a rellenar por (Img ID, Pers. Key)==(Img ID, track_id)")
    ap.add_argument("--merge-sheet", default="Sheet1", help="hoja del maestro a fusionar (default Sheet1)")
    ap.add_argument("--no-backup", action="store_true",
                    help="no escribir <maestro>.bak antes de fusionar (por defecto SÍ respalda)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.solo_demand and not args.phase1_json:
        warn("--solo-demand requiere --phase1-json (de dónde salen los bboxes demand/*)."); sys.exit(1)

    # Sin --solo-demand se puntúa TODA cara → el mapa debe llevar el track_id de CADA persona
    # (demand_only=False) para poder enlazar la belleza con su fila en el merge.
    demand_map = (load_phase1_demand(args.phase1_json, demand_only=args.solo_demand)
                  if args.phase1_json else {})

    if args.input:
        items = collect_inputs(args)
    elif demand_map:
        items = build_items_from_phase1(demand_map, args.solo_demand)
    else:
        warn("Indica --input o --phase1-json."); sys.exit(1)
    if args.limit:
        items = items[: args.limit]
    if not items:
        warn("No hay imágenes que procesar."); sys.exit(1)

    mode = "solo demand/*" if args.solo_demand else "todos los rostros frontales"
    header(f"Beauty pass ({config.BEAUTY_BACKEND}, {mode}) — {len(items)} imágenes")

    # Modelos: YOLO pose (detección/keypoints) + beauty backend según config.BEAUTY_BACKEND.
    from ultralytics import YOLO
    info(f"  YOLO pose: [dim]{config.POSE_MODEL} ({config.YOLO_DEVICE})")
    model_pose = YOLO(config.POSE_MODEL)
    if config.YOLO_DEVICE:
        model_pose.to(config.YOLO_DEVICE)
    from models.beauty import BeautyEstimator
    from models.beauty_backend_qwen35 import Qwen35BeautyBackend
    backend = Qwen35BeautyBackend()
    info("  Beauty backend: [dim]Transformers Qwen3.5-9B continuo (1-10 decimal)")
    if not backend.is_loaded():
        warn("  El backend de belleza no está disponible (¿adapter en models/beauty_adapter/?)."); sys.exit(1)
    estimator = BeautyEstimator(backend=backend)
    score_hi = int(estimator.scale[1])
    score_col = f"IA Belleza (1-{score_hi})"

    df_rows, df_img = score_demand_beauty(
        items, demand_map, model_pose, estimator, solo_demand=args.solo_demand,
        min_head_px=args.min_head_px)
    rows = df_rows.to_dict("records")
    per_image = df_img.to_dict("records")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df_rows.to_csv(out, index=False)
        df_img.to_csv(out.with_name(out.stem + "_por_imagen.csv"), index=False)
    else:
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            df_rows.to_excel(xw, sheet_name="por_rostro", index=False)
            df_img.to_excel(xw, sheet_name="por_imagen", index=False)

    n_scored = sum(1 for r in rows if isinstance(r.get(score_col), (int, float)))
    success(f"  {len(rows)} rostros ({n_scored} puntuados) en {len(per_image)} imágenes → {out}")

    if args.merge_into:
        merge_beauty_into_master(args.merge_into, df_rows, score_col,
                                 sheet=args.merge_sheet, backup=not args.no_backup)


if __name__ == "__main__":
    main()
