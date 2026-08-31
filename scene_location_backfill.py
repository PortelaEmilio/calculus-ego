#!/usr/bin/env python3
"""
Backfill de la UBICACIÓN de escena en runs YA procesados (2026-08-31).

Hasta el 2026-08-31 la ubicación solo se calculaba por persona detectada, así que las
imágenes SIN personas (capturas de tuit, tarjetas de texto, memes, paisajes) salían con
`IA Ubic.` vacía. Los runs ya hechos no la recuperan al reanudar: `main.py carpeta` salta
todo fichero que ya tenga su `summary_<stem>.json`.

Este script recorre un run existente, se queda con las imágenes de 0 personas que aún no
tienen entrada de ubicación de escena y lanza UNA llamada VLM por imagen con el prompt sin
TARGET (`SceneContextClassifier.classify_location_only`). NO carga YOLO y NO re-detecta:
la única entrada es `input_path` del summary.

  # sidecar para revisar antes de tocar nada
  python scene_location_backfill.py --run-dir runs/carpeta_20260709_164902 \
      --output ubicacion_backfill.xlsx --limit 50

  # fusionar en el xlsx maestro (respalda a <maestro>.bak)
  python scene_location_backfill.py --run-dir runs/carpeta_20260709_164902 \
      --output ubicacion_backfill.xlsx --merge-into multimedia_downloads_analizado.xlsx

VÍDEOS: no son backfilleables. Las escenas sin personas ni siquiera llegaban al
`phase1_<stem>.json` (Fase A las descartaba antes de registrarlas) y los frames de escena
son efímeros. Se listan en `<run-dir>/ubicacion_no_backfilleable.txt` y hay que re-procesar
esos vídeos con un `--run-prefix` nuevo si se quiere su ubicación.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

import config
from ui import header, info, success, warn, error, make_progress

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
UBIC_COL = "IA Ubic."


def build_scene_classifier():
    """Backend VLM + SceneContextClassifier, SIN YOLO ni el resto de clasificadores.

    Espeja el tramo de `models.loader.load_all_models()` que arma el backend (incluido el
    proxy `JsonFlatteningBackend` para los módulos de prompt JSON), pero sin cargar nada
    más: aquí solo se necesita una llamada de ubicación por imagen.
    """
    import os
    from models.backends import create_backend
    from models.scene_context import SceneContextClassifier

    model_name = os.environ.get("VLM_MODEL_OVERRIDE") or config.BEHAVIOUR_MODEL_NAME
    info(f"  Backend VLM: [dim]{config.VLM_BACKEND} · {model_name}")
    backend = create_backend(
        config.VLM_BACKEND,
        model_name=model_name,
        use_finetuned=config.USE_FINETUNED_MODEL,
        lora_path=config.FINETUNED_LORA_PATH,
        ollama_model=config.OLLAMA_MODEL,
        ollama_host=config.OLLAMA_HOST,
        ollama_hosts=config.OLLAMA_HOSTS,
    )
    if getattr(config, "VLM_PROMPT_MODULE", "") in (
            "prompts_gemma4_json", "prompts_qwen3", "prompts_qwen3_short"):
        from models.backends.json_flatten_backend import JsonFlatteningBackend
        backend = JsonFlatteningBackend(backend)
        info(f"  Proxy JSON→líneas activo ({config.VLM_PROMPT_MODULE})")
    # `create_backend` ya carga transformers/qwen35; esto solo cubre un backend que
    # llegue sin cargar (p.ej. Ollama con el servidor caído al construirlo).
    if not backend.is_loaded() and hasattr(backend, "load"):
        backend.load()
    return SceneContextClassifier(backend=backend)


def unwrap(summary: dict) -> dict:
    """Devuelve el bloque de resultado del summary.

    `manual`/`carpeta` escriben `{run_id, img_id, input_path, result: {...}}` mientras que
    `single` vuelca el `result_info` plano. Se devuelve el dict que contiene realmente
    `persons_detected` / `location_classifications` (el mismo objeto, no una copia: al
    mutarlo se muta el summary y basta con volver a serializar el envoltorio).
    """
    inner = summary.get("result")
    return inner if isinstance(inner, dict) else summary


def get_scene_location(payload: dict) -> str | None:
    """Ubicación de ESCENA ya presente en el resultado (entrada con track_id None), o None."""
    for e in (payload.get("location_classifications") or []):
        if e.get("track_id") is None:
            return e.get("location")
    return None


def scan_run(run_dir: Path):
    """Clasifica los summaries del run en (candidatos, vídeos no backfilleables, ya hechos)."""
    json_dir = run_dir / "json"
    if not json_dir.is_dir():
        error(f"  No existe {json_dir}")
        return None
    candidates, videos, already = [], [], []
    for jf in sorted(json_dir.glob("summary_*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, ValueError):
            warn(f"  summary ilegible, se salta: {jf.name}")
            continue
        payload = unwrap(summary)
        if payload.get("scenes") is not None:      # vídeo (validation dump)
            if not payload["scenes"]:
                videos.append(jf)
            continue
        if payload.get("persons_detected") is None:
            warn(f"  summary sin `persons_detected`, se salta: {jf.name}")
            continue
        if payload.get("persons_detected", 0) != 0:
            continue
        src = payload.get("input_path") or summary.get("input_path")
        prev = get_scene_location(payload)
        if prev is not None:
            # Ya puntuada en una pasada anterior. NO se vuelve a llamar al VLM, pero SÍ
            # entra en el sidecar/merge: sin esto, reanudar un run interrumpido fusionaría
            # solo lo puntuado en la última pasada y dejaría el resto fuera del xlsx.
            if src:
                already.append((Path(src), prev))
            continue
        if not src or not Path(src).exists():
            warn(f"  imagen original no encontrada ({jf.name}) → se salta")
            continue
        candidates.append((jf, summary, payload, Path(src)))
    return candidates, videos, already


def merge_into_master(master_path: str, df_rows: pd.DataFrame, sheet: str, backup: bool):
    """Rellena `IA Ubic.` en las filas SIN persona del xlsx maestro.

    El join va por `Archivo` (nombre de fichero), NO por `Img ID`: Excel guarda los ids de
    Instagram (>2^53) como float y pierde precisión al round-trip. Solo se tocan filas con
    `Pers. Key` vacía (las de 0 personas) y con `IA Ubic.` vacía — nunca se pisa un valor.
    """
    mp = Path(master_path)
    if not mp.exists():
        warn(f"  --merge-into: no existe el maestro {mp}"); return

    loc_by_file = {r["Archivo"]: r[UBIC_COL] for r in df_rows.to_dict("records")
                   if r.get("Archivo") and r.get(UBIC_COL)}
    if not loc_by_file:
        warn("  --merge-into: no hay ubicaciones nuevas → nada que fusionar."); return

    xls = pd.ExcelFile(mp, engine="openpyxl")
    if sheet not in xls.sheet_names:
        warn(f"  --merge-into: la hoja '{sheet}' no está en {mp.name} ({xls.sheet_names})."); return
    sheets = {sn: xls.parse(sn) for sn in xls.sheet_names}
    df = sheets[sheet]
    for req in ("Archivo", "Pers. Key"):
        if req not in df.columns:
            warn(f"  --merge-into: falta la columna '{req}' en la hoja '{sheet}'."); return
    if UBIC_COL not in df.columns:
        df[UBIC_COL] = None

    sin_persona = df["Pers. Key"].isna()
    sin_ubic = df[UBIC_COL].isna()
    target = sin_persona & sin_ubic
    n_before = int(df[UBIC_COL].notna().sum())
    df.loc[target, UBIC_COL] = df.loc[target, "Archivo"].map(loc_by_file)
    n_after = int(df[UBIC_COL].notna().sum())
    sheets[sheet] = df

    if backup:
        bak = mp.with_suffix(mp.suffix + ".bak")
        shutil.copy2(mp, bak)
        info(f"  backup → {bak.name}")
    with pd.ExcelWriter(mp, engine="openpyxl") as xw:
        for sn, sdf in sheets.items():
            sdf.to_excel(xw, sheet_name=sn, index=False)
    success(f"  maestro fusionado: {UBIC_COL} poblada {n_before} → {n_after} "
            f"(+{n_after - n_before}) en {mp.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="runs/<run_id>/ con la carpeta json/")
    ap.add_argument("--output", default=None, help="xlsx sidecar con las ubicaciones nuevas")
    ap.add_argument("--merge-into", default=None, help="xlsx maestro donde fusionar IA Ubic.")
    ap.add_argument("--merge-sheet", default="Sheet1", help="hoja del maestro (default Sheet1)")
    ap.add_argument("--no-backup", action="store_true", help="no respaldar el maestro a .bak")
    ap.add_argument("--limit", type=int, default=0, help="procesar como mucho N imágenes")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo listar candidatos: no llama al VLM ni escribe nada")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    header(f"Backfill de ubicación · {run_dir.name}")

    scanned = scan_run(run_dir)
    if scanned is None:
        sys.exit(1)
    candidates, videos, already = scanned
    info(f"  Candidatos (imagen, 0 personas, sin ubicación): [cyan]{len(candidates)}")
    if already:
        info(f"  Ya puntuadas en pasadas anteriores: [dim]{len(already)}[/] "
             "(no se re-puntúan, pero sí se fusionan)")
    if videos:
        warn(f"  {len(videos)} vídeo(s) sin escenas NO son backfilleables "
             "(la Fase A antigua no registraba las escenas vacías y sus frames son efímeros)")
        if not args.dry_run:
            nb = run_dir / "ubicacion_no_backfilleable.txt"
            nb.write_text("\n".join(v.name for v in videos) + "\n", encoding="utf-8")
            info(f"  listado → {nb.name}")

    if args.limit and len(candidates) > args.limit:
        candidates = candidates[:args.limit]
        info(f"  --limit {args.limit} → se procesan {len(candidates)}")
    if not candidates:
        if already and (args.output or args.merge_into) and not args.dry_run:
            info("  Nada que puntuar; se fusiona lo ya puntuado.")
        else:
            info("  Nada que hacer."); return
    if args.dry_run:
        for jf, _s, _p, src in candidates[:20]:
            info(f"    [dim]{src.name}")
        info("  --dry-run: no se ha llamado al VLM ni se ha escrito nada."); return

    # Semilla: lo puntuado en pasadas anteriores, para que el merge sea COMPLETO.
    rows = [{"Img ID": src.stem, "Archivo": src.name, "Ruta Img.": str(src),
             UBIC_COL: loc, "Ruta JSON": None} for src, loc in already]
    n_fail = 0
    # El VLM solo se carga si hay algo que puntuar (reanudar y fusionar no lo necesita).
    clf = build_scene_classifier() if candidates else None
    with make_progress() as progress:
        task = progress.add_task("Ubicación de escena", total=len(candidates))
        for jf, summary, payload, src in candidates:
            progress.update(task, advance=1, description=f"[dim]{src.name[:40]}")
            img = cv2.imread(str(src))
            res = clf.classify_location_only(img) if img is not None else None
            if not (res and res.get("success")):
                n_fail += 1
                continue
            entry = {
                "track_id": None,
                "frame": 0,
                "scene_level": True,
                "location": res.get("location"),
                "raw_response": res.get("raw_response"),
            }
            payload.setdefault("location_classifications", []).append(entry)
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            rows.append({
                "Img ID": src.stem,
                "Archivo": src.name,
                "Ruta Img.": str(src),
                UBIC_COL: res.get("location"),
                "Ruta JSON": str(jf),
            })

    n_new = len(rows) - len(already)
    success(f"  {n_new} imagen(es) puntuadas ahora"
            + (f" + {len(already)} de pasadas anteriores = {len(rows)} a fusionar" if already else "")
            + (f" · {n_fail} fallo(s)" if n_fail else ""))
    if not rows:
        return
    df_rows = pd.DataFrame(rows)
    if args.output:
        df_rows.to_excel(args.output, index=False)
        success(f"  sidecar → {args.output}")
    if args.merge_into:
        merge_into_master(args.merge_into, df_rows, args.merge_sheet, not args.no_backup)


if __name__ == "__main__":
    main()
