"""Índice y resolutor de metadatos post↔archivo para el modo `carpeta` de main.py.

Cada archivo de `multimedia_downloads/` se llama `{media_id}.{ext}`. Para posts de un
solo medio (videos/photos) el `media_id` == `id` del post. Para álbumes/carruseles el
`id` del post es un contenedor que NO aparece como archivo; cada miembro se descarga con
su propio media-id, presente en el array JSON `multimedia` de los CSV RAW.

`build_metadata_index(sources)` recorre una lista ORDENADA de CSV (fuentes) por
prioridad y devuelve un `MetaResolver` que, dado el `stem` de un archivo, resuelve el
post y sus metadatos con la PRIMERA fuente que casa:

    stem -> (post_id, meta_dict, media_url, match_source)

Prioridad por defecto (decisión del usuario 2026-07-09): masculinity_RAW_7 PRIMERO,
luego los RAW más antiguos (6→2) + los dumps `masculinity-IG-Raw` / `raw_nuevas_descargas`,
y `masculinity-IG-limpio` como último recurso. Un `stem` que no casa en NINGUNA fuente
queda con `Match Source="none"` → el orquestador NO lo procesa (ver `_process_directory`).

`attach(row, stem)` rellena in-place las columnas de salida (unión de columnas de todas
las fuentes con prefijo `Meta `, más `Post ID`/`Media ID`/`Media URL`/`Match Source`).
"""

import ast
import json
from pathlib import Path

import pandas as pd

# Cadena de fuentes por defecto (nombre de fichero en el repo, tag para `Match Source`,
# ¿tiene columna `multimedia`?). Orden = PRIORIDAD (la primera que casa gana). Todas
# menos limpio son dumps RAW con el array `multimedia` (resuelven álbumes + URLs).
DEFAULT_SOURCES = [
    ("masculinity_RAW_7.csv",     "raw7",       True),
    ("#masculinity_RAW_6.csv",    "raw6",       True),
    ("#masculinity_RAW_5.csv",    "raw5",       True),
    ("#masculinity_RAW_4.csv",    "raw4",       True),
    ("#masculinity_RAW_3.csv",    "raw3",       True),
    ("#masculinity_RAW_2.csv",    "raw2",       True),
    ("masculinity-IG-Raw.csv",    "raw_ig",     True),
    ("raw_nuevas_descargas.csv",  "raw_nuevas", True),
    ("masculinity-IG-limpio.csv", "limpio",     False),
]


def _parse_multimedia(val):
    """Parsea la celda `multimedia` (array JSON de {id,type,url,...}). Robusto a
    NaN/vacío/comillas simples. Devuelve lista (posiblemente vacía)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []


class MetaResolver:
    def __init__(self, by_id, src_of, inner_to_post, inner_url, meta_cols):
        self._by_id = by_id                # {post_id: {col: val}}  (first-wins por prioridad)
        self._src_of = src_of              # {post_id: tag}         (fuente que aportó el post)
        self._inner_to_post = inner_to_post  # {media_id: post_id}  (miembro de álbum → contenedor)
        self._inner_url = inner_url        # {media_id: url}
        self._meta_cols = list(meta_cols)  # unión de columnas de todas las fuentes

    def columns(self):
        """Columnas de metadatos que se anexan a cada fila del Excel, en orden."""
        return (["Post ID", "Media ID", "Media URL", "Match Source"]
                + [f"Meta {c}" for c in self._meta_cols])

    def resolve(self, stem):
        """Devuelve (post_id, meta_dict|None, media_url|None, match_source).

        Recorre las fuentes por prioridad (implícita en el orden de inserción de
        `by_id`/`inner_to_post`, first-wins). `match_source`:
          - "<tag>"        → single-media: el stem ES un id de post de esa fuente.
          - "<tag>-album"  → miembro de álbum cuyo contenedor está en esa fuente.
          - "album-nometa" → miembro de álbum sin fila de metadatos en ninguna fuente.
          - "none"         → sin correspondencia (NO se procesa)."""
        stem = str(stem)
        url = self._inner_url.get(stem)
        post = self._inner_to_post.get(stem)  # contenedor si es miembro de álbum
        if stem in self._by_id:
            return stem, self._by_id[stem], url, self._src_of[stem]
        if post is not None and post in self._by_id:
            return post, self._by_id[post], url, self._src_of[post] + "-album"
        if post is not None:
            return post, None, url, "album-nometa"
        return stem, None, url, "none"

    def has_match(self, stem):
        """True si el archivo casa con alguna fuente (≠ 'none'). Los que NO casan no
        se procesan (decisión del usuario 2026-07-09)."""
        return self.resolve(stem)[3] != "none"

    def attach(self, row, stem):
        """Rellena in-place las columnas de metadatos de `row` para el archivo `stem`."""
        post_id, meta, url, src = self.resolve(stem)
        row["Post ID"] = post_id
        row["Media ID"] = str(stem)
        row["Media URL"] = url
        row["Match Source"] = src
        for c in self._meta_cols:
            row[f"Meta {c}"] = meta.get(c) if meta else None
        return row


def build_metadata_index(sources):
    """Construye el índice recorriendo `sources` = lista ORDENADA de (path, tag, has_mm).
    First-wins: la primera fuente (mayor prioridad) que trae un post/inner-id gana.
    Tolera que falte cualquier fichero.

    `dtype=str` SIEMPRE (los `id` son numéricos de hasta 16 dígitos; nunca int/float por
    la pérdida de precisión >2^53). `keep_default_na=False` → celdas vacías = "" en vez de
    NaN, para no arrastrar floats NaN al Excel. `encoding=utf-8-sig` → tolera el BOM de RAW."""
    by_id, src_of, inner_to_post, inner_url = {}, {}, {}, {}
    meta_cols = []
    for path, tag, has_mm in sources:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        for c in df.columns:
            if c != "multimedia" and c not in meta_cols:  # el blob JSON no va como columna
                meta_cols.append(c)
        for rec in df.to_dict(orient="records"):
            pid = str(rec.get("id", "")).strip()
            if pid and pid not in by_id:
                by_id[pid] = rec
                src_of[pid] = tag
            if has_mm:
                for el in _parse_multimedia(rec.get("multimedia")):
                    if not isinstance(el, dict):
                        continue
                    mid = el.get("id")
                    if mid is None:
                        continue
                    mid = str(mid).strip()
                    if not mid:
                        continue
                    inner_to_post.setdefault(mid, pid)
                    if el.get("url"):
                        inner_url.setdefault(mid, el.get("url"))
        del df
    return MetaResolver(by_id, src_of, inner_to_post, inner_url, meta_cols)


def build_default_index(repo_dir):
    """Índice con la cadena de fuentes por defecto (`DEFAULT_SOURCES`) buscada en `repo_dir`."""
    repo = Path(repo_dir)
    sources = [(repo / name, tag, has_mm) for name, tag, has_mm in DEFAULT_SOURCES]
    return build_metadata_index(sources)


if __name__ == "__main__":
    # Autotest: python -m processing.metadata_join <repo_dir> <stem1> [stem2 ...]
    import sys
    from collections import Counter
    repo = sys.argv[1]
    stems = sys.argv[2:]
    resolver = build_default_index(repo)
    print(f"columns ({len(resolver.columns())}): {resolver.columns()[:8]} ...")
    for st in stems:
        pid, meta, url, src = resolver.resolve(st)
        uname = meta.get("post_owner.username") if meta else None
        ctype = meta.get("content_type") if meta else None
        print(f"  {st}: src={src} post_id={pid} content_type={ctype} "
              f"user={uname} url={'yes' if url else 'no'}")
