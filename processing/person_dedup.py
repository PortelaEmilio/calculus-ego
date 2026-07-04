"""Dedup de personas dentro de una (sub)escena de vídeo.

El tracker de YOLO FRAGMENTA (la misma persona reaparece con otro track_id tras una
oclusión → IDs secuenciales 1002,1005,1009…) y el fallback ``det_{frame}_{idx}`` crea
una "persona" por cada detección de cada frame cuando el tracker no devuelve ID. Ambas
cosas inflan el conteo por escena: la misma persona se cuenta N veces.

Este módulo fusiona los track_id que son la MISMA persona combinando dos señales:
  1. **Apariencia** — embedding ResNet18 (ImageNet) del mejor crop; coseno ≥ umbral.
  2. **Co-ocurrencia** — dos tracks que aparecen en el MISMO frame son necesariamente
     personas distintas → NUNCA se fusionan. Esto evita colapsar multitudes reales
     (jugadores con la misma camiseta) por pura similitud de apariencia.

Degrada con elegancia: si torch/torchvision/los pesos no cargan, devuelve los
candidatos intactos (el pipeline sigue, solo sin dedup).
"""
from __future__ import annotations

import numpy as np

_MODEL = None
_TORCH = None
_TF = None
_DEVICE = None
_INIT_FAILED = False


def _lazy_init():
    """Carga ResNet18 (sin fc) una sola vez. Marca _INIT_FAILED si no se puede."""
    global _MODEL, _TORCH, _TF, _DEVICE, _INIT_FAILED
    if _MODEL is not None or _INIT_FAILED:
        return
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision import transforms

        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        net.fc = nn.Identity()  # → embedding de 512-d (tras avgpool)
        net.eval()
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        net.to(_DEVICE)
        _MODEL = net
        _TORCH = torch
        _TF = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    except Exception as e:  # pragma: no cover
        _INIT_FAILED = True
        try:
            from ui import warn
            warn(f"  Dedup de personas desactivado (no cargó ResNet18: {e})")
        except Exception:
            pass


def embed_crops(crops: list[np.ndarray]) -> np.ndarray | None:
    """Embeddings L2-normalizados (n, 512) de crops BGR (OpenCV). None si falla."""
    _lazy_init()
    if _MODEL is None:
        return None
    try:
        import cv2
        tensors = []
        for c in crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            tensors.append(_TF(rgb))
        batch = _TORCH.stack(tensors).to(_DEVICE)
        with _TORCH.no_grad():
            feats = _MODEL(batch)
        feats = _TORCH.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()
    except Exception as e:  # pragma: no cover
        try:
            from ui import warn
            warn(f"  Dedup: fallo al embeber crops ({e}) → sin dedup en esta escena")
        except Exception:
            pass
        return None


def dedup_candidates(best: dict, sim_threshold: float = 0.86) -> dict:
    """Fusiona track_id que son la misma persona dentro de una (sub)escena.

    `best`: {track_id: {'general': {crop,score,frame_local_idx,bbox,kpts},
                        'body_shape': {...}|None, 'frames': set[int]|None}}.
    Devuelve un dict con la MISMA forma pero con los fragmentos colapsados: por cada
    cluster se conserva el candidato de mayor `general.score` (representante) y el
    mejor crop `body_shape` del cluster. Si no hay modelo, devuelve `best` intacto.
    """
    items = [(tid, c) for tid, c in best.items()
             if c.get('general') and c['general'].get('crop') is not None]
    if len(items) <= 1:
        return best

    crops = [c['general']['crop'] for _, c in items]
    emb = embed_crops(crops)
    if emb is None:
        return best  # degradación: sin dedup

    n = len(items)
    frames = [set(c.get('frames') or ()) for _, c in items]
    scores = [float(c['general'].get('score', 0.0)) for _, c in items]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    assigned = [-1] * n
    clusters: list[list[int]] = []
    for i in order:
        if assigned[i] != -1:
            continue
        cid = len(clusters)
        assigned[i] = cid
        members = [i]
        cluster_frames = set(frames[i])
        for j in order:
            if assigned[j] != -1 or j == i:
                continue
            # co-ocurrencia: si j comparte frame con algún miembro → persona distinta
            if frames[j] & cluster_frames:
                continue
            # apariencia contra el representante del cluster
            if float(np.dot(emb[i], emb[j])) >= sim_threshold:
                assigned[j] = cid
                members.append(j)
                cluster_frames |= frames[j]
        clusters.append(members)

    out: dict = {}
    for members in clusters:
        rep_tid, rep_cand = items[members[0]]  # mayor score (order desc)
        new_cand = dict(rep_cand)
        best_bs = rep_cand.get('body_shape')
        merged_frames = set(frames[members[0]])
        for k in members[1:]:
            _, c = items[k]
            bs = c.get('body_shape')
            if bs and (best_bs is None or bs.get('score', 0) > best_bs.get('score', 0)):
                best_bs = bs
            merged_frames |= set(c.get('frames') or ())
        new_cand['body_shape'] = best_bs
        new_cand['frames'] = merged_frames
        out[rep_tid] = new_cand
    return out
