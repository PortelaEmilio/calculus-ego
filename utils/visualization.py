"""
Utilidades de visualización: fuentes PIL, dibujo de esqueleto, anotación de detecciones
y helpers de pose (is_waist_visible, has_five_face_keypoints_visible, etc.).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from PIL import ImageFont, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    ImageFont = None
    ImageDraw = None

from config import (
    VISUALIZATION, SKELETON_CONNECTIONS, LIMB_COLORS,
    CATEGORY_COLORS, CATEGORY_TEXT_COLOR,
    ENABLE_POSE_ESTIMATION, ENABLE_TRACKING, ENABLE_PERSON_ID_LABEL,
    ENABLE_BEAUTY_ESTIMATION,
    ENABLE_GENDER_CLASSIFICATION, ENABLE_AGE_CLASSIFICATION,
    ENABLE_BEHAVIOUR_CLASSIFICATION, ENABLE_ACTIVITY_CLASSIFICATION,
    ENABLE_BODY_DISPLAY_CLASSIFICATION, ENABLE_LOCATION_CLASSIFICATION,
    ENABLE_BODY_SHAPE_CLASSIFICATION, ENABLE_ACCESSORY_CLASSIFICATION,
    ENABLE_SOCIAL_DISTANCE, USE_LIBERATION_SANS, PERSON_CLASS_ID,
)

# ============================================================================
# INICIALIZACIÓN DE FUENTE PIL/PILLOW
# ============================================================================

# Inicializar fuente Liberation Sans con PIL/Pillow
PIL_FONT = None
PIL_FONT_CACHE: dict[int, object] = {}  # caché de fuentes por tamaño
if PIL_AVAILABLE and USE_LIBERATION_SANS:
    try:
        font_path = VISUALIZATION["text_font_path"]
        if os.path.exists(font_path):
            PIL_FONT = ImageFont.truetype(font_path, VISUALIZATION["text_font_size"])
            print(f"✅ Fuente Liberation Sans cargada desde {font_path}")
        else:
            print(f"⚠️ No se encontró Liberation Sans en {font_path}")
            print("   Intentando rutas alternativas...")
            # Intentar rutas alternativas comunes
            alt_paths = [
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-ExtraLight.ttf",       # Fedora (fina)
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",         # Debian/Ubuntu (fina)
                "/usr/share/fonts/dejavu/DejaVuSans-ExtraLight.ttf",                  # Otras distros
                "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",  # Fedora
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",    # Debian/Ubuntu
                "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",        # Otras distros
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",             # Arch
                "/usr/share/fonts/liberation2/LiberationSans-Regular.ttf"             # Alternativa
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    PIL_FONT = ImageFont.truetype(alt_path, VISUALIZATION["text_font_size"])
                    VISUALIZATION["text_font_path"] = alt_path
                    print(f"✅ Fuente Liberation Sans cargada desde {alt_path}")
                    break
            else:
                print("⚠️ No se encontró Liberation Sans. Intentando fuente predeterminada...")
                # Intentar cargar una fuente predeterminada de PIL
                try:
                    PIL_FONT = ImageFont.load_default()
                    print("✅ Usando fuente predeterminada de PIL")
                except:
                    PIL_FONT = None
    except Exception as e:
        print(f"⚠️ Error al cargar fuente PIL: {e}")
        PIL_FONT = None
elif not USE_LIBERATION_SANS:
    print("ℹ️ Uso de Liberation Sans desactivado. Usando fuentes predeterminadas de OpenCV.")


def _get_pil_font(font_size: int):
    """Devuelve una fuente PIL cacheada para el tamaño solicitado."""
    global PIL_FONT_CACHE, PIL_FONT
    if not PIL_AVAILABLE or PIL_FONT is None:
        return None
    if font_size not in PIL_FONT_CACHE:
        font_path = VISUALIZATION.get("text_font_path", "")
        try:
            if font_path and os.path.exists(font_path):
                PIL_FONT_CACHE[font_size] = ImageFont.truetype(font_path, font_size)
            else:
                PIL_FONT_CACHE[font_size] = PIL_FONT  # fallback al tamaño global
        except Exception:
            PIL_FONT_CACHE[font_size] = PIL_FONT
    return PIL_FONT_CACHE[font_size]


def put_text_pil(img, text, org, font_size, color, thickness=-1):
    """
    Función auxiliar para dibujar texto usando PIL/Pillow si está disponible.
    Si PIL no está disponible, usa cv2.putText() con fuente predeterminada.

    Args:
        img: Imagen BGR de OpenCV donde dibujar el texto
        text: Texto a dibujar
        org: Coordenadas (x, y) donde dibujar el texto (esquina inferior izquierda)
        font_size: Tamaño de la fuente en píxeles (se aplica realmente con PIL)
        color: Color del texto (B, G, R)
        thickness: Grosor del texto (solo para fallback de OpenCV)

    Returns:
        img: Imagen con el texto dibujado
    """
    font = _get_pil_font(font_size)
    if font is not None:
        S = max(1, int(VISUALIZATION.get("text_supersample", 1)))
        big_font = _get_pil_font(font_size * S) if S > 1 else font
        if big_font is None:
            big_font = font
            S = 1

        color_rgb = (color[2], color[1], color[0])
        x, y_bottom = org

        measure_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        measure_draw = ImageDraw.Draw(measure_img)
        try:
            bbox = measure_draw.textbbox((0, 0), text, font=big_font)
            big_w = bbox[2] - bbox[0]
            big_h = bbox[3] - bbox[1]
            big_offset_x = bbox[0]
            big_offset_y = bbox[1]
        except Exception:
            big_w = font_size * S * len(text)
            big_h = font_size * S
            big_offset_x = 0
            big_offset_y = 0

        if big_w <= 0 or big_h <= 0:
            return img

        pad = max(2, S)
        canvas_w = big_w + pad * 2
        canvas_h = big_h + pad * 2
        text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        text_draw.text((pad - big_offset_x, pad - big_offset_y), text,
                       font=big_font, fill=color_rgb + (255,))

        if S > 1:
            target_w = max(1, canvas_w // S)
            target_h = max(1, canvas_h // S)
            text_layer = text_layer.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            target_w, target_h = canvas_w, canvas_h

        # Posición top-left donde compositar (PIL usa superior-izquierda; OpenCV
        # usa inferior-izquierda en put_text_pil, así que ajustamos por la altura
        # real del glifo, no por el padding)
        try:
            base_bbox = measure_draw.textbbox((0, 0), text, font=font)
            text_h_target = base_bbox[3] - base_bbox[1]
        except Exception:
            text_h_target = font_size
        paste_x = int(round(x - (pad // S if S > 1 else pad)))
        paste_y = int(round(y_bottom - text_h_target - (pad // S if S > 1 else pad)))

        img_h, img_w = img.shape[:2]
        x0 = max(0, paste_x)
        y0 = max(0, paste_y)
        x1 = min(img_w, paste_x + target_w)
        y1 = min(img_h, paste_y + target_h)
        if x1 <= x0 or y1 <= y0:
            return img

        crop = text_layer.crop((x0 - paste_x, y0 - paste_y,
                                 x1 - paste_x, y1 - paste_y))
        crop_arr = np.array(crop)
        alpha = crop_arr[:, :, 3:4].astype(np.float32) / 255.0
        text_bgr = crop_arr[:, :, [2, 1, 0]].astype(np.float32)
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (text_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    else:
        # Fallback a fuente predeterminada de OpenCV (sin PIL disponible)
        scale = font_size / 30
        cv2.putText(
            img=img,
            text=text,
            org=org,
            fontFace=VISUALIZATION["text_font"],
            fontScale=scale,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return img


# ============================================================================
# HELPERS DE POSE Y KEYPOINTS
# ============================================================================

def is_shoulder_visible(keypoints: np.ndarray, min_conf: float = 0.65) -> bool:
    """Devuelve True si AMBOS hombros (kp 5 y 6) tienen confianza ≥ min_conf.

    Gate relajado para body_shape iter5: en lugar de exigir cintura completa
    (hombros + caderas via YOLO), basta con que el VLM vea ambos hombros para
    intentar la clasificación. La visibilidad de caderas la juzga el propio
    VLM en STEP 1 del prompt.
    """
    if keypoints is None or len(keypoints) < 7:
        return False
    return bool(keypoints[5][2] >= min_conf and keypoints[6][2] >= min_conf)


def is_waist_visible(keypoints: np.ndarray, min_conf: float = 0.5) -> bool:
    """
    Detecta si la cintura es visible en los keypoints.
    Se considera visible cuando se ven los hombros (keypoints 5 y 6)
    y las caderas (keypoints 11 y 12).

    COCO keypoints:
    - 5: left_shoulder
    - 6: right_shoulder
    - 11: left_hip
    - 12: right_hip

    Args:
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        min_conf: Confianza mínima para considerar un keypoint visible

    Returns:
        True si la cintura es visible, False en caso contrario
    """
    if len(keypoints) < 13:  # Necesitamos al menos hasta el keypoint 12
        return False

    # Verificar que los keypoints de hombros y caderas sean visibles
    left_shoulder_visible = bool(keypoints[5][2] > min_conf)
    right_shoulder_visible = bool(keypoints[6][2] > min_conf)
    left_hip_visible = bool(keypoints[11][2] > min_conf)
    right_hip_visible = bool(keypoints[12][2] > min_conf)

    # Requerimos que estén visibles al menos ambos hombros y ambas caderas
    return left_shoulder_visible and right_shoulder_visible and left_hip_visible and right_hip_visible


def count_visible_keypoints(keypoints: np.ndarray, min_conf: float = 0.5) -> int:
    """
    Cuenta el número de keypoints visibles (confianza > min_conf).

    Args:
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        min_conf: Confianza mínima para considerar un keypoint visible

    Returns:
        Número de keypoints visibles
    """
    if len(keypoints) == 0:
        return 0
    return int(np.sum(keypoints[:, 2] > min_conf))


def is_frontal_pose_with_waist(keypoints: np.ndarray, min_conf: float = 0.5) -> bool:
    """
    Detecta si la persona está de frente mostrando hombros y caderas.

    Se considera de frente cuando:
    1. Los 4 keypoints necesarios son visibles (ambos hombros y ambas caderas)
    2. Los hombros están aproximadamente a la misma altura (diferencia Y < 20% del ancho de hombros)
    3. Las caderas están aproximadamente a la misma altura (diferencia Y < 20% del ancho de caderas)

    COCO keypoints:
    - 5: left_shoulder
    - 6: right_shoulder
    - 11: left_hip
    - 12: right_hip

    Args:
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        min_conf: Confianza mínima para considerar un keypoint visible

    Returns:
        True si la persona está de frente con cintura visible, False en caso contrario
    """
    # Primero verificar que la cintura sea visible
    if not is_waist_visible(keypoints, min_conf):
        return False

    # Obtener coordenadas de hombros y caderas
    left_shoulder = keypoints[5][:2]  # (x, y)
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]

    # Calcular diferencias en altura (Y) para hombros
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    shoulder_height_diff = abs(right_shoulder[1] - left_shoulder[1])

    # Calcular diferencias en altura (Y) para caderas
    hip_width = abs(right_hip[0] - left_hip[0])
    hip_height_diff = abs(right_hip[1] - left_hip[1])

    # Verificar que los hombros estén aproximadamente a la misma altura
    # (diferencia Y menor al 20% del ancho de hombros)
    if shoulder_width > 0:
        shoulder_threshold = shoulder_width * 0.2
        if shoulder_height_diff > shoulder_threshold:
            return False

    # Verificar que las caderas estén aproximadamente a la misma altura
    # (diferencia Y menor al 20% del ancho de caderas)
    if hip_width > 0:
        hip_threshold = hip_width * 0.2
        if hip_height_diff > hip_threshold:
            return False

    # Si ambos hombros y caderas están aproximadamente alineados horizontalmente,
    # la persona está de frente
    return True


def get_color_for_track_id(track_id: int | str) -> tuple:
    """
    Genera un color único y consistente para un track_id dado.

    Args:
        track_id: ID de tracking de la persona

    Returns:
        Tupla (B, G, R) con valores entre 0-255
    """
    # Lista de colores vibrantes y distinguibles
    colors = [
        (255, 100, 100),  # Azul claro
        (100, 255, 100),  # Verde claro
        (100, 100, 255),  # Rojo claro
        (255, 255, 100),  # Cian
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Amarillo
        (255, 150, 100),  # Azul-verde
        (150, 255, 100),  # Verde-amarillo
        (100, 150, 255),  # Rojo-púrpura
        (255, 200, 150),  # Azul pastel
        (200, 255, 150),  # Verde pastel
        (150, 200, 255),  # Rosa pastel
        (200, 100, 255),  # Púrpura
        (100, 200, 255),  # Naranja claro
        (255, 100, 200),  # Azul-púrpura
        (200, 255, 100),  # Verde lima
        (100, 255, 200),  # Verde agua
        (255, 200, 100),  # Azul cielo
    ]

    # Convertir track_id a entero si es string
    if isinstance(track_id, str):
        # Hash del string para obtener un índice consistente
        track_id_int = hash(track_id)
    else:
        track_id_int = track_id

    # Usar módulo para obtener un índice en la lista de colores
    color_idx = track_id_int % len(colors)
    return colors[color_idx]


def has_five_face_keypoints_visible(keypoints: np.ndarray, min_conf: float = 0.5) -> bool:
    """
    Verifica si los 5 keypoints faciales están visibles con confianza suficiente.

    Args:
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        min_conf: Confianza mínima para considerar un keypoint válido

    Returns:
        True si los 5 keypoints faciales (nose, left_eye, right_eye, left_ear, right_ear)
        tienen confianza > min_conf, False en caso contrario
    """
    if keypoints is None or len(keypoints) < 5:
        return False

    # Keypoints faciales: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)
    face_keypoint_indices = [0, 1, 2, 3, 4]

    # Verificar que los 5 keypoints tengan confianza suficiente
    valid_count = 0
    for idx in face_keypoint_indices:
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            if conf > min_conf:
                valid_count += 1

    return valid_count == 5


def count_face_keypoints_visible(keypoints: np.ndarray, min_conf: float = 0.5) -> int:
    """Nº de keypoints faciales COCO (nose, eyes, ears; índices 0-4) con conf > min_conf.
    Usado por el path de belleza para el gate ≥ BEAUTY_MIN_FACE_KEYPOINTS (reemplaza a
    has_five_face_keypoints_visible, que exigía los 5)."""
    if keypoints is None:
        return 0
    return len(_valid_face_keypoints(keypoints, min_conf))


def get_face_keypoint_centroid(keypoints: np.ndarray, min_conf: float = 0.3):
    """
    Calcula el centroide de los keypoints faciales visibles (índices 0-4 COCO:
    nose, left_eye, right_eye, left_ear, right_ear).

    Args:
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        min_conf: Confianza mínima para considerar un keypoint válido

    Returns:
        Tupla (cx, cy) con las coordenadas del centroide, o None si no hay
        ningún keypoint facial visible.
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    valid_pts = []
    for ki in range(min(5, len(keypoints))):
        x, y, conf = float(keypoints[ki][0]), float(keypoints[ki][1]), float(keypoints[ki][2])
        if conf > min_conf:
            valid_pts.append((x, y))
    if not valid_pts:
        return None
    return (float(np.mean([p[0] for p in valid_pts])),
            float(np.mean([p[1] for p in valid_pts])))


def _valid_face_keypoints(keypoints: np.ndarray, min_conf: float) -> dict:
    """Keypoints faciales COCO válidos (conf > min_conf) indexados por su índice:
    nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)."""
    valid = {}
    for idx in (0, 1, 2, 3, 4):
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            if conf > min_conf:
                valid[idx] = (float(x), float(y))
    return valid


def face_tight_px(keypoints: np.ndarray, min_conf: float = 0.5) -> int | None:
    """Métrica face_px HISTÓRICA: mín(alto, ancho) de la caja apretada de keypoints
    faciales + 30% de margen (= dimensiones del crop-tira previo al fix de geometría
    2026-07-10). Se conserva como proxy del tamaño real de la cara para que la
    calibración de low_res (LOW_RES_PX≈120) siga valiendo aunque el crop enviado al
    estimador sea ahora cabeza completa."""
    valid = _valid_face_keypoints(keypoints, min_conf)
    if len(valid) < 3:
        return None
    xs = [p[0] for p in valid.values()]
    ys = [p[1] for p in valid.values()]
    h = (max(ys) - min(ys)) * 1.6
    w = (max(xs) - min(xs)) * 1.6
    if h <= 0 or w <= 0:
        return None
    return int(min(h, w))


def extract_face_crop(frame: np.ndarray, keypoints: np.ndarray, bbox: tuple,
                      margin_percent: float = 0.3, min_conf: float = 0.5,
                      min_points: int = 5) -> np.ndarray | None:
    """
    Extrae el crop de la CABEZA COMPLETA desde el frame usando los keypoints faciales.

    ⚠️ Geometría corregida el 2026-07-10: la versión anterior devolvía la caja min/max
    de los 5 keypoints + 30% — pero esa caja solo abarca ojos→nariz en vertical (las
    orejas están a la altura de los ojos), así que el crop era una TIRA sin frente,
    boca ni mentón, fuera de la distribución de entrenamiento del estimador de belleza
    (retratos cabeza+hombros SCUT/CFD/MEBeauty) → anclaba las notas a ~2-3. Ahora se
    expande asimétricamente en función del ANCHO de cara a cabeza completa (verificado:
    la misma cara pasó de 2.3 a 8.1 con el mismo modelo).

    Args:
        frame: Frame completo BGR de OpenCV
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        bbox: Tupla (x1, y1, x2, y2) del bounding box de la persona (mismas coordenadas
            que los keypoints); el crop se recorta a este bbox expandido 10% para no
            invadir caras vecinas en multitudes. None → sin ese clamp.
        margin_percent: IGNORADO (legacy, se mantiene por compatibilidad de firma)
        min_conf: Confianza mínima para considerar un keypoint válido
        min_points: Nº mínimo de keypoints faciales válidos para extraer el crop
            (default 5 = cara completa; el path de belleza de vídeo lo baja para
            priorizar el frame con MÁS keypoints faciales aunque no estén los 5)

    Returns:
        Crop de la cabeza o None si no se puede extraer
    """
    if len(keypoints) < 5:
        return None

    valid = _valid_face_keypoints(keypoints, min_conf)

    # Necesitamos al menos `min_points` puntos faciales para el crop de belleza
    if len(valid) < min_points:
        return None

    # Caja apretada de los keypoints faciales
    xs = [p[0] for p in valid.values()]
    ys = [p[1] for p in valid.values()]
    kx1, ky1 = min(xs), min(ys)
    kx2, ky2 = max(xs), max(ys)

    # Ancho efectivo de cara: con los 5 kpts la caja es ~oreja-a-oreja. Si faltan las
    # orejas (min_points bajado en vídeo) la caja se estrecha a la distancia
    # inter-ocular → floor de 2.2×IOD como ancho real de cabeza.
    w_eff = kx2 - kx1
    if 1 in valid and 2 in valid:
        iod = float(np.hypot(valid[1][0] - valid[2][0], valid[1][1] - valid[2][1]))
        w_eff = max(w_eff, 2.2 * iod)
    if w_eff <= 0:
        return None

    # Expansión asimétrica a cabeza completa (la caja de kpts solo cubre ojos→nariz)
    face_x1 = kx1 - 0.25 * w_eff
    face_x2 = kx2 + 0.25 * w_eff
    face_y1 = ky1 - 0.8 * w_eff    # frente + pelo
    face_y2 = ky2 + 1.0 * w_eff    # boca + mentón

    # Clamp al bbox de persona expandido 10% (guard multitudes) y al frame
    if bbox is not None:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        mx, my = 0.10 * (bx2 - bx1), 0.10 * (by2 - by1)
        face_x1 = max(face_x1, bx1 - mx)
        face_y1 = max(face_y1, by1 - my)
        face_x2 = min(face_x2, bx2 + mx)
        face_y2 = min(face_y2, by2 + my)

    face_x1 = max(0, int(face_x1))
    face_y1 = max(0, int(face_y1))
    face_x2 = min(frame.shape[1], int(face_x2))
    face_y2 = min(frame.shape[0], int(face_y2))

    # Verificar que el crop sea válido
    if face_x2 <= face_x1 or face_y2 <= face_y1:
        return None

    # Extraer crop
    face_crop = frame[face_y1:face_y2, face_x1:face_x2]

    return face_crop if face_crop.size > 0 else None


def face_sharpness(crop: np.ndarray) -> float:
    """Nitidez de un crop de cara = varianza del Laplaciano en escala de grises.

    Métrica estándar de enfoque: alta en caras nítidas (~200-1500), baja en caras
    con desenfoque de movimiento (~10-80). Se usa para elegir el mejor frame de
    belleza por escena evitando crops borrosos (los keypoints de pose se detectan
    con alta confianza incluso en caras movidas, así que la confianza NO discrimina
    el blur). Devuelve 0.0 si el crop es inválido.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def bbox_occupancy(bbox, frame_w: int, frame_h: int):
    """Tamaño de un bbox (xyxy) y qué fracción del frame ocupa.

    Devuelve `(bbox_area_px, occupancy_pct)`:
      - `bbox_area_px` = (x2-x1)·(y2-y1) en píxeles (int).
      - `occupancy_pct` = 100·bbox_area/frame_area, redondeado a 2 decimales.
    Es la misma aritmética que el filtro `BBOX_MIN_FRAME_RATIO`, reutilizada para
    reportar "cuánto ocupa" cada persona en la imagen. `bbox` es `(x1,y1,x2,y2)` o
    una lista equivalente; `None`/inválido → `(None, None)`.
    """
    if not bbox or len(bbox) < 4:
        return None, None
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    if not frame_w or not frame_h:   # sin dimensiones de frame: solo el área, sin %
        return int(area), None
    frame_area = max(1, int(frame_w) * int(frame_h))
    return int(area), round(100.0 * area / frame_area, 2)


def get_limb_color(connection_idx: int) -> tuple:
    """Obtiene el color para una conexión del esqueleto según su parte del cuerpo."""
    if connection_idx < 4:
        return LIMB_COLORS["face"]
    elif connection_idx < 8:
        return LIMB_COLORS["torso"]
    elif connection_idx < 10:
        return LIMB_COLORS["left_arm"]
    elif connection_idx < 12:
        return LIMB_COLORS["right_arm"]
    elif connection_idx < 14:
        return LIMB_COLORS["left_leg"]
    else:
        return LIMB_COLORS["right_leg"]


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, fade_alpha: float = 1.0) -> np.ndarray:
    """
    Dibuja el esqueleto completo de una persona sobre el frame.

    Args:
        frame: Imagen BGR de OpenCV
        keypoints: Array de keypoints [num_keypoints, 3] donde 3 = (x, y, conf)
        fade_alpha: 0-1, atenúa el esqueleto (fade-in por escena). 1.0 = opaco.

    Returns:
        Frame con el esqueleto dibujado
    """
    if not ENABLE_POSE_ESTIMATION or fade_alpha <= 0.02:
        return frame

    s = _annotation_scale(frame.shape[0])  # escala por resolución (1.0 a ≤720p)
    min_conf = VISUALIZATION["min_keypoint_conf"]
    glow = VISUALIZATION.get("skeleton_glow", False)
    base_thick = max(1, int(round(VISUALIZATION["skeleton_thickness"] * s)))
    glow_thick = max(1, int(round(VISUALIZATION.get("skeleton_glow_thickness", VISUALIZATION["skeleton_thickness"] + 5) * s)))
    kp_radius = max(1, int(round(VISUALIZATION["keypoint_radius"] * s)))
    kp_color = VISUALIZATION["keypoint_color"]

    # Con fade o glow, dibujamos sobre una capa y la mezclamos → transparencia real.
    partial = fade_alpha < 0.99
    canvas = frame.copy() if (partial or glow) else frame

    if glow:
        # Capa de glow: líneas gruesas translúcidas (se mezcla al final vía addWeighted).
        glow_layer = frame.copy()
        for idx, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x1, y1, conf1 = keypoints[start_idx]
                x2, y2, conf2 = keypoints[end_idx]
                if conf1 > min_conf and conf2 > min_conf:
                    color = get_limb_color(idx)
                    cv2.line(glow_layer, (int(x1), int(y1)), (int(x2), int(y2)),
                             color, glow_thick, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.35, canvas, 0.65, 0, canvas)

    # Líneas finas nítidas encima
    for idx, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]
            if conf1 > min_conf and conf2 > min_conf:
                color = get_limb_color(idx)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                         color, base_thick, lineType=cv2.LINE_AA)

    # Keypoints con halo suave
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > min_conf:
            x, y = int(x), int(y)
            cv2.circle(canvas, (x, y), kp_radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (x, y), kp_radius, kp_color, -1, lineType=cv2.LINE_AA)

    if partial:
        cv2.addWeighted(canvas, fade_alpha, frame, 1.0 - fade_alpha, 0, frame)
    elif glow:
        # canvas ya es una copia con glow+líneas; volcamos al frame
        frame[:] = canvas
    return frame


# ============================================================================
# ATRIBUTOS POR PERSONA → texto legible (compartido classic/chips, SIN confianza)
# ============================================================================

# Etiquetas legibles por clase (EN inglés, SIN porcentaje de confianza — 2026-07-03)
_BEHAVIOUR_LABELS = {
    "demand/affiliation": "Affiliation",
    "demand/seduction":   "Seduction",
    "demand/submission":  "Submission",
    "offer/ideal":        "Ideal",
}
_ACTIVITY_LABELS = {
    "sports": "Sports", "romance": "Romance", "posing": "Posing", "other": "Other",
    "entertaining": "Entertaining", "everyday doings": "Everyday", "no activities": "None",
}
_BODY_DISPLAY_LABELS = {
    "wearing revealing or hardly any clothes": "Revealing",
    "no clothes at all": "Nude",
    "normal clothes": "Normal",
}
_LOCATION_LABELS = {
    "indoors": "Indoors", "wilderness": "Wilderness", "city": "City",
    "no background": "No background",
}
_SILHOUETTE_LABELS = {
    "inverted triangle": "Inv. triangle", "rectangle": "Rectangle",
    "pear": "Pear", "hourglass": "Hourglass", "triangle": "Triangle",
}
_WEIGHT_LABELS = {"thin": "Thin", "median": "Median", "overweight": "Overweight"}
_ATTIRE_LABELS = {
    "underwear/swimwear": "Underwear/Swim", "sportswear": "Sportswear",
    "uniform": "Uniform", "formal": "Formal", "casual": "Casual",
}
_SOCIAL_DISTANCE_LABELS = {
    "intimate distance": "Intimate", "close personal distance": "Close personal",
    "far personal distance": "Far personal", "close social distance": "Close social",
    "far social distance": "Far social", "public distance": "Public",
}
_ACCESSORY_LETTERS = {
    "makeup": "M", "tattoos": "T", "bags": "B", "belts": "Cn",
    "jewelry": "J", "headwear": "S", "eyewear": "G",
}

_BLANK_VALUES = {None, "", "no visible", "not visible", "na", "n/a", "unknown", "none"}


def _blank(value) -> bool:
    """True si el valor de una clase es vacío/no-evaluable."""
    return value is None or (isinstance(value, str) and value.strip().lower() in _BLANK_VALUES)


def _label(mapping: dict, value: str) -> str:
    """Resuelve la etiqueta legible; si no está en el mapa devuelve el valor capitalizado."""
    if value is None:
        return ""
    key = value.strip().lower()
    return mapping.get(key, value.strip().capitalize())


def build_person_display_attrs(gender_info=None, age_info=None, behaviour_info=None,
                                activity_info=None, body_display_info=None,
                                location_info=None, body_shape_info=None,
                                accessory_info=None, social_distance_info=None,
                                beauty_score=None, occupancy=None) -> list[tuple[str, str]]:
    """Construye la lista ordenada [(category_key, display_text)] de una persona.

    Contenido reducido y SIN confianza: silueta/peso/musculatura/distancia social
    solo aparecen si son evaluables (≠ 'not visible'). Compartido por el estilo
    'classic' y 'chips'.
    """
    attrs: list[tuple[str, str]] = []

    if gender_info and gender_info.get("success") and ENABLE_GENDER_CLASSIFICATION:
        g = gender_info.get("gender")
        if not _blank(g):
            attrs.append(("gender", g.strip().capitalize()))

    if age_info and age_info.get("success") and ENABLE_AGE_CLASSIFICATION:
        a = age_info.get("age_group")
        if not _blank(a):
            attrs.append(("age", a.strip()))

    if behaviour_info and behaviour_info.get("success") and ENABLE_BEHAVIOUR_CLASSIFICATION:
        b = behaviour_info.get("behaviour")
        if not _blank(b):
            attrs.append(("behaviour", _label(_BEHAVIOUR_LABELS, b)))

    if activity_info and activity_info.get("success") and ENABLE_ACTIVITY_CLASSIFICATION:
        act = activity_info.get("activity")
        if not _blank(act):
            attrs.append(("activity", _label(_ACTIVITY_LABELS, act)))

    if body_display_info and body_display_info.get("success") and ENABLE_BODY_DISPLAY_CLASSIFICATION:
        bd = body_display_info.get("body_display")
        if not _blank(bd):
            attrs.append(("body_display", _label(_BODY_DISPLAY_LABELS, bd)))

    if location_info and location_info.get("success") and ENABLE_LOCATION_CLASSIFICATION:
        loc = location_info.get("location")
        if not _blank(loc):
            attrs.append(("location", _label(_LOCATION_LABELS, loc)))

    if body_shape_info and body_shape_info.get("success") and ENABLE_BODY_SHAPE_CLASSIFICATION:
        weight = body_shape_info.get("body_weight")
        if not _blank(weight):
            attrs.append(("body_weight", _label(_WEIGHT_LABELS, weight)))
        muscle = body_shape_info.get("muscle")
        if not _blank(muscle) and muscle.strip().lower() == "visible":
            attrs.append(("muscle", "Muscular"))
        attire = body_shape_info.get("attire")
        if not _blank(attire):
            attrs.append(("attire", _label(_ATTIRE_LABELS, attire)))
        # silueta ELIMINADA 2026-07-04

    if social_distance_info and social_distance_info.get("success") and ENABLE_SOCIAL_DISTANCE:
        cat = social_distance_info.get("category")
        if not _blank(cat):
            attrs.append(("social_distance", _label(_SOCIAL_DISTANCE_LABELS, cat)))

    if ENABLE_BEAUTY_ESTIMATION and beauty_score is not None:
        if isinstance(beauty_score, (int, float)):
            attrs.append(("beauty", f"Beauty {float(beauty_score):.1f}"))

    # Tamaño / ocupación del bbox (% del frame). Siempre que se conozca (no gated por flag).
    if occupancy is not None and isinstance(occupancy, (int, float)):
        attrs.append(("size", f"Size {float(occupancy):.1f}%"))

    if accessory_info and accessory_info.get("success") and ENABLE_ACCESSORY_CLASSIFICATION:
        letters = [ltr for key, ltr in _ACCESSORY_LETTERS.items() if accessory_info.get(key) == 1]
        if letters:
            attrs.append(("accessory", "Acc: " + ",".join(letters)))

    return attrs


# ============================================================================
# ESTILO "CHIPS": píldoras redondeadas por atributo + bbox de esquinas
# ============================================================================

def _mix(color_bgr: tuple, alpha: float) -> tuple:
    """Devuelve el color escalado por alpha sobre negro (para atenuar líneas cv2)."""
    a = max(0.0, min(1.0, alpha))
    return tuple(int(c * a) for c in color_bgr)


def _annotation_scale(frame_h: int) -> float:
    """Factor de escala de las anotaciones según la resolución del frame.

    Los tamaños de config están calibrados para ~720p; a 1080p/4K se escalan para
    que chips/esquinas/esqueleto/contador mantengan proporción. A ≤720p el factor
    es 1.0 (retrocompatible). Clamp [1.0, 4.0]."""
    return max(1.0, min(frame_h / 720.0, 4.0))


def draw_corner_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color: tuple, thickness: int = 3, bracket_len: int = 20,
                     fade_alpha: float = 1.0) -> np.ndarray:
    """Dibuja el bounding box como 4 esquinas en L (corner brackets)."""
    if fade_alpha <= 0.02:
        return frame
    s = _annotation_scale(frame.shape[0])
    thickness = int(round(thickness * s))
    bracket_len = int(round(bracket_len * s))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    L = max(6, min(bracket_len, (x2 - x1) // 2, (y2 - y1) // 2))
    t = max(1, thickness)
    if fade_alpha < 0.99:
        overlay = frame.copy()
        _corner_lines(overlay, x1, y1, x2, y2, L, color, t)
        cv2.addWeighted(overlay, fade_alpha, frame, 1.0 - fade_alpha, 0, frame)
    else:
        _corner_lines(frame, x1, y1, x2, y2, L, color, t)
    return frame


def _corner_lines(img, x1, y1, x2, y2, L, color, t):
    # esquina sup-izq
    cv2.line(img, (x1, y1), (x1 + L, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + L), color, t)
    # sup-der
    cv2.line(img, (x2, y1), (x2 - L, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + L), color, t)
    # inf-izq
    cv2.line(img, (x1, y2), (x1 + L, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - L), color, t)
    # inf-der
    cv2.line(img, (x2, y2), (x2 - L, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - L), color, t)


def _measure_text(text: str, font) -> tuple[int, int]:
    """Ancho/alto de un texto con la fuente PIL dada (fallback cv2)."""
    if font is not None:
        try:
            tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bb = tmp.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            pass
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    return w, h


def draw_person_chips(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      track_id, person_color: tuple,
                      attrs: list[tuple[str, str]], fade_alpha: float = 1.0) -> np.ndarray:
    """Dibuja las clases de una persona como píldoras redondeadas flotantes.

    Las píldoras se disponen en filas (flow-wrap) preferentemente encima del bbox;
    si no caben arriba, debajo. Cada píldora usa el color de su categoría
    (CATEGORY_COLORS) con `chip_opacity` y texto oscuro. Sin cabecera de ID.
    `fade_alpha` atenúa el conjunto (fade-in por escena). `person_color` se conserva
    en la firma por compatibilidad pero ya no se usa (las esquinas son blancas fijas).
    """
    if fade_alpha <= 0.02 or not PIL_AVAILABLE:
        if not PIL_AVAILABLE:
            # Fallback mínimo: caja de esquinas, sin píldoras
            return draw_corner_bbox(frame, x1, y1, x2, y2,
                                    VISUALIZATION.get("corner_color", (255, 255, 255)),
                                    VISUALIZATION.get("corner_bracket_thickness", 3),
                                    VISUALIZATION.get("corner_bracket_len", 20), fade_alpha)
        return frame

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    s = _annotation_scale(fh)  # escala por resolución (1.0 a ≤720p, hasta 4× en 4K)
    font_size = int(round(VISUALIZATION.get("chip_font_size", 15) * s))
    font = _get_pil_font(font_size)
    pad_x = int(round(VISUALIZATION.get("chip_pad_x", 9) * s))
    pad_y = int(round(VISUALIZATION.get("chip_pad_y", 5) * s))
    gap = int(round(VISUALIZATION.get("chip_gap", 5) * s))
    radius = int(round(VISUALIZATION.get("chip_radius", 9) * s))
    opacity = float(VISUALIZATION.get("chip_opacity", 0.82)) * max(0.0, min(1.0, fade_alpha))

    # Sin cabecera de ID: los chips son solo los atributos de clase.
    chips = list(attrs)
    if not chips:
        return frame

    # Medir cada píldora
    measured = []  # (key, text, w, h)
    max_text_h = font_size
    for key, text in chips:
        tw, th = _measure_text(text, font)
        max_text_h = max(max_text_h, th)
        measured.append((key, text, tw, th))
    chip_h = max_text_h + pad_y * 2

    # Flow-wrap en filas limitadas al ancho disponible (bbox extendido, mínimo 220px×escala)
    avail_w = max(int(220 * s), min(fw - 8, (x2 - x1) + int(160 * s)))
    rows: list[list[tuple]] = [[]]
    row_w = 0
    for key, text, tw, th in measured:
        cw = tw + pad_x * 2
        if row_w > 0 and row_w + gap + cw > avail_w:
            rows.append([])
            row_w = 0
        rows[-1].append((key, text, cw))
        row_w += (gap if row_w > 0 else 0) + cw
    block_w = max(sum(c[2] for c in r) + gap * (len(r) - 1) for r in rows if r)
    block_h = len(rows) * chip_h + gap * (len(rows) - 1)

    # Posicionar el bloque: encima del bbox; si no cabe, debajo; clamp al frame.
    off = int(round(6 * s))
    bx = int(np.clip(x1, 4, max(4, fw - block_w - 4)))
    by = y1 - block_h - off
    if by < 4:
        by = y2 + off
    if by + block_h > fh - 4:
        by = max(4, fh - block_h - 4)

    # Capa RGBA para todas las píldoras del bloque (un solo composite sobre el ROI).
    rx1, ry1 = bx, by
    rx2, ry2 = min(fw, bx + block_w + 2), min(fh, by + block_h + 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return frame
    layer = Image.new("RGBA", (rx2 - rx1, ry2 - ry1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cy = 0
    for row in rows:
        cx = 0
        for key, text, cw in row:
            fill = CATEGORY_COLORS.get(key, (200, 200, 200))
            txt_color = CATEGORY_TEXT_COLOR
            # BGR→RGB + alpha
            fill_rgba = (fill[2], fill[1], fill[0], int(255 * opacity))
            draw.rounded_rectangle([cx, cy, cx + cw, cy + chip_h],
                                    radius=radius, fill=fill_rgba)
            # texto centrado verticalmente
            tx = cx + pad_x
            ty = cy + (chip_h - font_size) // 2
            draw.text((tx, ty), text, font=font,
                      fill=(txt_color[2], txt_color[1], txt_color[0], int(255 * min(1.0, fade_alpha))))
            cx += cw + gap
        cy += chip_h + gap

    # Composite del ROI
    layer_arr = np.array(layer)
    alpha = layer_arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = layer_arr[:, :, [2, 1, 0]].astype(np.float32)  # RGBA→BGR
    roi = frame[ry1:ry2, rx1:rx2].astype(np.float32)
    frame[ry1:ry2, rx1:rx2] = (rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return frame


def draw_detection_with_info(frame: np.ndarray, box, track_id: int | None = None,
                              beauty_score: float | None = None,
                              gender_info: dict | None = None, age_info: dict | None = None,
                              behaviour_info: dict | None = None, activity_info: dict | None = None,
                              body_display_info: dict | None = None, location_info: dict | None = None,
                              body_shape_info: dict | None = None, accessory_info: dict | None = None,
                              social_distance_info: dict | None = None,
                              person_color: tuple | None = None,
                              occupancy: float | None = None) -> np.ndarray:
    """
    Dibuja un bounding box con información de tracking, género, edad, belleza y distancia social.
    Las métricas se posicionan inteligentemente para estar siempre visibles en el frame.

    Posiciones prioritarias (en orden de preferencia):
    1. Dentro del bbox (esquina inferior derecha)  <-- PRIORIDAD
    2. A la izquierda del bbox
    3. A la derecha del bbox
    4. Arriba del bbox
    5. Abajo del bbox
    6. Dentro del bbox (esquina inferior izquierda, último recurso)

    Args:
        frame: Imagen BGR de OpenCV
        box: Objeto box de YOLO con xyxy, conf, cls
        track_id: ID de tracking (opcional)
        beauty_score: Score de belleza 0-100 (opcional)
        gender_info: Información de género (opcional)
        age_info: Información de edad (opcional)
        behaviour_info: Información de comportamiento (opcional)
        activity_info: Información de actividad (opcional)
        body_display_info: Información de exposición del cuerpo (opcional)
        location_info: Información de ubicación (opcional)
        body_shape_info: Información de tipo de cuerpo (opcional)
        accessory_info: Información de accesorios (opcional)
        social_distance_info: Información de distancia social (opcional)
        person_color: Color único para esta persona (B, G, R) (opcional)

    Returns:
        Frame con la detección dibujada
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Obtener dimensiones del frame
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # Calcular área del bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height

    # Determinar si el bbox es más del 50% del frame
    is_large_bbox = (bbox_area / frame_area) > 0.5

    # Color del bounding box basado en condiciones:
    # - Usar color personalizado por persona si está disponible
    # - Gris para Public Distance (opcional)
    # - Verde por defecto
    if person_color is not None:
        bbox_color = person_color
    elif social_distance_info and social_distance_info.get("success") and social_distance_info.get("category") == "Public Distance":
        bbox_color = (128, 128, 128)  # Gris para Public Distance
    else:
        bbox_color = VISUALIZATION["bbox_color"]  # Verde por defecto

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  bbox_color, VISUALIZATION["bbox_thickness"])

    # Construir lista de etiquetas (una por línea) — SIN confianza (retirada 2026-07-03)
    label_lines = ["Person"]

    if track_id is not None and ENABLE_PERSON_ID_LABEL:
        label_lines.append(f"ID:{track_id}")

    # Atributos de clase compartidos con el estilo chips (ya sin confianza).
    attrs = build_person_display_attrs(
        gender_info, age_info, behaviour_info, activity_info, body_display_info,
        location_info, body_shape_info, accessory_info, social_distance_info, beauty_score,
        occupancy=occupancy,
    )
    label_lines.extend(text for _key, text in attrs)

    # Calcular dimensiones para el fondo
    # Escalar el texto proporcionalmente al alto del bounding box, con clamp
    adaptive_font_size = int(max(11, min(18, bbox_height * 0.025)))
    adaptive_text_scale = adaptive_font_size / 33.0   # 20pt ≈ scale 0.6 → factor 33
    adaptive_text_thickness = max(1, int(adaptive_text_scale))
    line_height = adaptive_font_size + 3  # Altura de cada línea de texto
    max_width = 0

    _pil_font_for_measure = _get_pil_font(adaptive_font_size)
    for line in label_lines:
        if _pil_font_for_measure is not None:
            try:
                from PIL import ImageDraw as _PD, Image as _PI
                _tmp = _PD.Draw(_PI.new("RGB", (1, 1)))
                _bb = _tmp.textbbox((0, 0), line, font=_pil_font_for_measure)
                text_w = _bb[2] - _bb[0]
            except Exception:
                (text_w, _), _ = cv2.getTextSize(
                    line, VISUALIZATION["text_font"],
                    adaptive_text_scale, adaptive_text_thickness
                )
        else:
            (text_w, _), _ = cv2.getTextSize(
                line, VISUALIZATION["text_font"],
                adaptive_text_scale, adaptive_text_thickness
            )
        max_width = max(max_width, text_w)

    total_height = len(label_lines) * line_height + 4

    # Decidir posición de las etiquetas de forma inteligente
    text_padding = 8  # Espacio alrededor del texto

    # Prioridades de posiciones (en orden de preferencia):
    # 1. Dentro del bbox (esquina inferior derecha)  <-- PRIORIDAD MÁXIMA
    # 2. Izquierda del bbox
    # 3. Derecha del bbox
    # 4. Arriba del bbox
    # 5. Abajo del bbox
    # 6. Dentro del bbox (esquina inferior izquierda, último recurso)

    # Opción 1: Dentro del bbox (esquina inferior derecha) — PRIORIDAD
    inside_br_bg_x2 = x2 - 4
    inside_br_bg_y2 = y2 - 4
    inside_br_bg_x1 = x2 - max_width - 12
    inside_br_bg_y1 = y2 - total_height - 4
    # Válida solo si el bloque de texto cabe dentro del bbox
    inside_br_fits_in_bbox = (
        inside_br_bg_x1 >= x1 and inside_br_bg_y1 >= y1 and
        inside_br_bg_x2 <= x2 and inside_br_bg_y2 <= y2
    )

    # Opción 2: A la izquierda del bbox
    left_bg_x1 = x1 - max_width - text_padding
    left_bg_y1 = y1
    left_bg_x2 = x1 - 4
    left_bg_y2 = y1 + total_height

    # Opción 3: A la derecha del bbox
    right_bg_x1 = x2 + 4
    right_bg_y1 = y1
    right_bg_x2 = x2 + max_width + text_padding
    right_bg_y2 = y1 + total_height

    # Opción 4: Arriba del bbox
    top_bg_x1 = x1
    top_bg_y1 = y1 - total_height - 4
    top_bg_x2 = x1 + max_width + text_padding
    top_bg_y2 = y1 - 4

    # Opción 5: Abajo del bbox
    bottom_bg_x1 = x1
    bottom_bg_y1 = y2 + 4
    bottom_bg_x2 = x1 + max_width + text_padding
    bottom_bg_y2 = y2 + total_height + 4

    # Opción 6: Dentro del bbox (esquina inferior izquierda, último recurso)
    inside_bg_x1 = x1 + 4
    inside_bg_y1 = y2 - total_height - 4
    inside_bg_x2 = x1 + max_width + 12
    inside_bg_y2 = y2 - 4

    # Función para verificar si las coordenadas están dentro del frame
    def is_within_frame(bg_x1, bg_y1, bg_x2, bg_y2):
        return (bg_x1 >= 0 and bg_y1 >= 0 and
                bg_x2 <= frame_width and bg_y2 <= frame_height)

    # Elegir la mejor posición disponible (prioridad: esquina inferior derecha del bbox)
    if inside_br_fits_in_bbox and is_within_frame(inside_br_bg_x1, inside_br_bg_y1, inside_br_bg_x2, inside_br_bg_y2):
        # Primera opción: esquina inferior derecha dentro del bbox
        bg_x1, bg_y1, bg_x2, bg_y2 = inside_br_bg_x1, inside_br_bg_y1, inside_br_bg_x2, inside_br_bg_y2
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15
    elif is_within_frame(left_bg_x1, left_bg_y1, left_bg_x2, left_bg_y2):
        # Segunda opción: izquierda del bbox
        bg_x1, bg_y1, bg_x2, bg_y2 = left_bg_x1, left_bg_y1, left_bg_x2, left_bg_y2
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15
    elif is_within_frame(right_bg_x1, right_bg_y1, right_bg_x2, right_bg_y2):
        # Tercera opción: derecha del bbox
        bg_x1, bg_y1, bg_x2, bg_y2 = right_bg_x1, right_bg_y1, right_bg_x2, right_bg_y2
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15
    elif is_within_frame(top_bg_x1, top_bg_y1, top_bg_x2, top_bg_y2):
        # Cuarta opción: arriba del bbox
        bg_x1, bg_y1, bg_x2, bg_y2 = top_bg_x1, top_bg_y1, top_bg_x2, top_bg_y2
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15
    elif is_within_frame(bottom_bg_x1, bottom_bg_y1, bottom_bg_x2, bottom_bg_y2):
        # Quinta opción: debajo del bbox
        bg_x1, bg_y1, bg_x2, bg_y2 = bottom_bg_x1, bottom_bg_y1, bottom_bg_x2, bottom_bg_y2
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15
    else:
        # Último recurso: dentro del bbox esquina inferior izquierda
        bg_x1, bg_y1, bg_x2, bg_y2 = inside_bg_x1, inside_bg_y1, inside_bg_x2, inside_bg_y2
        # Asegurar que quede dentro del frame
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame_width, bg_x2)
        bg_y2 = min(frame_height, bg_y2)
        text_x = bg_x1 + 4
        text_y_base = bg_y1 + 15

    # Dibujar fondo para todas las etiquetas
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bbox_color, -1)

    # Dibujar cada línea de texto verticalmente
    for idx, line in enumerate(label_lines):
        y_offset = text_y_base + (idx * line_height)
        # Asegurar que el texto no se salga del frame
        if y_offset > 0 and y_offset < frame_height:
            put_text_pil(
                img=frame,
                text=line,
                org=(text_x, y_offset),
                font_size=adaptive_font_size,
                color=(0, 0, 0),
                thickness=adaptive_text_thickness
            )

    return frame


def save_person_crop(image_crop: np.ndarray, output_dir: Path,
                     frame_or_filename: str, person_idx: int,
                     track_id: int | None = None) -> Path:
    """
    Guarda el recorte de una persona detectada.

    Args:
        image_crop: Recorte de imagen BGR
        output_dir: Directorio de salida
        frame_or_filename: Número de frame o nombre de archivo
        person_idx: Índice de la persona en el frame
        track_id: ID de tracking (opcional)

    Returns:
        Ruta del archivo guardado
    """
    crops_dir = output_dir / "person_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if track_id is not None:
        filename = f"{frame_or_filename}_person_{person_idx}_track_{track_id}.jpg"
    else:
        filename = f"{frame_or_filename}_person_{person_idx}.jpg"

    crop_path = crops_dir / filename
    cv2.imwrite(str(crop_path), image_crop)

    return crop_path


def render_person_annotated_crop(
    frame: np.ndarray,
    box,
    *,
    margin_percent: float = 0.30,
    keypoints: np.ndarray | None = None,
    track_id: int | str | None = None,
    beauty_score=None,
    gender_info: dict | None = None,
    age_info: dict | None = None,
    behaviour_info: dict | None = None,
    activity_info: dict | None = None,
    body_display_info: dict | None = None,
    location_info: dict | None = None,
    body_shape_info: dict | None = None,
    accessory_info: dict | None = None,
    social_distance_info: dict | None = None,
    person_color: tuple | None = None,
) -> np.ndarray | None:
    """Renderiza un recorte anotado para una sola persona.

    Toma el frame *limpio* (sin anotaciones de otras personas), recorta una
    región expandida por `margin_percent` alrededor del bbox y dibuja sobre
    ese recorte únicamente el bbox y las etiquetas de la persona indicada.
    El margen permite que las etiquetas posicionadas fuera del bbox queden
    incluidas en el recorte.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Ocupación respecto al FRAME COMPLETO original (no al recorte), para el chip de tamaño.
    _, _occ = bbox_occupancy([x1, y1, x2, y2], w, h)

    mx = max(1, int(bw * margin_percent))
    my = max(1, int(bh * margin_percent))
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2].copy()

    class _LocalBox:
        """Mock de ultralytics Box con coordenadas trasladadas al recorte."""
        def __init__(self, xyxy_local, conf, cls):
            self.xyxy = np.array([xyxy_local], dtype=np.float32)
            self.conf = np.array([conf], dtype=np.float32)
            self.cls = np.array([cls], dtype=np.float32)

    local_box = _LocalBox(
        [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
        float(box.conf[0]),
        int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls),
    )

    if ENABLE_POSE_ESTIMATION and keypoints is not None and len(keypoints) > 0:
        local_kpts = np.array(keypoints, dtype=np.float32, copy=True)
        local_kpts[:, 0] -= cx1
        local_kpts[:, 1] -= cy1
        crop = draw_skeleton(crop, local_kpts)

    crop = draw_detection_with_info(
        crop, local_box,
        track_id=track_id, beauty_score=beauty_score,
        gender_info=gender_info, age_info=age_info,
        behaviour_info=behaviour_info, activity_info=activity_info,
        body_display_info=body_display_info, location_info=location_info,
        body_shape_info=body_shape_info, accessory_info=accessory_info,
        social_distance_info=social_distance_info,
        person_color=person_color,
        occupancy=_occ,
    )
    return crop


def save_person_annotated_crop(annotated_crop: np.ndarray | None,
                                output_dir: Path,
                                frame_or_filename: str,
                                person_idx: int,
                                track_id: int | str | None = None) -> Path | None:
    """Guarda un recorte anotado de una persona en `person_crops_annotated/`.

    Devuelve la ruta o None si el recorte no es válido.
    """
    if annotated_crop is None or annotated_crop.size == 0:
        return None

    crops_dir = output_dir / "person_crops_annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if track_id is not None:
        filename = f"{frame_or_filename}_person_{person_idx}_track_{track_id}_annotated.jpg"
    else:
        filename = f"{frame_or_filename}_person_{person_idx}_annotated.jpg"

    crop_path = crops_dir / filename
    cv2.imwrite(str(crop_path), annotated_crop)
    return crop_path


# ============================================================================
# HELPERS DE ESCENAS (VISUALIZACIÓN)
# ============================================================================

def get_current_scene_number(frame_idx: int, scene_change_frames: list) -> int:
    """
    Determina el número de escena actual basándose en el frame y los cambios de escena.

    Args:
        frame_idx: Número de frame actual (base 0)
        scene_change_frames: Lista de frames donde ocurren cambios de escena

    Returns:
        Número de escena (base 1)
    """
    if not scene_change_frames:
        return 1

    # Asegurar que la lista está ordenada
    sorted_frames = sorted(scene_change_frames)

    # Encontrar en qué escena está el frame actual
    # La escena N comienza en sorted_frames[N-1]
    scene_num = 1
    for i, change_frame in enumerate(sorted_frames):
        if frame_idx >= change_frame:
            scene_num = i + 1

    return scene_num


def draw_scene_number(frame: np.ndarray, scene_number: int, total_scenes: int) -> np.ndarray:
    """
    Dibuja el contador de escena como una píldora redondeada (mismo lenguaje visual que
    los chips): fondo oscuro semi-transparente, esquinas redondeadas, texto claro, en la
    esquina superior derecha. Fallback a cv2 si PIL no está disponible.
    """
    frame_height, frame_width = frame.shape[:2]
    scene_text = f"Scene {scene_number}/{total_scenes}"

    if not PIL_AVAILABLE:
        # Fallback: rectángulo negro + texto blanco (comportamiento previo)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(scene_text, font, 0.8, 2)
        tx, ty = frame_width - tw - 15, 15 + th
        cv2.rectangle(frame, (tx - 8, ty - th - 5), (tx + tw + 8, ty + 5), (0, 0, 0), -1)
        cv2.putText(frame, scene_text, (tx, ty), font, 0.8, (255, 255, 255), 2)
        return frame

    s = _annotation_scale(frame_height)
    font_size = int(round(VISUALIZATION.get("chip_font_size", 15) * s))
    font = _get_pil_font(font_size)
    pad_x = int(round(VISUALIZATION.get("chip_pad_x", 9) * s))
    pad_y = int(round(VISUALIZATION.get("chip_pad_y", 5) * s))
    radius = int(round(VISUALIZATION.get("chip_radius", 9) * s))
    opacity = float(VISUALIZATION.get("chip_opacity", 0.82))
    bg_bgr = VISUALIZATION.get("scene_pill_color", (30, 30, 30))   # gris muy oscuro
    txt_bgr = VISUALIZATION.get("scene_text_color", (255, 255, 255))

    tw, th = _measure_text(scene_text, font)
    pill_w = tw + pad_x * 2
    pill_h = max(th, font_size) + pad_y * 2
    margin = max(10, int(round(frame_width * 0.008)))

    px1 = frame_width - pill_w - margin
    py1 = margin
    px2, py2 = px1 + pill_w, py1 + pill_h
    if px1 < 0 or py2 > frame_height:
        return frame

    layer = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, pill_w, pill_h], radius=radius,
                           fill=(bg_bgr[2], bg_bgr[1], bg_bgr[0], int(255 * opacity)))
    draw.text((pad_x, (pill_h - font_size) // 2), scene_text, font=font,
              fill=(txt_bgr[2], txt_bgr[1], txt_bgr[0], 255))

    layer_arr = np.array(layer)
    alpha = layer_arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = layer_arr[:, :, [2, 1, 0]].astype(np.float32)  # RGBA→BGR
    roi = frame[py1:py2, px1:px2].astype(np.float32)
    frame[py1:py2, px1:px2] = (rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return frame
