"""
Configuración global del proyecto Analisis_imagenes.
Todas las constantes ENABLE_*, rutas de modelos y parámetros de visualización.
"""

import os as _os_pm
import cv2
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN - PARÁMETROS BOOLEANOS
# ============================================================================

# ⚙️ ACTIVAR/DESACTIVAR FUNCIONALIDADES
ENABLE_POSE_ESTIMATION = True       # Estimación de pose y keypoints
ENABLE_TRACKING = True              # Tracking multi-objeto (solo videos)
# Dedup de personas por escena (vídeo): fusiona fragmentos del tracker + det_* por-frame
# que son la MISMA persona (embedding ResNet18 + guard de co-ocurrencia). Corrige el
# sobreconteo (misma persona contada N veces en una escena). Ver processing/person_dedup.py.
ENABLE_PERSON_DEDUP = True
PERSON_DEDUP_SIM_THRESHOLD = 0.93   # coseno mínimo de apariencia para fusionar dos track_id.
#   CALIBRADO (2026-07-03, validacion_imagenes/dedup_calibrar.py sobre 8 vídeos, 3466 pares
#   CO-OCURRENTES = personas distintas garantizadas): p99 de "personas distintas" = 0.936 → 0.93
#   deja <~1.5% de falsos merges de personas distintas (y el guard de co-ocurrencia bloquea las
#   simultáneas). El 0.86 inicial estaba por DEBAJO del p95 (0.866) → habría fusionado distintas.
#   Más alto = más conservador (fusiona menos, prioriza no crear "quimeras" sobre no dejar duplicados).
ENABLE_PERSON_ID_LABEL = True      # Mostrar el ID de cada persona en imágenes/videos anotados
ENABLE_BEAUTY_ESTIMATION = True     # Belleza INLINE (chips de vídeo), ACTIVA por defecto (2026-07-03):
#   usa el base COMPARTIDO Qwen3.5-9B vía generate_beauty() sin cargar 2º modelo. Nota: en el path de
#   IMAGEN esto corre belleza inline para TODAS las personas, además del pase diferido demand/*
#   (ENABLE_BEAUTY_PASS) → cómputo duplicado en imágenes; ponerla en False si se procesa un lote de imágenes.
ENABLE_BEAUTY_INLINE_IMAGES = False # 2026-07-07: la inline queda SOLO para vídeo (chips). En IMÁGENES la
#   belleza la da únicamente el pase DIFERIDO (crop de cara, 5 keypoints faciales, LoRA checkpoint-1400,
#   solo demand/*) → elimina la doble puntuación por imagen. True = comportamiento previo (inline también
#   en imágenes, además del pase).
BEAUTY_MIN_FACE_KEYPOINTS = 4       # Belleza (vídeo E IMÁGENES): mín. keypoints faciales visibles (de 5)
#   para puntuar una cara. 2026-07-21: subido 3→4 y aplicado en TODOS los paths (antes imágenes exigían 5
#   vía has_five_face_keypoints_visible; vídeo usaba 3). La belleza ya NO se limita a personas demand/* —
#   se puntúa toda cara con ≥ este nº de keypoints (ver _collect_beauty_pending / score_demand_beauty).
#   En vídeo se elige por escena el frame con MÁS keypoints faciales y, a igualdad, el MÁS NÍTIDO.
# Selección del frame de belleza por escena: frontalidad (nº de keypoints faciales) PRIMARIA, nitidez
# (var. del Laplaciano) como desempate fuerte entre frames de igual frontalidad → evita elegir un frame
# con desenfoque de movimiento (los keypoints se detectan con alta confianza aunque la cara esté movida,
# así que la confianza no discriminaba el blur). sel_score = n_face + sharp/(sharp + K).
BEAUTY_SHARP_TIEBREAK_K = 150.0     # K del squash de nitidez al desempate [0,1). Mayor K = nitidez pesa menos.
BEAUTY_MIN_SHARPNESS = 0.0          # 2026-07-21: bajado 60→0 (decisión del usuario). Descartaba el crop de
#   cara si var(Laplaciano) < umbral. 60 estaba calibrado sobre vídeo NATIVO (720p, crops nítidos 60–398) pero
#   en vídeo AI-UPSCALE el alisado deja TODOS los crops <20 → 60 descartaba el 100% (p.ej. Friends 4K). Con 0
#   no se filtra por nitidez → el AI-upscale produce belleza. ⚠️ CONTRAPARTIDA: en vídeo nativo ya NO se filtran
#   caras con desenfoque de movimiento (antes 60 las tiraba). Ver "Selección de frame de belleza por nitidez".
BEAUTY_MIN_HEAD_PX = 256            # 2026-07-21: mín. de min(alto,ancho) del recorte de CABEZA (extract_face_crop)
#   para puntuar belleza; por debajo se DESCARTA (no se puntúa), en vídeo E imágenes. Ancla el tamaño de cara
#   mínimo al de los datasets de entrenamiento (SCUT-FBP5500 / CFD / MEBeauty): el estimador preprocesa con
#   thumbnail(672) SIN ampliar, así que una cara diminuta (p.ej. ~49px de IG) se queda diminuta y ancla la nota
#   bajo (fuera de distribución). Valor = p5 del tamaño de cabeza percibido en esos datasets (mismo pipeline:
#   thumbnail 672 + YOLO26-pose + extract_face_crop), medido sobre 891 caras: SCUT p5=241 / CFD p5=324 /
#   MEBeauty p5=257 → p5 COMBINADO=256 (mediana 321). ⚠️ Descarta la mayoría de caras pequeñas de IG.
ENABLE_GENDER_CLASSIFICATION = True # Clasificación de género
ENABLE_AGE_CLASSIFICATION = True    # Clasificación de edad
ENABLE_BEHAVIOUR_CLASSIFICATION = True # Clasificación de comportamiento (Demand/Offer con detalles)
ENABLE_ACTIVITY_CLASSIFICATION = True # Clasificación de actividades (entertaining, sports, romance, etc.)
ENABLE_BODY_DISPLAY_CLASSIFICATION = True # Clasificación de exposición del cuerpo
ENABLE_LOCATION_CLASSIFICATION = True # Clasificación de ubicación (indoors, wilderness, city)
ENABLE_BODY_SHAPE_CLASSIFICATION = True # Bloque body_shape → ahora transporta SOLO PESO + MUSCULATURA
#   (la SILUETA se ELIMINÓ del repo el 2026-07-04: κ 0.306 en el sample 500 real, no se recuperó).
#   Mantener True: peso/musculatura viajan aquí. El nombre "body_shape" se conserva por el esquema.
ENABLE_ACCESSORY_CLASSIFICATION = True # Clasificación de accesorios visibles (makeup, tattoos, bags, belts, jewelry, headwear, eyewear)
ENABLE_OCR = False                   # Reconocimiento de texto (OCR) en cambios de escena
ENABLE_SOCIAL_DISTANCE = True       # Clasificación de distancia social (proxémica)
USE_LEGACY_SOCIAL_DISTANCE = True   # True (activo): módulo standalone con SOCIAL_DISTANCE_PROMPT sobre per-person crop, lowest-visible-part.
                                    # False: cuando el gate de keypoints no trigea, la distancia social se delega al SCENE_PROMPT con frame completo.
                                    # 2026-05-24 — rollback al baseline tras iter5/iter6 que dieron resultados desastrosos.
                                    # NOTA (2026-06-29): con SOCIAL_DISTANCE_KEYPOINT_ONLY=True el gate clasifica las
                                    # 6 categorías de forma determinista y NUNCA delega al VLM (este flag solo aplica
                                    # en el fallback raro de "sin keypoints").

# ⚙️ DISTANCIA SOCIAL 100% DETERMINISTA POR KEYPOINTS (2026-06-29)
# Reemplaza el fallback al VLM por el mapeo "parte COCO visible más baja → categoría"
# (espeja _map_parts_to_category pero con visibilidad de keypoints de YOLO26-pose).
# Validado sobre el banco FLUX social_distance: 31/36 = 86% crudo, y los 5 fallos son
# imágenes FLUX cuyo encuadre no coincide con su etiqueta (mal generadas) → ~36/36 en
# imágenes válidas. Determinista, sin VLM, sin coste extra (reusa la pose ya calculada).
SOCIAL_DISTANCE_KEYPOINT_ONLY = True
SOCIAL_DISTANCE_PART_CONF = 0.5     # conf mínima para considerar "visible" una parte (hombros/caderas/piernas).
                                    # En validación 0.3/0.5/0.65 dieron lo mismo (la frontera la fija qué partes
                                    # se detectan, no su conf). 0.5 = el umbral de visibilidad del pipeline.
SOCIAL_DISTANCE_FILL_RATIO = 0.375  # bbox_area/frame_area que separa close social (bbox llena) de far social
                                    # (bbox pequeña). ⚠️ calibrado sobre n=12 del banco; re-validar con más datos.

# ⚙️ USAR MODELO FINETUNEADO CON LORA
USE_FINETUNED_MODEL = False         # Usar modelo Qwen finetuneado con adaptadores LoRA (LoRA era para Qwen3-VL-4B; con Qwen3.5-9B no aplica)

# ⚙️ USAR FUENTE LIBERATION SANS
USE_LIBERATION_SANS = True          # Usar fuente TrueType (PIL) para el texto en visualizaciones (requiere PIL/Pillow)

# ⚙️ SALIDAS ANOTADAS A DISCO (imágenes)
SAVE_ANNOTATED_OUTPUTS = True       # Escribir {stem}_annotated.jpg + person_crops/ + person_crops_annotated/.
                                    #   El modo `carpeta` lo fuerza a False (no persiste imágenes: escribirlas
                                    #   y borrarlas era trabajo perdido). No afecta a la clasificación ni al
                                    #   pase de belleza (re-lee la imagen ORIGINAL, no los crops).

# ⚙️ VÍDEOS LARGOS AL FINAL (modo `carpeta`)
CARPETA_LONG_VIDEO_THRESHOLD_S = 60  # Vídeos con duración ≥ este umbral (s) se procesan AL FINAL,
                                     #   ordenados de menor a mayor duración (el peor queda el último).
                                     #   Examen 2026-07-11 (36.353 mp4): mediana 21s, p95 104s, máx 106min;
                                     #   ≥60s = 17,6% de los vídeos pero 54,2% del tiempo total. Duraciones
                                     #   desde validacion_imagenes/duraciones_videos.csv (caché; lo que falte
                                     #   se sondea con cv2, solo cabecera). 0/None = desactivado (orden actual).

# ⚙️ IMAGEN ANOTADA POR PERSONA
ENABLE_PER_PERSON_ANNOTATED_IMAGE = True  # Guardar un recorte anotado por cada persona detectada
                                          #   (imágenes y videos). Carpeta: <output>/person_crops_annotated/.
                                          #   Permite que cada fila del Excel comparativo muestre su propio recorte.
PER_PERSON_CROP_MARGIN = 0.30             # Margen relativo (fracción del bbox) para incluir las
                                          #   etiquetas que se dibujan fuera del bbox

# ⚙️ DETECCIÓN DE COLLAGES/COMPOSICIONES
ENABLE_COLLAGE_DETECTION = False     # Detectar y procesar imágenes compuestas por múltiples paneles
COLLAGE_MIN_PANELS = 1              # Mínimo de paneles para considerar un collage
COLLAGE_MIN_PANEL_SIZE_PERCENT = 8  # Tamaño mínimo de panel (% del área total)
COLLAGE_HOUGH_THRESHOLD = 80        # Umbral para detección de líneas con HoughLinesP (reducido)
COLLAGE_HOUGH_MIN_LINE_LENGTH = 150 # Longitud mínima de línea divisoria (reducido)
COLLAGE_HOUGH_MAX_LINE_GAP = 50     # Máxima brecha en líneas divisorias

# ============================================================================
# CONFIGURACIÓN DE BATCH PROCESSING VLM - OPTIMIZACIÓN DE RENDIMIENTO
# ============================================================================
# El VLM (Visual Language Model) es el componente más lento. Estas configuraciones
# permiten balancear entre velocidad y cobertura de análisis.

# ============================================================================
# CONFIGURACIÓN FRAME POR FRAME (SIN OPTIMIZACIONES BATCH)
# ============================================================================

# Procesamiento frame por frame sin optimizaciones
FRAME_BY_FRAME_PROCESSING = True    # Procesar cada frame individualmente

# ============================================================================
# CONFIGURACIÓN COMPORTAMIENTO EN VIDEOS
# ============================================================================

# Configuración para análisis de comportamiento en vídeos
ENABLE_SCENE_DETECTION = True          # Usar PySceneDetect para cambios de escena
SCENE_MIN_LEN = 5                      # Longitud mínima de escena en frames (detect-content)
SCENE_CONTENT_THRESHOLD = 27.0         # Umbral de cambio de contenido (detect-content --threshold)

# ⚙️ DETECCIÓN DE CAMBIOS DE ESCENA POR KEYPOINTS FACIALES
ENABLE_FACE_KEYPOINT_SCENE_DETECTION = False  # Detectar sub-escenas por DESPLAZAMIENTO del centroide facial
FACE_KEYPOINT_DISPLACEMENT_THRESHOLD = 0.20  # Desplazamiento normalizado máximo permitido por frame
                                              # (fracción del ancho del bounding box de la persona)
FACE_KEYPOINT_MIN_SCENE_LEN = 10             # Mínimo de frames entre cambios de escena por keypoints

# Sub-escena cuando el Nº de keypoints faciales visibles cambia (la persona gira la cara:
# frontal↔perfil/de espaldas). Motivación: al girar la cara suele cambiar el comportamiento
# (mirada/Demand-Offer), así que se re-clasifica en una sub-escena nueva. Barato (reusa la pose ya
# calculada). Aumenta el nº de llamadas VLM por vídeo (una por sub-escena) → más lento pero más fiel.
ENABLE_FACE_KEYPOINT_COUNT_SCENE_DETECTION = True  # ← petición 2026-07-03
FACE_KEYPOINT_COUNT_FRONTAL_MIN = 4          # ≥ este nº de kpts faciales (de 5) = cara frontal
FACE_KEYPOINT_COUNT_PROFILE_MAX = 2          # ≤ este nº = perfil / de espaldas (corte al cruzar entre ambos)

# ⚙️ LÍMITE DE MEMORIA PARA ESCENAS LARGAS
MAX_SCENE_FRAMES_IN_MEMORY = 4000     # Frames máximos bufferizados en RAM por escena.
                                       # Escenas más largas usan modo streaming (pre-scan sin
                                       # guardar frames completos + seek-back para anotación).
                                       # Un frame 1080p ≈ 6 MB; 4000 frames ≈ 24 GB de RAM.
                                       # Ajustar según la RAM disponible del sistema.

# Nota: Con ENABLE_SCENE_DETECTION=True, se detectan cambios de escena y para cada
# escena se pre-escanean todos los frames para encontrar el mejor frame por persona
# por función VLM. Cada función VLM se ejecuta UNA SOLA VEZ por persona por escena
# usando el frame que cumple mejor sus condiciones específicas:
#   - Beauty: frame con 5 keypoints faciales visibles (mayor confianza facial)
#   - Body shape: frame con pose frontal y cintura visible
#   - General (behaviour, activity, body_display, location, accessory):
#     frame con mayor (confianza × área del bbox)

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

# Directorios
OUTPUT_DIR = Path("/home/emilio/Documentos/#masculinity_IG/LLMs/Participation/participation_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modelos YOLO26
DETECTION_MODEL = "yolo26x.pt"      # Detección de personas (solo se carga si USE_POSE_AS_DETECTOR=False)
POSE_MODEL = "yolo26x-pose.pt"      # Pose estimation

# ⚙️ POSE-ÚNICO (2026-06-29): usar SOLO yolo26x-pose como detector + pose.
# Validado sobre el banco FLUX social_distance: pose-único @640 iguala al dual
# (35/36 idénticas; la única diferencia era una caja espuria del detector
# redundante) y elimina el riesgo de mal-asignación de keypoints por índice del
# detector. Con True, loader.py carga 1 solo modelo (model_detect = model_pose)
# y annotate_frame empareja keypoints a cada caja por IoU (no por índice).
# Rollback al dual = False (recarga yolo26x.pt + indexado por índice legacy).
USE_POSE_AS_DETECTOR = True
YOLO_POSE_IMGSZ = 640               # imgsz nativo de YOLO26-pose; subirlo (1280) DEGRADA
                                    # (conf kpt 0.96→0.77, menos personas) — no tocar.

# Dispositivo para los modelos YOLO. "cuda"/"0" usa GPU; "cpu" libera VRAM
# para el VLM. En GPUs ≤12 GB el VLM (gemma-4-E2B-it BF16 ~9.6 GB) deja
# poco margen → YOLO26x (57M params) provoca OOM al primer batch.
# Cambio para mantener la GPU dedicada al VLM. Coste empírico: ~1-2 s por
# imagen extra para detección+pose (vs ~50 ms en GPU).
YOLO_DEVICE = "cuda"

# Modelo finetuneado con LoRA para todas las tareas VLM (comportamiento, actividades, belleza, etc.)
FINETUNED_LORA_PATH = "/home/emilio/Documentos/#masculinity_IG/LLMs/FT-Qwen4b-IT/outputs/final_model"

# Modelo VLM principal usado por TODOS los clasificadores (Merge A + Merge B +
# social_distance + individuales). Cambiar con cuidado: las categorías y
# prompts están ajustados al estilo de razonamiento de este modelo.
#   - "google/gemma-4-E4B-it"  → 4.5B effective, multimodal nativo, Apache 2.0.
#                                ~3 GB con NF4 en GPU; visual token budget
#                                configurable vía GEMMA_VISUAL_TOKENS.
#   - "Qwen/Qwen3.5-9B"        → modelo anterior, fallback histórico.
BEHAVIOUR_MODEL_NAME = "Qwen/Qwen3.5-9B"   # PROD 2026-07-02: clasificador general (NF4, attn eager) vía Qwen35VLMBackend; requiere gcc-15 en PATH. Rollback → "google/gemma-4-E4B-it"

# ============================================================================
# BELLEZA FACIAL — gemma4 finetuneado (Transformers + PEFT 4-bit, pase separado)
# ============================================================================
# Estimador de belleza 1-5 con gemma-4-E4B + adapter LoRA (visión+texto) entrenado
# sobre SCUT-FBP5500. Se sirve por Transformers/PEFT, NO por Ollama: Ollama no puede
# cuantizar gemma4 a Q4 en CUDA/Linux (requiere MLX) y llama.cpp no soporta gemma4.
# El modelo en 4-bit cabe en 12 GB y se ejecuta como PASE SEPARADO (beauty_pass.py),
# nunca en el mismo proceso que el backend Ollama de los otros 15 clasificadores.
# Solo se activa cuando ENABLE_BEAUTY_ESTIMATION=True (beauty_pass lo fuerza en runtime).
BEAUTY_USE_TRANSFORMERS_BACKEND = True
BEAUTY_MODEL_NAME = "google/gemma-4-E4B-it"
BEAUTY_LORA_PATH = ("/home/emilio/Documentos/#masculinity_IG/LLMs/pruebas de FT/"
                    "FT-Belleza_1-5/ft_outputs/gemma4_beauty_v3/final_model")
# El prompt DEBE ser idéntico al de entrenamiento (preprocess_en_1_5.BEAUTY_PROMPT_EN).
BEAUTY_PROMPT = ("Rate the facial attractiveness of this face on a scale of 1 to 5, "
                 "where 1 is least attractive and 5 is most attractive. "
                 "Respond with only the number.")
BEAUTY_MAX_NEW_TOKENS = 8        # solo el número; no malgastar generación
BEAUTY_SCALE = (1.0, 5.0)        # escala nativa SCUT (1-5), NO 0-100

# --- Belleza vía Ollama (Qwen3-VL-4B finetuneado, GGUF Q4_K_M) — PRODUCCIÓN ---
# Camino ganador (ver CLAUDE.md): Qwen3-VL-4B ≈0.92 Pearson en Transformers; servido en
# Ollama (GGUF Q4 via llama.cpp + mmproj) rinde ≈0.857 (el gap 0.92→0.857 es del path de
# preprocesado de visión de llama.cpp, NO de la cuantización: Q8 dio lo mismo, 0.850).
# Pase DIFERIDO: se sirve solo cuando gemma4 está parado (un único swap de Ollama).
# Selector de backend del estimador de belleza:
#   "ollama"       → OllamaBeautyBackend (Qwen3.5-9B deciles 1-10 @ :11435, Ollama)
#   "qwen35_cont"  → Qwen35BeautyBackend (Qwen3.5-9B CONTINUO 1-10, Transformers, checkpoint-1400) ← PRODUCCIÓN
#   "transformers" → Gemma4BeautyBackend (gemma4 1-5, legacy/dormant) — usa BEAUTY_USE_TRANSFORMERS_BACKEND
BEAUTY_BACKEND = "qwen35_cont"    # 2026-06-26: salida CONTINUA (decimal) para análisis. Rollback → "ollama"
# PRODUCCIÓN 2026-06-23: Qwen3.5-9B finetuneado en 5 bases (SCUT+CFD+MEBeauty+HotOrNot+M2B,
# checkpoint-1400) — sustituye a beauty-qwen3vl:q4. Mejor cobertura multi-dominio; en
# Transformers daba SCUT 0.925/CFD 0.819/MEBeauty 0.814. Mismo host :11435, mismo prompt EN 1-10.
# GGUF texto+mmproj de llama.cpp (convert_hf_to_gguf --no-mtp, IMPRESCINDIBLE: sin el flag,
# block_count=33 cuenta la capa MTP sin emitir sus tensores → Ollama crashea con
# 'missing tensor blk.32.attn_norm.weight'). Empaquetado: merge_5db_for_ollama.py + Modelfile.qwen35.
# Q8 sobre Q4: SCUT 0.847 vs 0.842, CFD 0.658 vs 0.610, MEBeauty 0.742 vs 0.687 → Q8 mejor (beauty
# pass diferido = VRAM libre, no compite). ⚠️ HALLAZGO 2026-06-23: el path de visión de qwen3vl está
# ROTO en Ollama 0.30.7 (el qwen3-vl:8b OFICIAL y el beauty-qwen3vl:q4 viejo devuelven '' con imagen);
# qwen35 SÍ funciona. Por eso producción = qwen35, NO qwen3vl. requires think=False (ver backend).
BEAUTY_OLLAMA_TAG = "beauty-qwen35-5db:q8"
# Setup dual de Ollama (2026-06-13): el pipeline corre en Ollama 0.23.2 (motor Go, κ 0.90)
# en :11434, pero beauty-qwen3vl REQUIERE Ollama ≥0.30 → segunda instancia (0.30.7) en :11435.
# El OllamaBeautyBackend usa este host (no el :11434 del pipeline). Ver CLAUDE.md "setup dual".
BEAUTY_OLLAMA_HOST = "http://localhost:11435"
# Prompt EN 1-10 (parity 0.857 vs continuo; el ES de entrenamiento dio 0.845 → EN gana).
BEAUTY_PROMPT_QWEN = ("Rate the beauty of this person on a scale from 1 to 10. "
                      "Respond with only the number.")
BEAUTY_SCALE_QWEN = (1.0, 10.0)  # escala nativa del adapter Qwen, SIN reescalar

# --- Belleza CONTINUA vía Transformers (Qwen3.5-9B 4-bit + adapter checkpoint-1400) — PRODUCCIÓN 2026-06-26 ---
# Modelo del experimento de target CONTINUO (rango-percentil 1-10, un decimal): iguala al keeper de
# DECILES en Pearson (SCUT 0.934 / CFD 0.806 / MEBeauty 0.807 / HotOrNot 0.606 / M2B 0.524, epoch 2.04)
# pero da MEJOR MAE + salida decimal → más útil para el análisis estadístico downstream.
# Se sirve por Transformers/PEFT, NO por Ollama: el path de visión de llama.cpp degrada el Pearson
# ~0.93→0.85. Pase DIFERIDO (gemma4 parado → VRAM libre, cabe el 9B 4-bit en la RTX 4070). Mismo prompt
# EN 1-10 y escala (1,10) que el backend Ollama (BEAUTY_PROMPT_QWEN / BEAUTY_SCALE_QWEN). Ver CLAUDE.md.
BEAUTY_QWEN35_BASE = "Qwen/Qwen3.5-9B"
# LoRA adapter dir. Defaults to the copy shipped in the repo (Git LFS); override
# with the BEAUTY_ADAPTER_PATH env var to point elsewhere.
BEAUTY_QWEN35_ADAPTER_PATH = _os_pm.environ.get(
    "BEAUTY_ADAPTER_PATH", str(Path(__file__).parent / "models" / "beauty_adapter"))

# Pase de belleza DIFERIDO integrado en main.py (single/batch/manual): tras clasificar
# todas las imágenes con gemma4, se hace `ollama stop` (libera VRAM en :11434), se carga
# Qwen3-VL en :11435 y se puntúa belleza SOLO de rostros demand/* (IoU≥0.5 con un bbox
# behaviour=demand de la fase 1). Distinto de ENABLE_BEAUTY_ESTIMATION (estimador INLINE
# legacy, que cargaría belleza junto a gemma4 y no cabe en 12 GB → sigue en False).
# Opt-out puntual con --no-beauty. Degrada con elegancia si :11435 no responde.
ENABLE_BEAUTY_PASS = True  # 2026-07-03: RE-ACTIVADO gracias al BASE COMPARTIDO (abajo).
# Antes False porque en el modo Transformers el clasificador NF4 se queda en VRAM y NO cabía
# una 2ª carga del modelo de belleza. Ahora el clasificador (Qwen3.5-9B) y el estimador de
# belleza (mismo base + LoRA checkpoint-1400) comparten UNA sola instancia en VRAM togglando
# el adapter (BEAUTY_SHARE_CLASSIFIER_BASE) → el pase diferido reutiliza el modelo ya cargado,
# sin 2ª copia y sin `ollama stop`. Opt-out puntual con --no-beauty. Rollback = False.

# Base COMPARTIDO clasificador↔belleza (2026-07-03): cuando el clasificador es Qwen3.5-9B
# (BEHAVIOUR_MODEL_NAME) y el backend de belleza es "qwen35_cont" (mismo base), el
# Qwen35VLMBackend carga el LoRA de belleza sobre la base y sirve AMBAS tareas con una sola
# instancia en VRAM (~7.9 GB): clasificación con `disable_adapter()` (salida IDÉNTICA al base
# puro, verificado byte a byte) y belleza con el adapter activo. Elimina la doble carga del
# pase diferido. Ver memoria beauty-classifier-shared-base-toggle. Desactivar → cada tarea
# carga su propia copia (comportamiento previo; NO cabe con el pase diferido activo).
BEAUTY_SHARE_CLASSIFIER_BASE = True

# ============================================================================
# CONFIGURACIÓN DEL BACKEND VLM
# ============================================================================
# Selecciona qué motor de inferencia usar para todos los clasificadores VLM.
#
# "transformers" — carga el modelo localmente con HuggingFace Transformers.
#                  Usa BEHAVIOUR_MODEL_NAME como modelo y aplica LoRA si
#                  USE_FINETUNED_MODEL=True. Requiere ~8 GB de VRAM.
#
# "vllm"         — motor de inferencia de alto rendimiento (FlashAttention 2,
#                  PagedAttention, continuous batching). 3–10× más rápido que
#                  "transformers" en GPU. Soporta LoRA en caliente vía
#                  LoRARequest (sin fusionar pesos). Requiere GPU NVIDIA y
#                  ~10 GB de VRAM. Instalar con: pip install 'vllm>=0.7.0'
#
# "ollama"       — delega la inferencia a un servidor Ollama local.
#                  Requiere: ollama serve + ollama pull <OLLAMA_MODEL>
#                  Instalar cliente: pip install ollama
#                  Modelos compatibles: llava, llama3.2-vision, minicpm-v,
#                  moondream, bakllava, etc.
#
VLM_BACKEND = "transformers"   # "transformers" | "vllm" | "ollama" | "llamacpp"
# ⚠️ PRODUCCIÓN 2026-06-29: cambiado a "transformers" (NF4) + prompts JSON refinados
# (prompts_gemma4_json) + híbrido de comportamiento (gate de pose). Validado SOLO sobre
# el banco sintético FLUX edad_genero_cat (no sobre el sample 500 real). Rollback al
# baseline Ollama 0.23.2 (motor Go, κ 0.90 en sample 500 real) = restaurar
# config.before_json_hybrid_prod_20260629_190500.py. Ver CLAUDE.md "Config JSON+híbrido".
# Ollama ≥0.30 degrada gemma4 a κ 0.70 — ver "NO actualizar Ollama más allá de 0.23.2".
# "llamacpp" (llama-server hi-res) y "transformers" (NF4) se probaron y NO recuperan
# el 0.90 (0.735 / 0.721). models/backends/llamacpp_backend.py queda como referencia.

# Servidor llama-server (solo si VLM_BACKEND="llamacpp").
LLAMACPP_HOST  = "http://127.0.0.1:8080"
LLAMACPP_HOSTS = ["http://127.0.0.1:8080"]   # round-robin si hay varios
LLAMACPP_TIMEOUT = 300

# ============================================================================
# CUANTIZACIÓN DEL VLM (sólo para backend "transformers")
# ============================================================================
# "none" — carga en BF16/FP16 nativo (~14 GB para Gemma-4-E4B). No cabe en 12 GB.
# "int8" — bitsandbytes 8-bit (~10-11 GB con Gemma-4-E4B). Más calidad que nf4
#          en las capas LM, deja ~1-2 GB para activaciones/KV cache.
# "nf4"  — bitsandbytes 4-bit nf4 + double quant + bf16 compute (~9 GB con
#          Gemma-4-E4B: vision tower no se cuantiza). Más compresión.
#          Riesgo: en algunos prompts complejos (Merge A) Gemma-4 con NF4
#          contesta "na" a todo y dice "image is gray" — al parecer NF4
#          degrada el vision tower de este modelo.
VLM_QUANTIZATION = "nf4"   # PROD 2026-06-29 (transformers): NF4 (vision tower bf16); "none" para Ollama

# Implementación de atención del backend Transformers.
#   "sdpa"  → PyTorch scaled_dot_product_attention (defecto seguro, funciona).
#   "flash_attention_2" → requiere flash-attn; MÁS RÁPIDO en el prefill... PERO
#       ⚠️ gemma-4-E4B NO es compatible: su atención de TEXTO (head_dim=256)
#       dispara "FlashAttention forward only supports head dimension at most 256"
#       en runtime y TODA clasificación cae a "no visible" (probado 2026-07-01,
#       incluso con dict text=fa2/vision=sdpa). flash-attn quedó instalado pero
#       INERTE por este flag. No poner "flash_attention_2" con gemma-4.
#   "auto"  → autodetecta flash-attn si está instalado (comportamiento legacy;
#       peligroso ahora que flash-attn ESTÁ instalado → rompería gemma-4).
VLM_ATTN_IMPLEMENTATION = "sdpa"

# Tope de VRAM en GiB para el VLM cuando se usa `device_map="auto"` (no
# cuantizado). Sin este límite, transformers carga todo el modelo en GPU y
# luego OOMs en las activaciones de cada forward pass. 9 GiB deja ~3 GiB
# libres para KV cache + activaciones en una RTX 4070 de 12 GiB. Los 0.5-1
# GiB restantes del modelo se offloadean a CPU (con coste de latencia).
VLM_GPU_MAX_MEMORY_GIB = 11

# Si True, fuerza `device_map={"": 0}` (TODO el VLM en GPU 0), saltándose el
# `max_memory` y el offload parcial a CPU. Requiere que el modelo entero
# quepa en VRAM dejando margen para activaciones (~1.5-2 GiB). En RTX 4070
# de 12 GiB con Gemma-4-E2B-it (BF16 ≈ 9.6 GiB) cabe sólo si el pipeline
# usa una única invocación VLM por imagen — con Merge A → Merge B →
# social_distance encadenados, el KV cache + activaciones acumuladas
# desbordan los 12 GiB. Mantener False salvo en pruebas aisladas.
VLM_FORCE_GPU = False

# ============================================================================
# MAX TOKENS POR LLAMADA VLM
# ============================================================================
# Tras fusionar prompts (Merge A: 6 atributos por persona; Merge B: 2 atributos
# de escena), las respuestas son más largas. Se elevan los tokens de salida
# para evitar truncamientos.
# Nota: Qwen3.5 hace chain-of-thought por defecto antes de emitir el formato
# estructurado, así que necesita un budget grande para llegar al bloque de
# veredictos. 2000 / 1000 son seguros con cuantización nf4 y dejan ~50% margen.
# Límite de tokens generados por llamada VLM. Único parámetro para TODOS los
# clasificadores (Merge A, Merge B, social_distance, individuales). Subir si el
# modelo (con OLLAMA_THINK=True) trunca la respuesta antes del bloque de
# veredictos; bajar para inferencia más rápida cuando el modelo no razona.
# - Con VLM_BACKEND="ollama" + Qwen3.5 / gemma4 thinking: 4000 (calibrado
#   para sacar métricas válidas en Merge A).
# - Con VLM_BACKEND="transformers" + Gemma-4-E4B en GPU de 12 GB el KV cache
#   crece con max_new_tokens; bajar a 1024 si aparecen OOMs (ver
#   TransformersBackend.generate_batch).
VLM_MAX_TOKENS = 1024   # 2026-07-01: duplicado 512→1024 para que el JSON de merge-A (arrays "features" verbosos) NO se trunque en ~4% de imágenes; el flatten tolerante recupera lo truncado, esto lo evita de raíz. Vigilar VRAM (era 512; 2500 OOMea)

# ============================================================================
# CHAIN-OF-THOUGHT (thinking mode)
# ============================================================================
# Tanto OllamaBackend como TransformersBackend leen esta constante. En
# Transformers/Gemma-4 se traduce a `enable_thinking=True` en
# `processor.apply_chat_template()`. En Ollama se traduce a `think=True`
# (Qwen) o al token "<|think|>" inyectado en el system prompt (Gemma).
VLM_ENABLE_THINKING = False   # PROD 2026-06-29: prompts_gemma4_json son "no-reasoning"; era True para Ollama+CoT

# ============================================================================
# Módulo de prompts a importar (los 3 classifiers Merge A / Merge B / social
# distance hacen `importlib.import_module(f"models.{config.VLM_PROMPT_MODULE}")`)
# ============================================================================
#   "prompts_it"            → prompts adaptados a IT sin CoT
#   "prompts_ollama_cot"    → prompts CoT del baseline gemma4 (E4B+thinking)
#   "prompts_ollama_qwen35" → prompts para Qwen3.5-9B (thinking nativo)
#   "prompts_gemma4_json"   → prompts JSON refinados no-reasoning (PRODUCCIÓN 2026-06-29)
import os as _os_pm   # override por entorno para experimentos (p.ej. prompts_qwen3)
VLM_PROMPT_MODULE = _os_pm.environ.get("VLM_PROMPT_MODULE") or "prompts_qwen3"   # PROD 2026-07-02 (Qwen3.5): fork de prompts_gemma4_json con submission→gate + sports-por-acción; requiere el proxy JsonFlatteningBackend (loader). Rollback: "prompts_gemma4_json" / "prompts_ollama_cot"

# ============================================================================
# Transformers (Gemma-4 NF4): presupuesto de tokens de imagen y gate de pose
# para demand/submission (híbrido comportamiento). PRODUCCIÓN 2026-06-29.
# ============================================================================
# Override escalar de max_soft_tokens (todas las tareas) — 560 cabe en 12 GB con
# NF4; el default GEMMA_VISUAL_TOKENS (1120 person_attrs) OOMea. Solo afecta a
# VLM_BACKEND="transformers"; Ollama lo ignora.
VLM_MAX_SOFT_TOKENS = 560

# Gate de pose para demand/submission: gemma4 es CIEGO a la elevación de cámara
# (toda pista de ángulo satura), pero el picado SÍ está en la geometría de los
# keypoints. Si el VLM dice demand/affiliation y los hombros están escorzados
# hacia la cámara (shoulders_below_eyes < umbral), se reescribe a demand/submission.
# Validado en el banco FLUX comportamiento: submission 0/6 → 6/6, global 50% → 95.8%.
# ⚠️ Umbral calibrado con n=12 (gap submission≤2.81 / affiliation≥2.96). Ver CLAUDE.md
# y [[behaviour-submission-pose-gate]].
ENABLE_SUBMISSION_POSE_GATE = True
SUBMISSION_SBE_THRESHOLD = 2.88   # shoulders_below_eyes < umbral → demand/submission
SUBMISSION_SBE_PART_CONF = 0.3    # conf mínima de ojos/hombros para fiar el cálculo

# ============================================================================
# GEMMA-4 — Visual token budget por tarea (solo backend "transformers")
# ============================================================================
# Gemma-4 acepta image_soft_token_budget ∈ {70, 140, 280, 560, 1120}. Más
# tokens = más detalle visual = más latencia. Valores bajos (140-280)
# producen "image is gray, no discernible person" para crops humanos
# realistas — el modelo no recibe suficientes tokens visuales para discernir
# la persona dentro del crop. Subido al máximo para análisis fino de personas.
GEMMA_VISUAL_TOKENS = {
    "person_attrs":    1120,  # Merge A: 6 atributos requieren ver cara, ropa, accesorios
    "scene":            560,  # Merge B: activity/location (escena general)
    "social_distance":  560,  # framing entre personas
    "ocr":             1120,  # OCR requiere máximo detalle
    "beauty":          1120,  # cara legible
    "default":          560,
}

# Tamaño máximo de batch dentro de TransformersBackend.generate_batch().
# Con NF4 + Gemma-4-E4B el modelo ocupa ~9 GB en VRAM (vision tower y
# multimodal projections no se cuantizan). En una GPU de 12 GB quedan ~2 GB
# libres para activaciones, suficiente para 2-3 crops por forward pass.
# Subir a 4-6 sólo si la GPU tiene ≥16 GB; bajar a 1 desactiva el batching
# pero conserva la interfaz (útil para debug o GPUs muy pequeñas).
VLM_BATCH_SIZE = 1

# ============================================================================
# Configuración específica de Ollama (solo se usa si VLM_BACKEND="ollama")
# ============================================================================
# Perfiles de modelos Ollama probados. Para añadir uno nuevo: pull el tag y
# añade una entrada con el método de thinking que usa.
#   - "api"          → think=True como parámetro de client.chat() (Qwen).
#   - "system_token" → token "<|think|>" al inicio del system prompt (Gemma).
OLLAMA_MODEL_PROFILES = {
    "qwen35": {
        "tag": "qwen3.5:9b",
        "think_method": "api",
        "options": {
            "num_ctx":   8192,   # margen para think interno (~1-3k) + prompt CoT-aware (~1.5k) + output (~500)
            "num_batch": 1024,   # prompt batch agresivo (RTX 4070)
            "num_thread": 0,     # 0 = auto
        },
        "keep_alive": "24h",     # mantenerlo cargado entre runs
    },
    "gemma4": {
        # Producción = "gemma4:e4b" (registry) CON Ollama 0.23.2 (motor Go, κ 0.90).
        # El tag "gemma4:e4b-native" (GGUF llama.cpp, LLMs/gemma4_native_gguf/) se
        # creó para probar Ollama 0.30 y NO recupera el κ — ver CLAUDE.md.
        "tag": "gemma4:e4b",
        "think_method": "system_token",
        "options": {
            "num_ctx":   8192,   # prompt scene v2 (~2k) + person_attrs (~1.5k) + VLM_MAX_TOKENS=2500 + imagen + margen
            "num_batch": 1024,   # prompt batch agresivo (RTX 4070 con ~6 GB VRAM libres)
            "num_thread": 0,     # 0 = auto
        },
        "keep_alive": "24h",     # mantenerlo cargado entre runs
    },
    "gemma4_12b": {
        # Experimento 2026-06-13: ¿el modelo grande (12B) recupera el κ que el
        # e4b perdió con Ollama 0.30 (path de visión llama.cpp/mtmd)? NO — demasiado
        # lento (~100s/img, ~14h/500) y sigue pasando por el path de visión llama.cpp.
        # ABANDONADO en favor del setup dual de Ollama (0.23.2 pipeline + 0.30 belleza).
        "tag": "gemma4:12b-it-qat",
        "think_method": "system_token",
        "options": {
            "num_ctx":   8192,
            "num_batch": 1024,
            "num_thread": 0,
        },
        "keep_alive": "24h",
    },
}
ACTIVE_OLLAMA_PROFILE = "gemma4"            # "gemma4" (producción, Ollama 0.23.2 :11434) | "gemma4_12b" (abandonado) | "qwen35"
OLLAMA_HOST  = "http://localhost:11434"     # URL del servidor Ollama (legacy, mantener
                                            # por compatibilidad con otros consumidores)
# Lista de servidores Ollama. Si tiene 2+ entradas, OllamaBackend hace
# round-robin para que cada copia del modelo procese requests en paralelo
# (cada server tiene su propio scheduler / forward pass).
# Para añadir un segundo servidor, ver CLAUDE.md (sección "Dos servidores Ollama").
OLLAMA_HOSTS = [
    "http://localhost:11434",
    #"http://localhost:11435",   # segundo servidor (lanzar manualmente, ver CLAUDE.md)
]
OLLAMA_THINK = VLM_ENABLE_THINKING          # Alias retro-compat (leído por OllamaBackend)

# Nº máx. de personas procesadas en paralelo dentro de cada classify_batch.
# Debe ser <= OLLAMA_NUM_PARALLEL del servidor Ollama (env var, ver CLAUDE.md).
# Combinado con el outer pool de processing/image.py (max_workers=3), el
# máximo teórico de requests concurrentes es 3 * OLLAMA_INNER_PARALLEL.
# IGNORADO si VLM_BACKEND="transformers": el batching real sustituye al
# ThreadPool por persona (los N crops se procesan en un único forward pass).
OLLAMA_INNER_PARALLEL = 1

# Nº de archivos procesados en paralelo (batch.py / crear_qwen_2.py).
# SERIE POR DEFECTO (=1): cualquier valor >1 hace que el cliente mande varias
# requests concurrentes, el servidor Ollama las batchea y el batching concurrente
# cambia el argmax en casos borderline → deriva run-a-run del κ aunque la
# inferencia sea greedy+seed=42. Ver CLAUDE.md "Sampling determinism".
# Subir solo con OLLAMA_HOSTS de varias instancias del modelo (cada una serie).
OLLAMA_FILE_PARALLEL = 1

# Nº de tareas en paralelo dentro de cada imagen (person_attrs + scene +
# social_distance) en processing/image.py. SERIE POR DEFECTO (=1) por la misma
# razón que OLLAMA_FILE_PARALLEL: el batching concurrente del servidor degrada
# el κ run-a-run. No subir salvo que la reproducibilidad no importe.
OLLAMA_OUTER_PARALLEL = 1

# Retry de OllamaBackend.generate() ante 500 / EOF / ResponseError.
# La caída del runner deja a Ollama ~5-7 s sin responder mientras relanza
# el subproceso, así que el backoff debe ser generoso.
OLLAMA_GENERATE_RETRIES = 3       # nº de reintentos (4 intentos totales)
OLLAMA_GENERATE_RETRY_BASE = 3.0  # backoff base en segundos (exp: 3, 6, 12)

# Si la fracción de personas con success=False supera este umbral en una
# imagen, se marca como vlm_status="failed". Por debajo, "partial" si hay
# algún fallo, "ok" si no.
VLM_IMAGE_FAILURE_THRESHOLD = 0.5

# Alias derivado para consumidores existentes (loader.py, banners, etc.)
OLLAMA_MODEL = OLLAMA_MODEL_PROFILES[ACTIVE_OLLAMA_PROFILE]["tag"]

# ============================================================================
# PARÁMETROS DE SAMPLING VLM (greedy + reproducible para todos los backends)
# ============================================================================
# Estos valores neutralizan los defaults del Modelfile de Ollama
# (presence_penalty=1.5, top_k=20, top_p=0.95) y los de Transformers,
# garantizando greedy decoding estricto y reproducible.
VLM_TEMPERATURE       = 0.0
VLM_TOP_K             = 1
VLM_TOP_P             = 1.0
VLM_PRESENCE_PENALTY  = 0.0
VLM_FREQUENCY_PENALTY = 0.0
VLM_REPEAT_PENALTY    = 1.0
VLM_SEED              = 42

# Configuración de detección y tracking
TRACKER_CONFIG = "botsort.yaml"     # BoT-SORT para mejor tracking
CONFIDENCE_THRESHOLD = 0.35         # Umbral de confianza mínimo (bajado para detectar estatuas, caricaturas y fotos pequeñas; subido a 0.30 para filtrar reflejos de personas en agua/espejos)
IOU_THRESHOLD = 0.30                # IoU threshold para NMS (bajado de 0.45 a 0.30 para suprimir duplicados de la misma persona con bboxes desplazados — p. ej. cuando un texto/overlay rompe la continuidad visual)
MIN_PERSON_CROP_SIZE = 4            # Tamaño mínimo de crop de persona en píxeles (alto y ancho)
# Filtro absoluto: ignorar detecciones cuyo bbox cubre menos de este ratio del
# área total del frame. Motivación (2026-06-03): personas en multitudes
# (Chicago Bean: ~17×60 px en 1080² = 0.001 del frame) son invisibles para el
# VLM tras el downsize a 896², y el modelo describe al sujeto dominante del
# panel emitiendo `posing` espurio. 0.01 = el bbox debe cubrir ≥1% del frame
# para procesarse; si no, la persona se ignora completamente (no aparece en
# classifications). Avatares legítimos típicamente cubren ≥1% del frame.
BBOX_MIN_FRAME_RATIO = 0.01
PERSON_CLASS_ID = 0                 # ID de la clase "person" en COCO

# Configuración de visualización
VISUALIZATION = {
    "bbox_color": (0, 255, 0),         # Verde - bounding boxes
    "bbox_thickness": 2,
    "keypoint_color": (255, 0, 0),     # Rojo - keypoints
    "keypoint_radius": 4,
    "skeleton_color": (0, 255, 255),   # Amarillo - conexiones esqueléticas
    "skeleton_thickness": 2,
    "track_color": (0, 0, 255),        # Azul - tracking IDs
    "text_font": cv2.FONT_HERSHEY_SIMPLEX,
    "text_scale": 0.4,
    "text_thickness": 2,
    "text_font_path": "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-ExtraLight.ttf",  # Fuente fina para zoom legible (Fedora)
    "text_font_size": 10,              # Tamaño de fuente en puntos para FreeType
    "text_supersample": 3,             # Factor de supersampling para renderizado vectorial-like
    "min_keypoint_conf": 0.5,          # Confianza mínima para dibujar keypoint
    # ---- Estilo de anotación (2026-07-03) --------------------------------
    "style": "chips",                  # "chips" (píldoras) | "classic" (bloque de texto legacy)
    "chip_opacity": 0.82,              # Opacidad de fondo de cada píldora (0-1)
    "chip_radius": 9,                  # Radio de esquina de la píldora (px)
    "chip_pad_x": 9,                   # Padding horizontal texto↔borde de la píldora
    "chip_pad_y": 5,                   # Padding vertical
    "chip_gap": 5,                     # Separación entre píldoras (px)
    "chip_font_size": 15,              # Tamaño de fuente del texto de la píldora
    "corner_bracket_len": 20,          # Longitud de las esquinas en L del bbox (px)
    "corner_bracket_thickness": 3,     # Grosor de las esquinas en L
    "corner_color": (255, 255, 255),   # Color BGR de las esquinas del bbox (blanco fijo, no por track)
    "fade_enabled": True,              # Fade-in de chips + esqueleto al iniciar cada escena
    "fade_seconds": 0.3,               # Duración del fade-in
    "skeleton_glow": True,             # Esqueleto con glow (línea gruesa translúcida + fina encima)
    "skeleton_glow_thickness": 7,      # Grosor de la capa de glow
    "scene_pill_color": (30, 30, 30),  # BGR fondo de la píldora del contador de escena (gris oscuro)
    "scene_text_color": (255, 255, 255),  # BGR texto del contador de escena
}

# Colores por categoría (BGR) para las píldoras del estilo "chips".
# Tonos pastel distinguibles; el texto va oscuro por contraste.
CATEGORY_COLORS = {
    "gender":          (239, 141,  91),   # azul
    "age":             (201, 201,  59),   # teal
    "behaviour":       (250, 139, 167),   # púrpura
    "activity":        (153, 211,  52),   # verde
    "body_display":    (182, 114, 244),   # rosa
    "location":        ( 36, 191, 251),   # ámbar
    "body_weight":     ( 60, 146, 251),   # naranja
    "muscle":          (113, 113, 248),   # rojo
    "silhouette":      (248, 140, 129),   # índigo
    "attire":          (140, 180, 248),   # salmón claro
    "social_distance": (238, 211,  34),   # cian
    "beauty":          (175, 164, 253),   # rosa palo
    "accessory":       ( 53, 230, 163),   # lima
}
CATEGORY_TEXT_COLOR = (0, 0, 0)           # BGR negro puro para el texto de la píldora (legibilidad)

# Keypoints faciales para detección frontal
FACE_KEYPOINTS = {
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}

# Conexiones del esqueleto COCO (17 keypoints)
SKELETON_CONNECTIONS = [
    # Cara
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso superior
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Brazos
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Piernas
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Colores por parte del cuerpo
LIMB_COLORS = {
    "face": (255, 255, 0),      # Amarillo
    "torso": (0, 255, 0),       # Verde
    "left_arm": (255, 0, 128),  # Magenta
    "right_arm": (0, 128, 255), # Azul claro
    "left_leg": (128, 0, 255),  # Púrpura
    "right_leg": (255, 128, 0), # Naranja
}
