"""
Prompts JSON "no-reasoning" para Transformers + gemma-4-E4B-it con thinking OFF.

Experimento (2026-06-28): variante de prompt para el modo SIN razonamiento. En vez
del árbol de decisión de producción (`prompts_ollama_cot`, pensado para un modelo que
razona: "STOP-at-first-match", "CHECK 1/2/3", enumeración previa), aquí cada categoría
se define como DESCRIPCIÓN + INDICADORES VISUALES y se exige una salida JSON limpia,
con un preámbulo que prohíbe explícitamente la cadena de pensamiento.

El closed-set de valores es IDÉNTICO a producción (mismos labels) → los parsers
downstream y el ground truth no cambian. La salida JSON la aplana a `Label: value` un
backend-proxy externo (ver scratchpad `experimento_json.py`) ANTES de llegar a los
parsers line-based, así que este módulo solo define los prompts.

2026-07-27 — variante "VH": `demand/submission` vuelve al closed-set de behaviour, descrita por
TIPO DE PLANO ("a HIGH-ANGLE SHOT … picado", con el contrapicado excluido explícitamente). Sustituye
al gate de pose post-VLM, que se ELIMINÓ de `processing/image.py` porque medía postura encorvada y no
elevación de cámara (en el sample 500 de IG: 16 de 21 submissions falsas, precisión 24%). VH mide
22/24 con 0 falsos positivos en el banco FLUX de comportamiento. Con esto vídeo también puede emitir
la clase (el gate solo existía en el path de imagen). Ver el informe del experimento en
`validacion_imagenes/datos-sinteticos/flux/edad_genero_cat/resultados_submission_prompt/`.

2026-08-31 — ubicación SIN persona: nuevo `build_location_only_prompt()` para imágenes y escenas de
vídeo donde YOLO no detecta a nadie (capturas de tuit, tarjetas de texto, paisajes). El prompt de
escena no vale ahí porque afirma que hay una persona en una caja amarilla. Las 4 viñetas de pistas
se comparten vía `_LOCATION_CUES`, así que `_LOCATION_DEF` (el path CON persona) queda byte-idéntico.

API expuesta (idéntica a prompts_ollama_cot / prompts_gemma4_official):
    - build_person_attrs_prompt(include_body_shape: bool) -> str
    - build_scene_prompt(include_social_distance: bool) -> str
    - build_location_only_prompt() -> str        (imágenes/escenas SIN personas)
    - SCENE_PROMPT / LOCATION_ONLY_PROMPT
    - SOCIAL_DISTANCE_PROMPT
    - PROMPT_VERSION
"""
from typing import Final

# Social distance: JSON de partes-visibles que `social_distance.py:_map_parts_to_category()`
# parsea por su cuenta. El proxy JsonFlatteningBackend NO lo toca (solo aplana person_attrs/scene).
SOCIAL_DISTANCE_PROMPT: Final = """You are a visual analyst. Look at the highlighted TARGET person (yellow box labeled TARGET) in the image.

For each body part listed below, decide if it is VISIBLE inside the image frame for the TARGET person. Answer with a JSON object — Python will map the visibility pattern to a category afterwards. Do NOT classify anything yourself.

VISIBILITY RULES:
- A part is VISIBLE (1) when its outline/silhouette is INSIDE the image frame AND not hidden behind an opaque object.
- A part is NOT visible (0) when it falls OUTSIDE the image frame OR is fully occluded by an opaque object (car seat, desk, table, counter, another person, large prop, vehicle door).
- Thin fabric, hair, or shadow over a body region do NOT make it "not visible".
- NEVER infer that a part "continues downward" out of frame. If you cannot see it in the image, it is 0.
- The SIZE of the TARGET in the frame is IRRELEVANT. A tiny TARGET / avatar / thumbnail is judged by which body parts are inside its own frame.

BODY PARTS:
- head: face / head of the TARGET (any portion of face or skull).
- shoulders: BOTH shoulders of the TARGET clearly inside the frame (set 1 ONLY if both shoulder lines are drawn or photographed, not just hinted at by the edge of an avatar circle).
- chest: upper torso below the shoulders (collarbones-to-bottom-of-ribs region — neckline, shirt collar, top of pectorals, tank-top straps continuing into torso fabric).
- waist: midsection at the hipline / belly-button level.
- legs: any portion of thighs, knees, or calves.
- feet: feet and/or ankles.

FRAMING (only matters when the full body is visible — feet=1):
- frame_filled: 1 if the TARGET fills most of the image frame with LITTLE empty space around them; 0 if there is significant empty space and the background dominates around the TARGET.
- If feet=0, set frame_filled to 0 (it will be ignored).

Output EXACTLY one line of valid JSON, NOTHING ELSE — no preamble, no explanation, no reasoning, no thinking tags, no commentary:
{"head": 0|1, "shoulders": 0|1, "chest": 0|1, "waist": 0|1, "legs": 0|1, "feet": 0|1, "frame_filled": 0|1}
"""

PROMPT_VERSION: Final = "qwen3_scene_location_20260831"


# ===========================================================================
# Preámbulo común (rol "system" embebido como texto — Gemma-4 no toma rol system fiable)
# ===========================================================================

_NOREASON_PREAMBLE: Final = (
    "You are an expert visual classification model.\n"
    "Instructions:\n"
    "- Provide ONLY visual observations.\n"
    "- Do NOT provide reasoning or chain-of-thought.\n"
    "- Do NOT explain how you reached the decision.\n"
    "- Do NOT invent details that are not visible.\n"
    "- Be concise and factual.\n"
    "- For each category, choose EXACTLY ONE value from its closed list, using "
    "lowercase and the exact spelling shown."
)


# ===========================================================================
# Person attributes (Merge A) — descripción + indicadores visuales por categoría
# ===========================================================================

_GENDER_DEF: Final = """gender — biological sex read from the visible face.
- male: face visible (frontal or 3/4) with male sex-typical features.
- female: face visible (frontal or 3/4) with female sex-typical features.
- na: face not visible (back of head, out of frame, masked/covered, blurred beyond recognition) or genuinely ambiguous.
Indicators: visible facial sex features. NEVER infer sex from hair length, build, or clothing alone."""

_AGE_DEF: Final = """age — life stage read from the visible face, skin, and hair.
- childhood: 0-12. Child face with childlike facial features.
- youth: 13-29. Smooth skin, no significant wrinkles.
- adulthood: 30-59. Visible skin-aging: wrinkles around eyes/forehead/mouth, partial grey hair, receding hairline, sagging.
- old age: 60+. Advanced aging: deep wrinkles, mostly grey/white hair, age spots, advanced sagging.
- na: only when the face/skin is genuinely not assessable at all (face entirely hidden, fully covered, or too blurred).
Indicators: skin texture and hair colour only. Body size, frame, muscles, beard, or jawline are NOT aging cues. If skin is smooth and lacks wrinkles, default to 'youth'. Do not use 'na' if any skin is visible."""

_BEHAVIOUR_DEF: Final = """behaviour — gaze and head configuration toward the viewer, plus the shot type.
- demand/affiliation: head frontal or slightly turned, both pupils visible and pointing at the camera lens, neutral/friendly engagement.
- demand/seduction: head frontal/slight, pupils at the lens, PLUS at least two seductive cues (parted/pouty lips, heavy-lidded sultry eyes, arched/suggestive body tilt, flirtatious smile, hand sensually near lips/hair/neck, skin deliberately exposed). An intense or tough stare alone is NOT seduction.
- demand/submission: a HIGH-ANGLE SHOT (also called an overhead shot, a bird's-eye-view portrait, or "picado"): the camera is placed above the subject and tilted down, and the subject looks up into it. If you would caption this photo "shot from a high angle" or "photographed from above", it is demand/submission. A LOW-ANGLE shot (from below, "contrapicado") is NOT submission.
- offer/ideal: head in strong profile / strongly turned / back of head, OR pupils NOT at the lens (looking down/up/sideways, eyes closed/shadowed/covered, face too small or blurred to resolve the pupils).
Indicators: head angle, pupil direction, and the shot type."""

_BODY_DISPLAY_DEF: Final = """bodydisplay — how much the body is exposed. Decide in THIS order and assign the FIRST class that fits:
1) partially naked (check FIRST) — Wearing ONLY swimwear/underwear, or has a completely bare chest, bare torso, or bare midriff (exposed belly). * CRITICAL: Bare arms, bare shoulders, or tank tops/sleeveless shirts do NOT count here (they belong to category 3).
2) no clothes at all — fully nude (no garment at all) AND the body is visible at least down to the knees.
3) revealing clothes — everyday STREET clothing that deliberately exposes significant skin but is NOT swimwear/underwear: a tank top, sleeveless/muscle shirt, spaghetti-strap or crop top, deep neckline, mini-skirt, or short shorts, worn together with normal bottoms (trousers, jeans, leggings, or a skirt) or normal top. Bare shoulders/arms from a sleeveless top with normal bottoms = revealing (NOT partially naked). Short sleeves and knee-length skirts are NOT enough — those are normal clothes.
4) normal clothes — everyday clothing covering the body without significant skin exposure (t-shirts incl. short-sleeve, shirts, sweaters, jackets, trousers, jeans, normal skirts, sportswear). DEFAULT when clothing is unclear or only the head is visible.
Indicators: is the lower body in swim/underwear briefs with bare legs, or in normal trousers/leggings/skirt? Is the chest/torso bare? Swim/underwear or bare torso → partially naked; sleeveless street top + normal bottoms → revealing."""

_BODY_WEIGHT_DEF: Final = """bodyweight — body fatness on one ordered scale: light build < median < overweight. Decide in THIS order and assign the FIRST class that fits (do NOT weigh all three at once, and do NOT pick the middle as a safe default):
1) light build (check FIRST) — narrow or slim limbs, a waist small relative to the shoulders/hips, and a flat or hollow stomach. Lean, toned, athletic, slight, or bony bodies are ALL light build. An older or elderly person with a thin, lean, or slight frame is light build, NOT median — ageing does not add fat. If the body carries little flesh, choose light build here and stop.
2) overweight — only if clearly NOT light: a belly that is rounded or pushes outward past the chest line, a thick waist, and soft or padded limbs. A soft, rounded midsection is enough; the body need not look very large.
3) median — ONLY when the body is clearly NEITHER light NOR overweight: ordinary limbs and waist (neither slim nor thick) with a flat-to-slightly-soft stomach. If the body looks lean, slim, or slight, it is light build, NOT median.
- not visible: cannot see enough of the torso (head/face only, cropped at/above the waist, body fully sideways, heavy occlusion).
Indicators: width of limbs and waist, contour of the belly. Judge the body itself — not clothing, pose, camera angle, or age."""

_MUSCLE_DEF: Final = """muscle — visible muscle definition on the person's body.
- visible: clear, defined muscles are apparent — muscle shape, separation, or striations in the arms, shoulders, chest, abdomen, or back; a toned, athletic, or muscular physique. Judge the body itself, even when read through fitted clothing or on exposed arms/neck.
- not visible: no apparent muscle definition (soft, smooth body contours, no muscle separation) OR the body cannot be assessed at all (only the head/bust is shown, the body is occluded, or the person is too distant/blurred to judge).
Indicators: muscle relief and separation on arms, shoulders, and torso. NEVER infer musculature from clothing style (e.g. sportswear) or pose alone — judge the actual body."""

_ATTIRE_DEF: Final = """attire — the STYLE/formality of the garments the person is wearing. Judge the KIND of clothing, NOT how much skin is exposed (skin exposure is a SEPARATE category). Assign the FIRST class that fits:
1) underwear/swimwear — the person is wearing ONLY swimwear or underwear: bikini, swimsuit, swim trunks, bra, briefs, boxers, lingerie. Beachwear worn as the sole garment counts here.
2) sportswear — athletic/gym clothing built for exercise: tracksuit, gym shorts + athletic top, leggings + sports bra, football/basketball kit or team jersey worn to play, cycling/running gear, martial-arts gi.
3) uniform — an occupational or institutional uniform that signals a role: military fatigues or dress uniform, police/firefighter/medical/security/pilot/chef/waiter/school uniform, or a numbered team uniform worn as a formal kit.
4) formal — dressy or ceremonial clothing: suit and tie, blazer with dress trousers, tuxedo, evening gown, cocktail dress, or TRADITIONAL/CULTURAL formal dress (kimono, sari, tunic, kilt, thobe, regional costume).
5) casual — everyday street clothing not covered above: t-shirt, jeans, jumper, hoodie, casual dress, shorts with a normal top, everyday jacket. DEFAULT when the outfit is ordinary or ambiguous but the body is visible.
- not visible: cannot judge the clothing — only the head/bust is shown, the person is heavily occluded, or too distant/blurred.
Indicators: garment type and formality. A team jersey worn to play sport is sportswear; the same crest on a stiff dress kit is uniform. Traditional/cultural formal dress is formal. Do NOT use skin exposure to decide this."""

_ACCESSORIES_DEF: Final = """accessories — for each, set class to "1" if visibly present, else "0".
- makeup: visible cosmetics on the face (lipstick, eyeliner, eyeshadow) or painted nails.
- tattoos: a clearly bounded ink design whose content you can NAME (letter, word, symbol, animal, figure). An unnamed dark patch, muscle shading, or anatomical line art is NOT a tattoo.
- bags: visible handbag, backpack, clutch, or carried container.
- belts: a decorative or functional strap around the waist.
- jewelry: earring, necklace, pendant, bracelet, watch, ring, or piercing.
- headwear: hat, cap, beanie, headband.
- eyewear: glasses, sunglasses, goggles."""


# Silueta ELIMINADA (2026-07-04): κ 0.306 en el sample 500 real, no se recuperó.
# Peso (bodyweight) y musculatura (muscle) SIGUEN aquí — no dependían de la silueta.
_PA_OUTPUT: Final = """Output ONLY this JSON object, nothing else (no prose, no markdown fences, no reasoning):
{
  "gender":      {"class": "<male|female|na>", "features": ["<short visual cue>"]},
  "age":         {"class": "<childhood|youth|adulthood|old age|na>", "features": ["<short visual cue>"]},
  "behaviour":   {"class": "<demand/affiliation|demand/seduction|demand/submission|offer/ideal>", "features": ["<short visual cue>"]},
  "bodydisplay": {"class": "<normal clothes|revealing clothes|partially naked|no clothes at all>", "features": ["<short visual cue>"]},
  "bodyweight":  {"class": "<light build|median|overweight|not visible>"},
  "muscle":      {"class": "<visible|not visible>"},
  "attire":      {"class": "<underwear/swimwear|sportswear|uniform|formal|casual|not visible>"},
  "makeup":   {"class": "<0|1>"},
  "tattoos":  {"class": "<0|1>"},
  "bags":     {"class": "<0|1>"},
  "belts":    {"class": "<0|1>"},
  "jewelry":  {"class": "<0|1>"},
  "headwear": {"class": "<0|1>"},
  "eyewear":  {"class": "<0|1>"}
}"""


def build_person_attrs_prompt(include_body_shape: bool = False) -> str:
    # include_body_shape se conserva por compatibilidad de firma pero se IGNORA:
    # la silueta se eliminó; peso y musculatura se preguntan siempre.
    blocks = [
        _NOREASON_PREAMBLE,
        "Classify the SINGLE person shown in the image across the categories below.",
        _GENDER_DEF,
        _AGE_DEF,
        _BEHAVIOUR_DEF,
        _BODY_DISPLAY_DEF,
        _BODY_WEIGHT_DEF,
        _MUSCLE_DEF,
        _ATTIRE_DEF,
        _ACCESSORIES_DEF,
        _PA_OUTPUT,
    ]
    return "\n\n".join(blocks)


# ===========================================================================
# Scene context (Merge B — activity + location)
# ===========================================================================

_SCENE_INTRO: Final = (
    "There is EXACTLY ONE person highlighted in the image: the person inside the "
    "single yellow rectangular box labeled TARGET. Classify ONLY that ONE person. "
    "Do NOT enumerate panels, analyze multiple TARGETs, or describe other people "
    "who are NOT inside the yellow box."
)

_ACTIVITY_DEF: Final = """activity — the main action of the TARGET person.
- sports: active physical exercise or a documented training context (running, lifting, ball sports, yoga, cycling, gym, martial arts, tactical/military/airsoft field training). Judge by the ACTION, not the background: dynamic athletic motion (mid-run, jumping, fighting/guard stance, lifting, exercising) is sports EVEN on a plain/studio background.
- romance: two or more people kissing or embracing in a clearly romantic/intimate way (not a friendly hug); couples cradling, holding, or gazing at each other.
- posing: the TARGET's primary action is being photographed — body arranged for the camera, no other activity (modeling, fashion shots, mirror selfies, styled portraits, decorative prop-holding). If removing the camera leaves nothing happening → posing.
- other: a real activity beyond being photographed (eating, drinking, cooking, using a phone, chatting, walking, working, holding a baby/pet, dancing, singing, playing an instrument, traveling). ALSO use other when the TARGET is too small, distant, blurred, or back-turned to identify the action.
Indicators: what the TARGET is doing. A TARGET in the background or too small to read → other (never default to posing). Avatars/portraits with a person posed for the camera → posing."""

# Las 4 viñetas de pistas se comparten entre el prompt CON persona (`_LOCATION_DEF`, que
# ancla la ubicación al TARGET) y el prompt SIN persona (`_LOCATION_DEF_NOPERSON`, para
# imágenes/escenas donde YOLO no detecta a nadie). El texto de las viñetas debe quedar
# IDÉNTICO al histórico: tocarlo movería el κ 0.889 de Ubicación del sample 500.
_LOCATION_CUES: Final = """- indoors: inside any enclosed/built environment. Cues: walls, doors, windows, ceiling, furniture (bed, sofa, chair, table, desk, shelf), bedding/curtains, kitchen/bathroom fixtures, indoor flooring, interior lighting, vehicle interiors, any built venue.
- wilderness: natural outdoor setting. Cues: trees/foliage, sky/clouds/stars dominant, sea/river/lake, mountains/cliffs/rocks, sand/beach/dunes, desert/arid terrain, grass field, snow/ice.
- city: urban outdoor setting. Cues: buildings/skyscrapers seen from outside, streets/sidewalks/asphalt, cars, traffic signs/lights, lampposts, billboards, monuments, public squares.
- no background: the whole background is a flat colour, a simple gradient, a studio backdrop, a dark void with only text, or a seamless blur — no architectural, natural, or urban feature visible."""

_LOCATION_DEF: Final = (
    "location — where the TARGET person is.\n"
    + _LOCATION_CUES + "\n"
    + """Special case — avatar: if the TARGET sits inside a clear circular/oval profile-picture shape with a contrasting solid border, classify ONLY by what is inside that circle (ignore the rest of the image). You MUST pick one of the four values; there is no NA."""
)

# Variante para imágenes/escenas SIN ninguna persona detectada (2026-08-31). Misma lista
# de pistas y mismo closed-set, pero la tarea se formula sobre la imagen entera: sin caja
# amarilla, "where the TARGET person is" no tiene referente. Se explicita que las tarjetas
# de texto y las capturas (masivas en este corpus) son `no background`.
_LOCATION_DEF_NOPERSON: Final = (
    "location — where this photo was taken.\n"
    + _LOCATION_CUES + "\n"
    + "There is NO person to focus on: judge the setting from the image as a whole. A text "
      "card, a screenshot of a post, a plain graphic, a logo, or a drawing with no scene "
      "behind it is `no background`. You MUST pick one of the four values; there is no NA."
)

_SCENE_OUTPUT: Final = """Output ONLY this JSON object, nothing else (no prose, no markdown fences, no reasoning):
{
  "activity": {"class": "<sports|romance|posing|other>", "features": ["<short visual cue>"]},
  "location": {"class": "<indoors|wilderness|city|no background>", "features": ["<short visual cue>"]}
}"""

_SCENE_OUTPUT_WITH_SD: Final = """Output ONLY this JSON object, nothing else (no prose, no markdown fences, no reasoning):
{
  "activity":       {"class": "<sports|romance|posing|other>", "features": ["<short visual cue>"]},
  "location":       {"class": "<indoors|wilderness|city|no background>", "features": ["<short visual cue>"]},
  "socialdistance": {"class": "<intimate distance|close personal distance|far personal distance|close social distance|far social distance|public distance>"}
}"""


def build_scene_prompt(include_social_distance: bool) -> str:
    # En este pipeline USE_LEGACY_SOCIAL_DISTANCE=True → scene_context llama con
    # include_social_distance=False (la distancia social va por su clasificador propio
    # con SOCIAL_DISTANCE_PROMPT). El path True queda por compatibilidad de API.
    blocks = [_NOREASON_PREAMBLE, _SCENE_INTRO, _ACTIVITY_DEF, _LOCATION_DEF]
    blocks.append(_SCENE_OUTPUT_WITH_SD if include_social_distance else _SCENE_OUTPUT)
    return "\n\n".join(blocks)


SCENE_PROMPT: Final = build_scene_prompt(False)


# ===========================================================================
# Ubicación SIN persona (2026-08-31)
# ===========================================================================
# Cuando YOLO no detecta a nadie no hay caja amarilla que señalar, así que el
# prompt de escena (que afirma "There is EXACTLY ONE person highlighted…") no es
# aplicable. Esta variante pide SOLO la ubicación sobre la imagen completa. La
# consume `scene_context.SceneContextClassifier.classify_location_only()`; el
# proxy `JsonFlatteningBackend` aplana la clave `location` igual que en el
# prompt de escena, así que el parser downstream (`_parse_location`) no cambia.

_SCENE_ONLY_INTRO: Final = (
    "This image contains NO highlighted person and no TARGET box. Classify the SETTING "
    "of the image as a whole. Do NOT describe people, panels, or the text content — "
    "report only where the photo was taken."
)

_LOCATION_ONLY_OUTPUT: Final = """Output ONLY this JSON object, nothing else (no prose, no markdown fences, no reasoning):
{
  "location": {"class": "<indoors|wilderness|city|no background>", "features": ["<short visual cue>"]}
}"""


def build_location_only_prompt() -> str:
    """Prompt de ubicación para imágenes/escenas SIN ninguna persona detectada."""
    return "\n\n".join([
        _NOREASON_PREAMBLE, _SCENE_ONLY_INTRO, _LOCATION_DEF_NOPERSON, _LOCATION_ONLY_OUTPUT,
    ])


LOCATION_ONLY_PROMPT: Final = build_location_only_prompt()
