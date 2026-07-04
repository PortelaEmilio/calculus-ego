# calculus-ego

Automated visual content analysis of self-representation. Given an image or
video, it detects each person, then runs a vision-language model to classify
them across the categories below, producing structured output for statistical
analysis.

## Demo

▶️ **[Annotated video sample (1080p)](https://portela-navarro.com/assets/friends-intro-annotated-1080p.mp4)**
— the pipeline's per-person annotations rendered on a short clip (detection,
pose, and the classification labels). GitHub does not embed external `.mp4`
files inline, so the link opens the video in your browser.

## Categories

Per detected person:

- **Gender** — male / female
- **Age** — childhood / youth / adulthood / old age
- **Behaviour** — visual social semiotics (demand/affiliation, demand/seduction, demand/submission, offer/ideal)
- **Activity** — sports / romance / posing / other
- **Body display** — normal clothes / revealing clothes / partially naked / no clothes at all
- **Location** — indoors / wilderness / city / no background
- **Weight** — thin / median / overweight
- **Musculature** — visible / not visible
- **Social distance** — proxemic distance (Hall): intimate → public
- **Accessories** (multi-label) — makeup, tattoos, bags, belts, jewelry, headwear, eyewear
- **Beauty** *(optional)* — facial attractiveness on a continuous 1–10 scale (one decimal),
  scored in a deferred pass over `demand/*` faces (see [Beauty](#beauty-optional))

Plus **OCR** of any text in the image.

## Architecture

```
input image/video
      │
      ▼
YOLO26 detection + pose  ──►  per-person crops + keypoints
      │
      ▼
Qwen3.5-9B (Transformers)  ──►  2 merged VLM calls per person
      │                          · person attributes (gender, age, behaviour,
      │                            body display, weight, musculature, accessory)
      │                          · scene context (activity, location)
      ▼
JSON + CSV per-person classifications
```

Social distance is **deterministic from pose keypoints** (proxemic mapping from
the lowest visible body part + bbox size); `demand/submission` behaviour is
decided by a **pose gate** (camera-elevation read from shoulder-vs-eye geometry),
not by the VLM. A `BBOX_MIN_FRAME_RATIO` gate drops detections too small for the
VLM to read. Everything is greedy + seeded for reproducibility.

| Path | File |
|---|---|
| CLI entry | `main.py` |
| Model loading | `models/loader.py` |
| Image processing | `processing/image.py` |
| Video processing | `processing/video.py`, `processing/video_validation.py` |
| Batch over a CSV | `processing/batch.py` |
| Merged classifiers | `models/person_attributes.py`, `models/scene_context.py` |
| Per-task classifiers | `models/{behaviour,activity,age_gender,body_display,location,accessory,ocr,social_distance}.py` |
| Prompts | `models/prompts_qwen3.py` |
| VLM backend | `models/qwen35_vlm_backend.py` (+ `models/backends/json_flatten_backend.py`) |
| Annotation / visualization | `utils/visualization.py` |

## Requirements

- Python 3.12+
- A CUDA GPU with ~10 GB free (`Qwen/Qwen3.5-9B` in 4-bit NF4). The base model is
  downloaded from the Hugging Face Hub on first use.
- Python deps: `pip install -r requirements.txt`
- YOLO26 weights (`yolo26x-pose.pt`) — downloaded automatically by ultralytics on
  first run.

> **gcc ≤ 15 required.** Qwen3.5 compiles its GatedDeltaNet linear-attention
> kernels at runtime and CUDA rejects gcc 16+. If the first forward pass crashes
> with a compilation error, put a gcc-15 wrapper on `PATH`
> (e.g. symlinks `gcc`/`g++` → `gcc-15`/`g++-15`).

> **CUDA 13 note:** if PyTorch ships the CUDA 13 runtime in the venv without
> adding it to `LD_LIBRARY_PATH`, the first dynamic kernel can crash with
> `nvrtc: failed to open libnvrtc-builtins.so.13.0`. Fix by exporting the
> nvidia `cu13/lib` path before running.

## Usage

```bash
# Analyze one image or video
python main.py single path/to/image.jpg

# Batch over a CSV of post metadata
python main.py batch posts.csv --sample-size 400 \
    --multimedia-dir multimedia_downloads/ -o out.csv

# Re-analyze the unique images of a manual annotation Excel → AI Excel
python main.py manual manual.xlsx --output ai_results.xlsx --run-prefix run1

# Build an AI-vs-manual comparison workbook
python main.py compare manual.xlsx -o comparison.xlsx
```

Output is written under `output/` (JSON summaries and annotated images);
`manual`/`compare` write workbooks next to their inputs.

## Configuration

All feature flags, model paths and sampling settings live in `config.py`. The
active backend is `Qwen/Qwen3.5-9B` served via Transformers (4-bit NF4, eager
attention, thinking off); its JSON output is flattened to line-based fields by a
proxy backend. The classifier and the beauty adapter share **one** base model in
VRAM (the LoRA is toggled on/off), so the deferred beauty pass reuses the loaded
model instead of loading a second copy.

## Beauty (optional)

A continuous **1–10 facial-attractiveness** estimator (one decimal) runs as a
**deferred pass**: after the classifiers finish, a `Qwen3.5-9B` (4-bit) model
with a LoRA adapter scores **only the faces whose behaviour is `demand/*`**
(linked to phase-1 detections by IoU). Output is the decimal column
`IA Belleza (1-10)` (per-person in `manual`, a `*_belleza.xlsx` sidecar in
`batch`, printed in `single`).

- **Model.** The LoRA adapter ships in the repo at `models/beauty_adapter/`
  (via **Git LFS**, ~29 MB) — `git clone` brings it down. The base
  `Qwen/Qwen3.5-9B` is downloaded from the Hugging Face Hub on first use.
  Point elsewhere with `export BEAUTY_ADAPTER_PATH=/path/to/adapter`.
- **Requirements.** A CUDA GPU (~6 GB free during the deferred pass) and the
  extra deps in `requirements.txt` (`transformers`, `peft`, `bitsandbytes`,
  `accelerate`). With Git LFS: `git lfs install` before cloning (or
  `git lfs pull` after).
- **Toggle.** On by default (`config.ENABLE_BEAUTY_PASS`); skip per run with
  `--no-beauty`. Degrades gracefully: if the adapter is missing or there are no
  `demand/*` faces, the run finishes without the beauty column instead of failing.

### How the adapter was trained

The LoRA adapter was fine-tuned from `Qwen/Qwen3.5-9B` with **QLoRA** (4-bit, rank 8)
in a single **joint run** mixing five public facial-beauty datasets:

| Dataset | Faces | Raw score |
|---|---|---|
| SCUT-FBP5500 | 5500 | 1–5 mean rating |
| CFD 3.0 | ~600 | attractiveness mean |
| MEBeauty | ~2550 | 1–10 mean |
| HotOrNot | ~2000 | z-normalised |
| M2B | 1240 | 1–10 consensus |

Each dataset's score is mapped to a **continuous 1–10 target (one decimal) by
percentile rank** (`1 + 9·(rank−1)/(n−1)`), which puts the five heterogeneous
scales on one common, outlier-robust axis; data is split 80/10/10 stratified by
decile. The model is trained to emit the number (e.g. `7.3`).

**Performance** on the held-out test split of each dataset:

| Test set | n | Pearson r | Spearman ρ | MAE |
|---|---:|---:|---:|---:|
| SCUT-FBP5500 | 549 | 0.934 | 0.933 | 0.75 |
| CFD 3.0 | 60 | 0.806 | 0.800 | 1.17 |
| MEBeauty | 253 | 0.807 | 0.814 | 1.27 |
| HotOrNot | 204 | 0.606 | 0.625 | 1.87 |
| M2B | 130 | 0.524 | 0.556 | 2.13 |

The curated lab datasets (SCUT, CFD, MEBeauty) correlate strongly; the noisier
in-the-wild sets (HotOrNot, M2B) sit lower — their ceiling is the label noise
(few ratings per face, low-resolution crops), not the model.

> **The datasets are *not* redistributed here** — only the trained adapter ships.
> Obtain each from its original source under its own license if you want to
> reproduce training.

## Notes

- The `manual` / `compare` validation tooling and the batch CSV writer use
  Spanish column labels (`IA Gen.`, `Ruta Img.`, `ia_genero`, …) as their
  on-disk I/O contract; these are kept as-is so the workbooks remain compatible
  with existing manual-annotation files. The code, logs and JSON keys are English.
