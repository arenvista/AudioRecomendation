# Audio Recommendation

A content-based music recommendation system that skips collaborative filtering and metadata shortcuts (artist, genre tags, listening habits of other users) in favor of learning directly from what a track *sounds like* and *says*. Audio is converted to spectrograms and embedded with a pretrained Audio Spectrogram Transformer (AST), lyrics are embedded with CLIP's text encoder, the two modalities are aligned via contrastive learning, and the resulting embeddings are projected into a 3D PCA space where recommendations are made by nearest-neighbor distance.

## How it works

1. **Spectrogram generation** — MP3s are loaded and converted to log-frequency spectrograms via `librosa`, saved as images (and optionally flattened CSVs).
2. **Audio representation (AST)** — A pretrained Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) encodes each spectrogram into a 768-dim embedding.
3. **Multimodal alignment (CLIP)** — For tracks with lyrics, a CLIP text encoder produces a 512-dim lyric embedding. The AST embedding is projected down to 512 dims and aligned to the lyric embedding using a symmetric InfoNCE-style contrastive loss (learned temperature, AdamW optimizer).
4. **Dimensionality reduction (PCA)** — The aligned 512-dim embeddings are reduced to 3 principal components (PC1–PC3), giving an interpretable coordinate space for each track.
5. **Recommendation** — User "profiles" are built from a small library of songs. Recommendations and cluster-quality metrics (intra-/inter-cluster distance) are computed via Euclidean distance in PCA space.

## Project structure

```
audiorecomendation/
├── main.py                    # CLI entrypoint (Spectro | PCA modes)
├── data/
│   ├── gen_metadata_csv.py    # Builds metadata.csv (filename -> artist) from MP3 tags
│   ├── gen_map_csv.py         # Maps audio files to matching lyrics files
│   └── audio/                 # Source MP3s (not tracked)
├── src/
│   ├── audioencoder/
│   │   ├── audioprocessor.py  # AST embedding extraction + PCA visualization/export
│   │   ├── finder.py          # Fuzzy-search directory/file picker (prompt_toolkit)
│   │   ├── dataloader.py      # CSV-driven dataset loader (audio + lyrics manifest)
│   │   ├── contrastive.py     # ASTCLIPAlignmentModel: AST + CLIP contrastive alignment
│   │   ├── handler.py         # Training loop for the contrastive alignment model
│   │   ├── trainer.py         # Fine-tuning script for AST on a custom audio dataset
│   │   ├── ast/                # Duplicate/alternate copies of the above (AST-focused)
│   │   └── contrast/           # Duplicate/alternate copies of the above (contrastive-focused)
│   └── spectro/
│       ├── spectrogram.py     # SongData: loads audio, renders/exports spectrograms
│       └── selector.py        # BatchProceesor: batch-processes a chosen directory of songs
├── utils/
│   ├── gen_pca.py             # Re-plots PCA results (2D/3D) from a saved CSV, with highlighting
│   ├── tag.py                 # Enriches a PCA CSV with ID3 metadata (artist, album, genre, etc.)
│   └── user_recs.py           # Recommendation engine: nearest-neighbor + cluster distance metrics
├── output/
│   ├── pca/                   # Generated PCA CSVs and plots
│   └── specto/                # Generated spectrogram images/CSVs
├── tex/                       # LaTeX writeup of the project (main.tex, template.tex)
└── manim.cfg                  # Config for optional Manim visualizations
```

> **Note:** `src/audioencoder/ast/` and `src/audioencoder/contrast/` currently contain near-duplicate copies of files also found directly under `src/audioencoder/`. These appear to be in-progress refactors/experiments rather than separate modules.

## Requirements

- Python 3.12
- Managed with [uv](https://github.com/astral-sh/uv) (`pyproject.toml` / `uv.lock`)

Key dependencies: `torch`, `transformers`, `librosa`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `pandas`, `matplotlib`, `tinytag`, `prompt-toolkit`, `questionary`, `manim`.

Install with:

```bash
uv sync
```

## Usage

### 1. Generate spectrograms

```bash
uv run main.py Spectro
```

Prompts you to fuzzy-search for a directory under `data/` containing `.mp3` files, then batch-generates spectrogram images (and optionally CSVs) into `output/specto/`.

### 2. Generate embeddings + PCA

```bash
uv run main.py PCA
```

Prompts you to select a directory of songs, extracts AST embeddings for each, reduces them to 3D via PCA, and saves both a CSV (`output/pca/<timestamp>_pca.csv`) and a 3D scatter plot (`output/pca/<timestamp>_pca.png`).

### 3. Build supporting metadata

```bash
uv run data/gen_metadata_csv.py   # file_name -> artist label CSV from ID3 tags
uv run data/gen_map_csv.py        # matches audio files to lyrics .txt files by filename stem
uv run utils/tag.py               # enriches a PCA CSV with full ID3 metadata (title, album, genre, etc.)
```

### 4. Re-plot / inspect PCA results

```bash
uv run utils/gen_pca.py
```

Reloads a saved PCA CSV and regenerates 2D and 3D plots colored by Artist, Album, or Genre, with optional highlighting of specific artists.

### 5. Generate recommendations

```bash
uv run utils/user_recs.py
```

Loads a full-catalog PCA CSV and a per-user PCA CSV, finds each user's nearest-neighbor recommendations in PCA space, and reports intra-cluster (within a user's library) and inter-cluster (between users) average distances as a sanity check on clustering quality.

## Data

- **[FMA (Free Music Archive)](https://github.com/mdeff/fma)** — primary training corpus (106,574 tracks, 16,341 artists, 161 genres); the "large" subset was used, with each track trimmed to 30 seconds.
- Lyrics were sourced separately per track and matched to audio files by filename stem for the CLIP alignment stage.

## Known limitations

- Lyrical sentiment doesn't always match a track's musical tone, which can weaken the contrastive alignment.
- No practical way to pull real user listening data from platforms like Spotify, so evaluation used small, hand-curated user libraries instead.
- FFmpeg's `libcodec.dll` wasn't available on the multicore HPC partition used for training and had to be manually built and added to `PATH`.
- A centralized lyrics dataset covering the full corpus wasn't available; instrumental tracks have no lyrics by definition.

***See paper below for more information*** 

---
<p align="center">
  <img src="./tex/main-1.svg" alt="Page 1" width="80%">
</p>

