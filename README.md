# Iranian Athlete Multimodal MCQ RAG

**A multimodal Retrieval-Augmented Generation (RAG) system for Persian multiple-choice questions about Iranian athletes.**
It builds **text & image indexes** (SentenceTransformers + OpenCLIP), performs **score-level fusion** for retrieval, and supports **generative answer selection** with evidence. Includes full evaluation scripts for both **text-only** and **image-based** MCQs.

> A detailed Persian write-up is included in the repo (see `docs/NLP_HW3_Report_Final_Persian.pdf`). Key design choices, pipeline diagrams, and results are summarized there. 

---

## Table of Contents

* [Overview](#overview)
* [Main Features](#main-features)
* [Method & Architecture](#method--architecture)
* [Data & Expected Layout](#data--expected-layout)
* [Quickstart (Google Colab)](#quickstart-google-colab)
* [Local Setup](#local-setup)
* [Reproducible Pipeline](#reproducible-pipeline)

  * [1) Build Catalog + Embeddings + Indexes](#1-build-catalog--embeddings--indexes)
  * [2) Retrieval & Diagnostics](#2-retrieval--diagnostics)
  * [3) MCQ (Text) — Retrieval Scoring + CSV](#3-mcq-text--retrieval-scoring--csv)
  * [4) MCQ (Image) — Multimodal Retrieval](#4-mcq-image--multimodal-retrieval)
  * [5) Generative Phase (Text set)](#5-generative-phase-text-set)
  * [6) Generative Phase (Multimodal set)](#6-generative-phase-multimodal-set)
  * [7) Evaluation & Plots](#7-evaluation--plots)
* [Results (high-level)](#results-highlevel)
* [How to Use the Search Programmatically](#how-to-use-the-search-programmatically)
* [Design Notes & Tricks](#design-notes--tricks)
* [Limitations](#limitations)
* [Troubleshooting](#troubleshooting)
* [Folder Structure](#folder-structure)
* [Model & Data Licenses](#model--data-licenses)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)
* [Team](#Team)

---

## Overview

This project implements a **multimodal RAG** pipeline targeted at **Persian** questions about **Iranian athletes**. It:

* cleans & normalizes a catalog of athlete bios **in Persian** (+ mixed Persian/Arabic script issues),
* **pairs images to athletes** heuristically,
* builds **text embeddings** (E5) and **image embeddings** (OpenCLIP) and stores them in **FAISS** (or **hnswlib** fallback),
* performs **text retrieval**, **text→image retrieval**, then **fuses** the scores,
* evaluates **Hit@K** for entity retrieval and **answer-string Hit@K** for MCQs,
* runs **generative answer selection** with JSON-only outputs and **evidence spans**,
* exports rich CSVs and plots for analysis.

The approach and key findings are also discussed in the Persian report. 

---

## Main Features

* **Persian-friendly normalization** (digits, Arabic vs Persian letters, ZWNJ/RTL marks)
* **Multilingual E5** (`intfloat/multilingual-e5-base`) for robust **text retrieval**
* **OpenCLIP ViT-B/32** (`laion2b_s34b_b79k`) for **image** and **text→image** retrieval
* **FAISS** indexing with **HNSW** fallback if FAISS isn’t available
* **Late fusion** (`alpha`-weighted) of text & image scores
* **MCQ pipelines** for both **text-only** and **image-based** questions
* **Generative phase** producing **JSON** answers with **evidence** + optional **bio**
* **End-to-end evaluation**: per-set accuracy, category breakdowns, confusion tables, and plots

---

## Method & Architecture

**High-level flow:**

1. **Preprocess & Enrich**

   * Normalize athlete names & summaries; sanitize filenames; map image files to each athlete using heuristics.
   * Handle Persian/Arabic script variants and digits.

2. **Embeddings**

   * **Text**: SentenceTransformers `multilingual-e5-base`.

     * Documents encoded as `"passage: <name> | <summary>"`.
     * Queries encoded as `"query: <question>"`.
   * **Image**: OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`).

     * Also encode **text queries** with CLIP for **text→image** retrieval.

3. **Indexing**

   * **FAISS (IP / cosine)** when available; **hnswlib** fallback.
   * Save to `rag_out/faiss_text.idx`, `rag_out/faiss_image.idx`, and a `docs.jsonl` map.

4. **Retrieval (Fusion)**

   * Top-k from **text index** + top-k from **text→image index**.
   * **Min-max normalize** each score list; **alpha-weighted** linear fusion (default `alpha=0.6`).
   * Return top results with attached metadata (name, summary, image path).

5. **Evaluation**

   * **Entity Hit@K**: does the correct person appear in top-K?
   * **Answer-string Hit@K**: does the gold option text appear in top-K contexts?
   * Additional CLIP **text↔image recall@K** diagnostics.

6. **Generative phase**

   * Build **compact contexts** (top-3) and ask the model to:

     * choose **one** option,
     * return **evidence span** **verbatim** from context,
     * (multimodal) optionally add a **short bio** for the predicted athlete.
   * Outputs **strict JSON** rows saved to CSV.

The report gives more details and qualitative analysis of text vs. image vs. fused retrieval, plus error cases and ablations. 

---

## Data & Expected Layout

```
/ (repo root)
├── Data/
│   ├── athlete_merged_with_bios_imaged_crawled.json        # consolidated catalog (names + bios + image info)
│   ├── Questions/
│   │   ├── mcq_questions_full_new.json                     # TEXT MCQs (question, options[], answer)
│   │   └── qa_with_image_new.json                          # MULTIMODAL MCQs (includes image_url per item)
│   └── Raw data/
│       ├── athlete17.json                                  # raw sources (examples)
│       └── athlete19.json
├── Results/                                                # all artifacts produced by the notebook
│   ├── Final embeddings/
│   │   ├── faiss_text.idx
│   │   ├── faiss_image.idx
│   │   ├── docs.jsonl
│   │   ├── clip_text.idx
│   │   └── clip_text_rows.json
│   ├── Retrieval_results/
│   │   ├── text_mcq_top3_results.csv
│   │   └── Multimodal_mcq_results_top3.csv
│   ├── Generative_results/
│   │   ├── Text_set_generative_outputs.csv
│   │   └── Multimodal_set_generative_outputs.csv
│   └── Final_evaluation/
│       ├── evaluation_RAG/                                 # final metrics & plots
│       │   ├── eval_summary.csv
│       │   ├── text_eval_detailed.csv
│       │   ├── multi_eval_detailed.csv
│       │   ├── accuracy_overview.png
│       │   ├── text_accuracy_by_category.png
│       │   └── multi_accuracy_by_category.png
│       ├── evaluation_baseline/                            # baseline metrics & plots
│       │   ├── baseline_accuracy_overview.png
│       │   ├── summary_text_baseline.csv
│       │   └── summary_multimodal_baseline.csv
│       └── evaluation_compare/                             # side-by-side comparisons
│           └── compare_accuracy_overview.png
├── Athlete_multimodal_RAG_Final.ipynb
├── NLP_HW3_Report_Final_Persian.pdf
├── README.md
└── LICENSE
```

---

## Quickstart (Google Colab)

1. **Open** `Athlete_multimodal_RAG_Final.ipynb` in Colab with a GPU runtime.
2. **Mount Drive** when prompted:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Adjust `BASE`** to your Drive path:

```python
from pathlib import Path
BASE = Path("/content/drive/MyDrive/NLP/HW3")
```

4. **Install deps** (uncomment the `%pip` cells if running fresh).
5. **Run cells top-to-bottom**. It will:

   * enrich catalog (images ↔ names),
   * download models (E5 + OpenCLIP),
   * build indexes (FAISS/HNSW),
   * run retrieval demos and MCQ scoring,
   * produce generative outputs & evaluation artifacts.

---

## Local Setup

### Requirements

* Python 3.10+
* CUDA-enabled GPU (recommended); CPU will be **very slow** for embedding.
* Pip packages (see notebook cells):

  * `torch`, `sentence-transformers`, `open-clip-torch`, `pillow`, `tqdm`, `huggingface_hub`
  * `faiss-cpu` (or install GPU FAISS if available), `hnswlib`, `numpy`, `pandas`, `matplotlib`

### (Optional) Conda environment

```bash
conda create -n athlete-rag python=3.10 -y
conda activate athlete-rag
pip install torch sentence-transformers open-clip-torch pillow tqdm huggingface_hub faiss-cpu hnswlib numpy pandas matplotlib
```

## Reproducible Pipeline

All steps are implemented inside **`Athlete_multimodal_RAG_Final.ipynb`**. Below is a map to what each block does and the main artifacts it writes.

### 1) Build Catalog + Embeddings + Indexes

* **Input:** `athlete_merged_with_bios_imaged_crawled.json`, `athlete_images/`
* **Output:**

  * `rag_out/faiss_text.idx` (or `.bin`), `rag_out/faiss_image.idx` (or `.bin`)
  * `rag_out/docs.jsonl`
* **Key components:**

  * **Name & text normalization** (Arabic→Persian letters, Eastern digits→ASCII, ZWNJ/RTL cleanup)
  * **Image mapping heuristics** (filename tokens, priority rules like `_crawled` or `_imageN`)
  * **Embeddings**:

    * E5 (text): `"passage: <name> | <summary>"`
    * CLIP (image) & CLIP (text) for text→image queries
  * **Indexes**: FAISS IP (cosine) with hnswlib fallback

### 2) Retrieval & Diagnostics

* **Fusion retrieval**:

  * `alpha`-weighted combination of normalized text and image scores (default 0.6)
* **Helper**: `search(query: str, k=5, alpha=0.6)` returns top-K doc dicts

### 3) MCQ (Text) — Retrieval Scoring + CSV

* **Input:** `mcq_questions_full.json` or `mcq_questions_full_new.json`
* **Output:** `rag_out/text_mcq_top3_results.csv`
* **Metrics:**

  * **Entity Hit@K** (did we retrieve the right person?)
  * **Answer-string Hit@K** (did the gold option text appear in top-K contexts?)

### 4) MCQ (Image) — Multimodal Retrieval

* **Input:** `qa_with_image_new.json`
* **Output:** `rag_out/Multimodal_mcq_results_top3.csv`
* **Procedure:**

  * For each **option**, find top text matches (E5), ensure the candidate doc has an **image** embedding.
  * Encode the **query image** with CLIP; compute cosine vs. candidate images; pick top-3.

### 5) Generative Phase (Text set)

* **Input:** text MCQs + retrieval CSV (`text_mcq_top3_results.csv`)
* **Output:** `rag_out/Text_set_generative_outputs.csv`
* **Behavior:**

  * Build compact contexts (top-3 bios), ask model to **return JSON only** with:

    ```json
    { "target_name": "...", "predicted_option": "...", "evidence_span": "..." }
    ```
  * Ensures `predicted_option` matches one of the four options (with normalization).

### 6) Generative Phase (Multimodal set)

* **Input:** multimodal MCQs + retrieval CSV (`Multimodal_mcq_results_top3.csv`)
* **Output:** `rag_out/Multimodal_set_generative_outputs.csv`
* **Behavior:**

  * Given `top1_name` (from retrieval) and its summary, the model responds (JSON only):

    ```json
    { "predicted_option": "...", "evidence": "...", "bio": "..." }
    ```
  * Also tries to locate a local **image** for `top1_name` to log alongside predictions.

### 7) Evaluation & Plots

* **Input:**

  * Text: `mcq_questions_full_new.json` + `Text_set_generative_outputs.csv`
  * Multimodal: `qa_with_image_new.json` + `Multimodal_set_generative_outputs.csv` + `Multimodal_mcq_results_top3.csv`
* **Output:** under `rag_out/evaluation/`

  * `text_eval_detailed.csv`, `multi_eval_detailed.csv`
  * `eval_summary.csv`
  * Confusion tables: `confusions_text.csv`, `confusions_multi.csv`
  * Plots: `accuracy_overview.png`, `text_accuracy_by_category.png`, `multi_accuracy_by_category.png`
* **Categories** (regex-based): who, where/birthplace, when/age, sport, team/club, which-option, other.

---

## Results (high-level)

* **Text MCQs retrieval** reached **Hit@1 ≈ 97.9%** and **Hit@3 = 100%** (entity detection), after improving data cleaning and image–name mapping. 
* Qualitatively, **text retrieval** tended to be more stable when bios were informative, while **image retrieval** could be **misled** by confounding or low-quality images; score-level **fusion** helped but still reflects the underlying data quality. 

> Exact accuracies for the **generative** phase depend on the model you use and will be emitted by the evaluation notebook (see `eval_summary.csv`).

---

## How to Use the Search Programmatically

Once indexes are built, you can import/reuse the retrieval in a small script/notebook cell:

```python
# assuming the notebook cells built these globals:
# text_model, clip_model, tokenizer, text_index, img_index, docs, img_row_to_doc
# and the 'search' function is defined

query = "قهرمان مشهور کشتی آزاد ایران"
hits = search(query, k=5, alpha=0.6)
for d in hits:
    print(d["name"], "— image:", bool(d.get("image_path")), "|", d.get("image_path",""))
```

**Adjusting fusion:** increase `alpha` toward **text** for text-heavy questions; decrease for more **visual** questions.

---

## Design Notes & Tricks

* **Persian normalization**

  * Arabic→Persian letters (`ي→ی`, `ك→ک`, …), normalize Unicode (NFKC), Eastern→ASCII digits, strip ZWNJ/RTL marks, squeeze punctuation/spaces.
  * Helps both entity matching and answer-string presence checks.

* **Image ↔ name pairing**

  * Robust filename heuristics and tokenization; prioritize `_crawled` or low `_imageN`; fuzzy matches with fallbacks.

* **Scoring & Fusion**

  * Normalize each modality’s scores (min-max); linear combination with `alpha`.
  * Prevents one modality’s scale from dominating.

* **HNSW fallback**

  * If FAISS isn’t available, hnswlib is automatically used; files are saved as `.bin` instead of `.idx`.

* **Strict JSON for LLM outputs**

  * The prompts demand **JSON only** (no prose). Post-processing ensures the predicted option matches exactly one of the four choices (with normalization).

---

## Limitations

* **Data quality matters**: incomplete/irregular bios, mixed language, or weak image–name alignment can reduce accuracy; some images are low-quality or off-topic. 
* **Multimodal fusion** can still be **fooled by images** (e.g., contextual artifacts), especially when bios are thin. 
* **GPU recommended**: embedding on CPU is slow.
* **Model availability**: OpenCLIP and E5 downloads require internet; generative step requires access to a compatible LLM endpoint.

---

## Troubleshooting

* **`assert torch.cuda.is_available()` fails**

  * Enable GPU (Colab: Runtime → Change runtime type → GPU) or remove the assert (runs slower on CPU).

* **FAISS import error**

  * The code automatically falls back to **hnswlib**. Ensure `hnswlib` is installed, or install `faiss-cpu`.

* **No images found for many athletes**

  * Check the `athlete_images/` folder and filename patterns. The pairing relies on sanitized name substrings; adjust the heuristics if your filenames differ.

* **Generative phase can't call API**

  * Ensure `OPENAI_API_KEY` (or vendor key) is set in env vars; replace the client init accordingly.

---

## Folder Structure

```
.
├── Data/
│   ├── Questions/
│   │   ├── mcq_questions_full_new.json
│   │   └── qa_with_image_new.json
│   ├── Raw data/
│   │   ├── athlete17.json
│   │   └── athlete19.json
│   └── athlete_merged_with_bios_imaged_crawled.json
├── Results/
│   ├── Final embeddings/               # faiss_text.idx, faiss_image.idx, docs.jsonl, clip_text.*
│   ├── Retrieval_results/              # text_mcq_top3_results.csv, Multimodal_mcq_top3_results.csv
│   ├── Generative_results/             # Text_set_generative_outputs.csv, Multimodal_set_generative_outputs.csv
│   └── Final_evaluation/
│       ├── evaluation_RAG/             # final CSVs + PNG plots
│       ├── evaluation_baseline/        # baseline CSVs + PNGs
│       └── evaluation_compare/         # comparison PNGs
├── Athlete_multimodal_RAG_Final.ipynb   # end-to-end pipeline (Colab-friendly)
├── NLP_HW3_Report_Final_Persian.pdf     # Persian report
├── README.md
└── LICENSE

```

---

## Model & Data Licenses

* **Text encoder:** `intfloat/multilingual-e5-base`
* **Image encoder:** OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`)
* **Indexes:** FAISS / hnswlib
* **Data:** ensure you have the right to use and redistribute the athlete bios and images in your environment. When in doubt, keep the dataset private.

> Please review each model/dataset’s license and terms before use.

---

## Citation

If you use this repo, please cite:

```bibtex
@software{iranian_athlete_multimodal_mcq_rag_2025,
  title        = {Iranian Athlete Multimodal MCQ RAG},
  author       = {Beyrami, Sina and Daneshgar, Sina and Zahiri, Elahe},
  year         = {2025},
  url          = {https://github.com/<your-org>/<your-repo>},
  note         = {Multimodal retrieval-augmented generation for Persian MCQs about athletes}
}
```

---

## Acknowledgments

* SentenceTransformers (E5), OpenCLIP, FAISS, hnswlib, Hugging Face Hub.
* Special thanks to the course staff and peers. The Persian report consolidates much of the qualitative analysis and ablations. 

---

### Final Notes

* This README mirrors the end-to-end notebook so you can **run and reproduce** results with your own dataset layout.
* For contributions, consider refactoring the notebook into a small Python package (modules for `data`, `embed`, `index`, `retrieve`, `eval`, and `generate`) with CLI entry points.

---

## Team

- **Sina Beyrami**
- **Sina Daneshgar**
- **Elahe Zahiri**
