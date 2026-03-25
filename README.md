<div align="center">

# Quantum Accelerated Pattern Matching for Computer Vision using Grover's Algorithm

### Signal-Driven Vision Search with Quantum Amplification

A hybrid classical plus quantum computer vision pipeline that combines CLIP embeddings, YOLOv8 candidate discovery, and Grover search to localize a target pattern inside complex scenes with explainable confidence.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=ibm&logoColor=white)](https://qiskit.org)
[![CLIP](https://img.shields.io/badge/OpenAI_CLIP-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/clip)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

</div>

---

## Overview

Given a scene image and a target image, this project finds where the target appears through a five-stage fusion workflow:

> Detect -> Encode -> Compare -> Amplify -> Explain

| Stage | Description |
|:---:|---|
| Detection | Generates candidate regions with YOLO, contour-aware tiling, or uniform splitting |
| Feature Encoding | Extracts CLIP vision embeddings (ViT-B/32) for candidates and target |
| Similarity Modeling | Scores each candidate with AE-QIP cosine probability, edge structure, and color histogram affinity |
| Quantum Amplification | Runs Grover's algorithm on a simulator to amplify the best candidate index |
| Result Synthesis | Returns annotated image outputs, per-candidate diagnostics, and visualization charts |

---

## Architecture

```text
Scene + Target
      |
      v
[Candidate Detection]
      |
      v
[CLIP Feature Extraction]
      |
      v
[Hybrid Similarity Scoring]
      |
      v
[Grover Quantum Search]
      |
      v
Outputs: Match Index, Confidence, Charts, Annotated Scene
```

---

## Why This Is Different

- Uses semantic embeddings and structural visual cues together, not just one metric.
- Separates candidate generation from search, making the search stage replaceable and measurable.
- Includes both practical runtime and theoretical complexity reporting for fair classical vs quantum interpretation.
- Produces full diagnostics artifacts instead of a single black-box prediction.

---

## Project Structure

```text
Quantum-Accelerated-Pattern-Matching/
|
+-- server.py            # FastAPI API endpoints and analysis orchestration
+-- app.py               # Streamlit interface
+-- main.py              # Legacy monolithic pipeline
+-- requirements.txt     # Dependency list
|
+-- models.py            # CLIP and YOLO model loading with caching
+-- features.py          # Embedding extraction logic
+-- similarity.py        # AE-QIP, edge, and color similarity metrics
+-- detection.py         # Candidate segmentation strategies
+-- quantum.py           # Grover oracle, diffuser, and simulator execution
|
+-- static/
|   +-- index.html       # Frontend shell
|   +-- css/
|   |   +-- styles.css   # Main styles
|   +-- js/
|       +-- app.js       # UI logic, rendering, API integration
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, CPU fallback supported)

### Installation

```bash
git clone https://github.com/sankarrao4533/Quantum-Accelerated-Pattern-Matching.git
cd "Quantum Accelerated Pattern Matching for Computer Vision using Grover's Algorithm"

python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

YOLOv8 weights are loaded automatically if missing on first run.

### Run

```bash
python server.py
```

Open http://localhost:8000 and run analysis with a scene-target pair.

<details>
<summary>Alternative UI (Streamlit)</summary>

```bash
streamlit run app.py
```

</details>

---

## Pipeline Details

### Detection Modes

| Mode | Technique | Typical Usage |
|:---|:---|:---|
| Grid Tiles | Adaptive threshold plus contours | Structured, mosaic-like scenes |
| YOLOv8 | Object-aware detection | Natural multi-object images |
| Uniform Grid | Fixed 4x4 split fallback | Safety path when detections are sparse |

### Similarity Terms

| Term | Meaning |
|:---|:---|
| AE-QIP Similarity | Cosine similarity mapped to probability: $(1 + \cos\theta)/2$ |
| Edge Similarity | Canny edge maps followed by cosine matching |
| Color Similarity | HSV histogram correlation using OpenCV |

### Confidence Equation

$$
\text{Confidence} = 0.50\cdot\text{CLIP} + 0.25\cdot\text{Edge} + 0.25\cdot\text{Color}
$$

### Grover Search Setup

- Oracle marks the classically selected best candidate.
- Diffuser performs inversion about the mean.
- Iteration count follows: $\left\lfloor\frac{\pi}{4}\sqrt{2^n}\right\rfloor$.
- Execution uses Qiskit Aer qasm_simulator at 1024 shots.

### Target-Absence Logic

The system can classify no-match scenarios using:

- Score noise-floor checks
- Inter-candidate separation patterns
- Baseline gap thresholds
- Z-score significance tests

---

## Classical vs Quantum Role Split

YOLO and quantum.py are complementary modules in a staged pipeline, not mutually exclusive alternatives.

| Aspect | Classical (YOLO/Search) | Quantum (Grover Module) |
|:---|:---|:---|
| Main job | Find candidate regions, then evaluate matches | Amplify probability of best candidate index |
| Input type | Pixel-space detections and feature vectors | Candidate-space index search |
| Complexity view | $O(N)$ search effort | $O(\sqrt{N})$ search effort |
| Real runtime | Often lower for small candidate sets | May include simulator overhead |
| Scaling behavior | Degrades linearly with candidate count | Improves search scaling with larger spaces |

### Reporting Guidance

- Report measured runtime winner from experiment timings.
- Report search complexity winner from asymptotic analysis.
- Present both together to avoid misleading conclusions.

Example:

- If measured times are YOLO 75 ms vs Grover 125 ms, measured winner is YOLO.
- Complexity advantage still belongs to Grover for unstructured search.

---

## Results Snapshot

Observed in this implementation:

- Measured gap: 826 ms
- Speedup ratio (Classical/Grover): 3.549x

| Aspect | Classical Search | Grover Search |
|:---|:---|:---|
| Time Complexity | O(N) | O(sqrt(N)) |
| Search Steps | 14 | 3 |
| Search Time | 1.15 s | 324 ms |
| Scalability | Slower as N grows | Faster growth profile |
| Accuracy Mode | Deterministic | Probabilistic (high confidence) |

---

## API

### POST /api/analyze

Accepts multipart scene and target images.

Returns matched index, confidence breakdown, candidate scores, annotated output image, quantum histogram, similarity chart, circuit view, and timing diagnostics.

### GET /api/device

Returns active compute backend (cuda or cpu).

---

## Tech Stack

<div align="center">

| Layer | Technology |
|:---:|:---|
| Vision Embeddings | OpenAI CLIP (ViT-B/32) |
| Object Detection | Ultralytics YOLOv8 |
| Quantum Search | Qiskit + Aer Simulator |
| Image Processing | OpenCV + Pillow |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Alternate UI | Streamlit |
| Numerical Compute | PyTorch + NumPy + SciPy |
| Visualization | Matplotlib |

</div>
