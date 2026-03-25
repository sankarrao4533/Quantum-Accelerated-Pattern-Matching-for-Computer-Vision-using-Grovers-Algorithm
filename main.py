import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Quantum Pattern Matching", layout="wide")
st.title("Quantum Accelerated Pattern Matching for Computer Vision using Grover’s Algorithm")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# ===============================
# LOAD YOLO + CLIP
# ===============================
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8m.pt")  # Medium YOLO
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return yolo, clip_model, clip_processor

yolo_model, clip_model, clip_processor = load_models()

# ===============================
# CLIP FEATURE EXTRACTION
# ===============================
def extract_clip_features(img_crop):
    # Resize crop to 224x224 (no padding)
    pil_img = Image.fromarray(cv2.resize(img_crop, (224, 224)))
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Handle newer transformers versions
        if isinstance(features, torch.Tensor):
            vec = features.detach().cpu().numpy().flatten()
        else:
            # BaseModelOutputWithPooling object
            vec = features.pooler_output.detach().cpu().numpy().flatten()

    return vec / (np.linalg.norm(vec) + 1e-8)

# ===============================
# AE-QIP QUANTUM SIMILARITY
# ===============================
def ae_qip_similarity(v1, v2):
    cosine = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
    probability = (1 + cosine) / 2
    return probability

# ===============================
# GROVER ORACLE
# ===============================
def grover_oracle(qc, marked_state):
    n = len(marked_state)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)
    qc.h(n-1)
    if n > 1:
        qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)

# ===============================
# GROVER DIFFUSER
# ===============================
def diffuser(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    if n > 1:
        qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))

# ===============================
# FILE UPLOAD
# ===============================
scene_file = st.file_uploader("Upload Scene Image", ["jpg", "png", "jpeg"])
target_file = st.file_uploader("Upload Target Image", ["jpg", "png", "jpeg"])
backend = Aer.get_backend("qasm_simulator")

# ===============================
# MAIN EXECUTION
# ===============================
if scene_file and target_file:
    scene_img = np.array(Image.open(scene_file).convert("RGB"))
    target_img = np.array(Image.open(target_file).convert("RGB"))
    st.success("✅ Images Uploaded Successfully!")
    st.image([scene_img, target_img], caption=["Scene Image", "Target Image"], width=300)

    # -------------------------------
    # TARGET DETECTION
    # -------------------------------
    target_results = yolo_model(target_img)
    if len(target_results[0].boxes) == 0:
        st.error("❌ No object detected in Target Image!")
        st.stop()

    best_target_box = max(target_results[0].boxes, key=lambda b: b.conf[0])
    tx1, ty1, tx2, ty2 = map(int, best_target_box.xyxy[0])
    target_crop = target_img[ty1:ty2, tx1:tx2]
    target_class = int(best_target_box.cls[0])
    st.image(target_crop, caption="🎯 Target Object", width=250)
    st.write("Target Class =", yolo_model.names[target_class])

    # -------------------------------
    # SCENE DETECTION
    # -------------------------------
    scene_results = yolo_model(scene_img)
    candidates = []

    target_aspect = (tx2 - tx1) / (ty2 - ty1 + 1e-6)
    for box in scene_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id != target_class or conf < 0.1:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = scene_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        aspect = (x2 - x1) / (y2 - y1 + 1e-6)
        if abs(aspect - target_aspect) > 1.5:
            continue
        candidates.append((x1, y1, x2, y2, crop))

    n_candidates = len(candidates)
    if n_candidates == 0:
        st.error("❌ No same-class objects found in Scene!")
        st.stop()
    st.success(f"✅ Found {n_candidates} same-class candidates!")

    # -------------------------------
    # CLIP FEATURE EXTRACTION
    # -------------------------------
    candidate_feats = [extract_clip_features(c[-1]) for c in candidates]
    target_feat = extract_clip_features(target_crop)

    # -------------------------------
    # AE-QIP SIMILARITY
    # -------------------------------
    similarity_scores = [ae_qip_similarity(f, target_feat) for f in candidate_feats]
    st.subheader("📌 AE-QIP Quantum Similarity Scores")
    for i, score in enumerate(similarity_scores):
        st.write(f"Candidate {i}: {score:.4f}")

    best_classical = int(np.argmax(similarity_scores))
    st.success(f"✅ Best Match Before Grover = Candidate {best_classical}")

    # -------------------------------
    # GROVER SEARCH (optional, safe)
    # -------------------------------
    if n_candidates == 1:
        grover_index = best_classical
        counts = {format(best_classical, f'0{1}b'): 1024}
    elif (n_candidates & (n_candidates - 1)) == 0:  # only powers of 2
        n_qubits = int(np.log2(n_candidates))
        marked_state = format(best_classical, f"0{n_qubits}b")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        iterations = max(1, int(np.floor((np.pi/4) * np.sqrt(2**n_qubits))))
        for _ in range(iterations):
            grover_oracle(qc, marked_state)
            diffuser(qc, n_qubits)
        qc.measure_all()
        result = backend.run(transpile(qc, backend), shots=1024).result()
        counts = result.get_counts()
        best_state = max(counts, key=counts.get)
        grover_index = int(best_state, 2)
        if grover_index >= n_candidates:
            grover_index = best_classical
    else:
        # fallback to classical
        grover_index = best_classical
        counts = {format(best_classical, f'0{int(np.ceil(np.log2(n_candidates)))}b'): 1024}

    st.success(f"⚡ Grover Best Match Index = {grover_index}")

    # -------------------------------
    # FINAL VISUALIZATION
    # -------------------------------
    x1, y1, x2, y2, _ = candidates[grover_index]
    output_img = scene_img.copy()
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    st.image(output_img, caption="✅ Final Best Matching Object Found", use_column_width=True)

    # -------------------------------
    # GROVER HISTOGRAM
    # -------------------------------
    st.subheader("📊 Grover Measurement Histogram")
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_xlabel("Quantum State")
    ax.set_ylabel("Counts")
    st.pyplot(fig)