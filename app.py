import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models import load_models, DEVICE
from features import extract_clip_features
from similarity import ae_qip_similarity
from detection import detect_grid_tiles, uniform_grid_split
from quantum import run_grover_search

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Quantum Pattern Matching", layout="wide")
st.title("Quantum Accelerated Pattern Matching for Computer Vision using Grover's Algorithm")

st.write(f"Using device: {DEVICE}")

# ===============================
# LOAD MODELS (cached)
# ===============================
yolo_model, clip_model, clip_processor = load_models()

# ===============================
# FILE UPLOAD
# ===============================
scene_file = st.file_uploader("Upload Scene Image", ["jpg", "png", "jpeg"])
target_file = st.file_uploader("Upload Target Image", ["jpg", "png", "jpeg"])

# ===============================
# MAIN EXECUTION
# ===============================
if scene_file and target_file:
    scene_img = np.array(Image.open(scene_file).convert("RGB"))
    target_img = np.array(Image.open(target_file).convert("RGB"))
    st.success("✅ Images Uploaded Successfully!")
    st.image([scene_img], caption=["Scene Image",], width=300)

    # -------------------------------
    # TARGET: use full image (CLIP handles it well)
    # -------------------------------
    target_crop = target_img
    st.image(target_crop, caption="🎯 Target Pattern", width=250)

    # -------------------------------
    # SCENE: Hybrid detection strategy
    # 1) Try YOLO for real-world objects
    # 2) Try contour-based grid tile detection
    # 3) Fall back to uniform grid split
    # Pick whichever gives more meaningful candidates
    # -------------------------------
    st.info("🔍 Detecting candidate regions in scene...")

    # Strategy A: YOLO detection
    scene_results = yolo_model(scene_img, conf=0.15)
    yolo_candidates = []
    for box in scene_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = scene_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        yolo_candidates.append((x1, y1, x2, y2, crop))

    # Strategy B: Contour-based tile detection (for grids/mosaics)
    grid_candidates = detect_grid_tiles(scene_img)

    # Strategy C: Uniform grid split (fallback)
    # Estimate grid size from image aspect ratio
    h, w = scene_img.shape[:2]
    est_tiles = max(4, min(25, int(np.sqrt((w * h) / (200 * 200)))))
    est_rows = max(2, int(np.sqrt(est_tiles * h / (w + 1e-6))))
    est_cols = max(2, int(np.sqrt(est_tiles * w / (h + 1e-6))))
    uniform_candidates = uniform_grid_split(scene_img, est_rows, est_cols)

    # Choose the best strategy:
    # - If grid detection found >= 4 tiles, prefer it (pattern/mosaic image)
    # - If YOLO found good detections, use those
    # - Otherwise use uniform grid
    detection_method = ""
    if len(grid_candidates) >= 4:
        candidates = grid_candidates
        detection_method = "Grid Tile Detection (contour-based)"
    elif len(yolo_candidates) >= 2:
        candidates = yolo_candidates
        detection_method = "YOLO Object Detection"
    elif len(uniform_candidates) >= 4:
        candidates = uniform_candidates
        detection_method = f"Uniform Grid Split ({est_rows}×{est_cols})"
    elif len(yolo_candidates) >= 1:
        candidates = yolo_candidates
        detection_method = "YOLO Object Detection"
    else:
        # Last resort: 3x3 grid
        candidates = uniform_grid_split(scene_img, 3, 3)
        detection_method = "Uniform Grid Split (3×3 fallback)"

    n_candidates = len(candidates)
    if n_candidates == 0:
        st.error("❌ No candidate regions found in Scene!")
        st.stop()
    st.success(f"✅ Found {n_candidates} candidate(s) using **{detection_method}**")

    # -------------------------------
    # CLIP FEATURE EXTRACTION
    # -------------------------------
    candidate_feats = [extract_clip_features(c[-1], clip_model, clip_processor) for c in candidates]
    target_feat = extract_clip_features(target_crop, clip_model, clip_processor)

    # -------------------------------
    # NEGATIVE ANCHOR: Create synthetic non-matching references
    # to establish a "noise floor" for this specific scene.
    # This adapts to the actual images instead of using fixed thresholds.
    # -------------------------------
    # Anchor 1: Random noise image
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    noise_feat = extract_clip_features(noise_img, clip_model, clip_processor)
    # Anchor 2: Solid gray image
    gray_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    gray_feat = extract_clip_features(gray_img, clip_model, clip_processor)
    # Anchor 3: Full scene (pattern should localize better than whole scene)
    scene_feat = extract_clip_features(scene_img, clip_model, clip_processor)

    # Compute noise floor: how well do random/unrelated images score against candidates?
    noise_scores_to_candidates = [ae_qip_similarity(f, noise_feat) for f in candidate_feats]
    gray_scores_to_candidates = [ae_qip_similarity(f, gray_feat) for f in candidate_feats]
    noise_floor = max(max(noise_scores_to_candidates), max(gray_scores_to_candidates))

    # Also compute how similar the target is to the negative anchors
    target_vs_noise = ae_qip_similarity(target_feat, noise_feat)
    target_vs_gray = ae_qip_similarity(target_feat, gray_feat)

    # Cross-similarity: average similarity between ALL candidate pairs (without target)
    # This tells us the "natural similarity level" within the scene
    cross_sims = []
    for i in range(len(candidate_feats)):
        for j in range(i + 1, len(candidate_feats)):
            cross_sims.append(ae_qip_similarity(candidate_feats[i], candidate_feats[j]))
    avg_cross_similarity = np.mean(cross_sims) if cross_sims else 0.5

    # -------------------------------
    # AE-QIP SIMILARITY
    # -------------------------------
    similarity_scores = [ae_qip_similarity(f, target_feat) for f in candidate_feats]

    # Show ranked candidates with thumbnails
    st.subheader("📌 AE-QIP Quantum Similarity Scores")
    ranked_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
    # Show top candidates as thumbnails
    top_n = min(8, len(ranked_indices))
    cols = st.columns(min(4, top_n))
    for rank in range(top_n):
        idx = ranked_indices[rank]
        with cols[rank % len(cols)]:
            st.image(candidates[idx][-1], caption=f"#{rank+1} Score: {similarity_scores[idx]:.4f}", width=150)

    best_classical = int(np.argmax(similarity_scores))

    # ------------------------------------
    # ADAPTIVE PATTERN PRESENCE DETECTION
    # Uses relative comparisons against
    # known non-matches instead of fixed
    # thresholds.
    # ------------------------------------
    best_score = similarity_scores[best_classical]

    # Raw cosine similarities
    raw_cosines = [np.dot(f, target_feat) for f in candidate_feats]
    best_raw_cosine = raw_cosines[best_classical]

    # Scene baseline
    scene_baseline = ae_qip_similarity(scene_feat, target_feat)
    baseline_gap = best_score - scene_baseline

    # Margin above noise floor
    noise_margin = best_score - noise_floor

    # Margin above cross-similarity (how much better is best match vs scene self-similarity)
    cross_margin = best_score - avg_cross_similarity

    # Statistical separation
    if len(similarity_scores) > 1:
        other_scores = [s for i, s in enumerate(similarity_scores) if i != best_classical]
        mean_others = np.mean(other_scores)
        std_others = np.std(other_scores) + 1e-8
        separation = (best_score - mean_others) / std_others
        score_gap = best_score - mean_others
    else:
        mean_others = 0.0
        separation = 10.0
        score_gap = best_score

    # --- CLIP FEATURE VECTORS DISPLAY ---
    with st.expander("🔢 CLIP Feature Vectors", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Target Vector")
            st.write(target_feat)
        with col2:
            st.subheader("📍 Best Candidate Vector")
            st.write(candidate_feats[best_classical])

    # --- Always show diagnostic scores ---
    with st.expander("🔍 Pattern Detection Diagnostics", expanded=False):
        st.write(f"- **Best AE-QIP score**: {best_score:.4f}")
        st.write(f"- **Best raw cosine**: {best_raw_cosine:.4f}")
        st.write(f"- **Noise floor**: {noise_floor:.4f}")
        st.write(f"- **Noise margin (best − noise floor)**: {noise_margin:.4f}")
        st.write(f"- **Avg cross-similarity (within scene)**: {avg_cross_similarity:.4f}")
        st.write(f"- **Cross margin (best − avg cross)**: {cross_margin:.4f}")
        st.write(f"- **Scene baseline**: {scene_baseline:.4f}")
        st.write(f"- **Baseline gap**: {baseline_gap:.4f}")
        if len(similarity_scores) > 1:
            st.write(f"- **Mean others**: {mean_others:.4f}")
            st.write(f"- **Score gap**: {score_gap:.4f}")
            st.write(f"- **Z-score separation**: {separation:.2f}")
        st.write(f"- **All scores**: {[f'{s:.4f}' for s in similarity_scores]}")

    # --- ADAPTIVE Decision Logic ---
    # Instead of fixed thresholds, compare against computed baselines.
    pattern_absent = False
    rejection_reasons = []

    # (a) Best score barely above noise floor
    #     If a random noise image scores almost as well, the match is meaningless
    if noise_margin < 0.05:
        pattern_absent = True
        rejection_reasons.append(
            f"Best score ({best_score:.4f}) barely above noise floor "
            f"({noise_floor:.4f}), margin: {noise_margin:.4f} < 0.05"
        )

    # (b) Best score not better than scene self-similarity
    #     If candidates are as similar to the target as they are to each other,
    #     the target isn't adding any discriminative info
    if cross_margin < 0.03 and len(cross_sims) > 0:
        pattern_absent = True
        rejection_reasons.append(
            f"Best score ({best_score:.4f}) not above scene self-similarity "
            f"({avg_cross_similarity:.4f}), margin: {cross_margin:.4f} < 0.03"
        )

    # (c) No localization: best candidate not better than full scene
    if baseline_gap < 0.02:
        pattern_absent = True
        rejection_reasons.append(
            f"No localization — best candidate ({best_score:.4f}) ≈ full scene "
            f"({scene_baseline:.4f}), gap: {baseline_gap:.4f} < 0.02"
        )

    # (d) No statistical separation among candidates
    #     If the pattern IS present, the matching region should clearly stand out
    if len(similarity_scores) > 2 and separation < 2.0 and noise_margin < 0.10:
        pattern_absent = True
        rejection_reasons.append(
            f"No standout candidate (z-score: {separation:.2f} < 2.0) "
            f"and weak noise margin ({noise_margin:.4f} < 0.10)"
        )

    # (e) All candidates score similarly to the target — no discrimination
    if len(similarity_scores) > 2:
        score_range = max(similarity_scores) - min(similarity_scores)
        if score_range < 0.02 and noise_margin < 0.08:
            pattern_absent = True
            rejection_reasons.append(
                f"All candidates nearly identical scores (range: {score_range:.4f} < 0.02), "
                f"weak noise margin ({noise_margin:.4f} < 0.08)"
            )

    if pattern_absent:
        st.error(
            f"❌ **Pattern NOT found in the scene!**\n\n"
            f"The uploaded target pattern does not match any region in the scene image."
        )
        with st.expander("📋 Rejection Details", expanded=True):
            for r in rejection_reasons:
                st.write(f"  - {r}")
        st.warning("⚠️ Please upload a valid target pattern that is actually present in the scene.")
        st.stop()

    st.success(f"✅ Best Match (CLIP Similarity) = Candidate {best_classical} "
               f"(score: {similarity_scores[best_classical]:.4f})")

    # -------------------------------
    # MULTI-MATCH SELECTION
    # Keep only candidates within 2% of best similarity.
    # If difference is > 2%, reject/do not highlight.
    # -------------------------------
    diff_threshold_pct = 0.5
    diff_threshold = diff_threshold_pct / 100.0

    ranked_by_score = sorted(
        range(len(similarity_scores)),
        key=lambda i: similarity_scores[i],
        reverse=True
    )
    best_idx = ranked_by_score[0]
    second_idx = ranked_by_score[1] if len(ranked_by_score) > 1 else None

    best_score_pct = similarity_scores[best_idx] * 100.0
    second_score_pct = similarity_scores[second_idx] * 100.0 if second_idx is not None else None
    top_gap_pct = (best_score_pct - second_score_pct) if second_score_pct is not None else 100.0

    # Rule requested:
    # - If top-vs-second gap > 2 points: keep only highest score match.
    # - Else: keep all candidates within 2 points of highest score.
    if top_gap_pct > diff_threshold_pct:
        matched_indices = [best_idx]
    else:
        matched_indices = [
            i for i, score in enumerate(similarity_scores)
            if ((best_score - score) * 100.0) <= diff_threshold_pct
        ]

    match_threshold = best_score - diff_threshold

    # Safety fallback: always keep at least the best match
    if not matched_indices:
        matched_indices = [best_classical]

    st.success(
        f"✅ Detected {len(matched_indices)} matching region(s) "
        f"(within {diff_threshold * 100:.0f}% of best score)"
    )
    st.caption(
        f"Top gap = {top_gap_pct:.2f} percentage points; "
        f"rule: {'single best only' if top_gap_pct > diff_threshold_pct else 'highlight similar scores'}"
    )

    # -------------------------------
    # GROVER SEARCH
    # -------------------------------
    shots = 1024
    grover_index, counts, qc, n_qubits, marked_state, iterations = run_grover_search(
        n_candidates, best_classical, shots=shots
    )

    st.success(f"⚡ Grover Top Match Index = {grover_index}")

    st.info(
        f"**Quantum Confirmation Flow:**\n"
        f"1️⃣ **CLIP Similarity Stage** identified Candidate **{best_classical}** as the best match "
        f"(score: {similarity_scores[best_classical]:.4f})\n"
        f"2️⃣ **Grover's Algorithm** now uses this candidate as the oracle target to confirm/amplify the choice\n"
        f"3️⃣ The circuit below shows the quantum confirmation circuit with state `{marked_state}` marked."
    )

    # st.subheader("🧠 Quantum Circuit Used for State Generation")
    # if grover_circuit_fig is not None:
    #     st.pyplot(grover_circuit_fig, use_container_width=True)
    # if grover_circuit_text is not None:
    #     st.code(grover_circuit_text, language="text")
    # if n_candidates != 2**n_qubits:
    #     st.caption(
    #         f"Circuit used {2**n_qubits} quantum states for {n_candidates} candidates; "
    #         "out-of-range states are ignored during final index selection."
    #     )
    # st.caption(
    #     "This circuit is a **confirmation step**, not an independent search. "
    #     "The marked state was determined by CLIP similarity analysis."
    # )

    # -------------------------------
    # STATE GENERATION EXPLAINER
    # -------------------------------
    with st.expander("🧪 How Quantum States Are Obtained", expanded=False):
        st.markdown("""
        1. Initialize all qubits in $|0\rangle$.
        2. Apply Hadamard to all qubits to create superposition over all basis states.
        3. Use Grover oracle to phase-mark the selected state.
        4. Apply diffuser to amplify probability of marked state.
        5. Repeat oracle + diffuser for configured Grover iterations.
        6. Measure all qubits for multiple shots to obtain state counts.
        """)

        st.write(f"- **Candidates in scene**: {n_candidates}")
        st.write(f"- **Qubits used**: {n_qubits}")
        st.write(f"- **State space size**: {2**n_qubits}")
        st.write(f"- **Marked state (from best similarity)**: `{marked_state}`")
        st.write(f"- **Grover iterations**: {iterations}")
        st.write(f"- **Shots**: {shots}")

        sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_states = sorted_states[:min(8, len(sorted_states))]
        st.write("- **Top measured states**:")
        for state, cnt in top_states:
            st.write(f"  - `{state}`: {cnt} counts")

    # -------------------------------
    # FINAL VISUALIZATION
    # -------------------------------
    output_img = scene_img.copy()

    # Draw all candidates in light gray
    for i, (cx1, cy1, cx2, cy2, _) in enumerate(candidates):
        if i in matched_indices:
            continue
        cv2.rectangle(output_img, (cx1, cy1), (cx2, cy2), (100, 100, 100), 1)

    # Draw all matched regions in green
    for match_rank, idx in enumerate(matched_indices, start=1):
        x1, y1, x2, y2, _ = candidates[idx]
        thickness = 4 if idx == grover_index else 3
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            output_img,
            f"MATCH {match_rank}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    st.image(output_img, caption=f"✅ Final Matches: {len(matched_indices)}", use_container_width=True)

    # -------------------------------
    # REFINED VISUALIZATIONS
    # -------------------------------
    st.subheader("📊 Analysis Results")

    # --- Chart 1: Grover Measurement Histogram ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    states = list(counts.keys())
    count_vals = list(counts.values())
    colors_grover = ['#2ecc71' if s == format(grover_index, f'0{len(states[0])}b') else '#3498db'
                     for s in states]
    bars = ax1.bar(states, count_vals, color=colors_grover, edgecolor='white', linewidth=0.8)
    # Annotate bars with count values
    for bar, val in zip(bars, count_vals):
        if val > max(count_vals) * 0.05:  # only label visible bars
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(count_vals) * 0.02,
                     str(val), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    ax1.set_xlabel("Quantum State", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Measurement Counts", fontsize=11, fontweight='bold')
    ax1.set_title("Grover's Algorithm — Measurement Distribution", fontsize=13, fontweight='bold', pad=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', rotation=45 if len(states) > 8 else 0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    fig1.tight_layout()
    st.pyplot(fig1)

    # --- Chart 2: Candidate Similarity Scores ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cand_labels = [f"C{i}" for i in range(len(similarity_scores))]
    display_threshold = match_threshold
    colors_sim = ['#2ecc71' if i in matched_indices else '#e74c3c' if similarity_scores[i] < display_threshold
                  else '#3498db' for i in range(len(similarity_scores))]
    bars2 = ax2.barh(cand_labels, similarity_scores, color=colors_sim, edgecolor='white', linewidth=0.8)
    # Annotate with score values
    for bar, score in zip(bars2, similarity_scores):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", ha='left', va='center', fontsize=9, fontweight='bold', color='#2c3e50')
    ax2.axvline(x=noise_floor, color='#95a5a6', linestyle=':', linewidth=1.2, label=f'Noise Floor ({noise_floor:.3f})')
    ax2.axvline(x=display_threshold, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Min Match ({display_threshold:.3f})')
    ax2.set_xlabel("Similarity Score", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Candidate Region", fontsize=11, fontweight='bold')
    ax2.set_title("CLIP Similarity Scores per Candidate Region", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlim(0, min(1.05, max(similarity_scores) + 0.08))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    fig2.tight_layout()
    st.pyplot(fig2)

    plt.close('all')