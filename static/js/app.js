/* ── Helpers ──────────────────────────────────── */
const $ = s => document.querySelector(s);
const show = el => typeof el === 'string' ? $(el).style.display = 'block' : el.style.display = 'block';
const hide = el => typeof el === 'string' ? $(el).style.display = 'none' : el.style.display = 'none';

function showToast(msg, type) {
  var t = $('#toast');
  t.textContent = msg;
  t.className = 'toast toast-' + (type || 'success') + ' show';
  setTimeout(function() { t.classList.remove('show'); }, 3500);
}

/* ── Back to Top ──────────────────────────────── */
window.addEventListener('scroll', function() {
  var btn = $('#backToTop');
  if (window.scrollY > 400) btn.classList.add('visible');
  else btn.classList.remove('visible');
});

function init3DScroll() {
}

init3DScroll();

window.addEventListener('scroll', function() {
  var sections = ['resultBanner', 'navMetrics', 'navDetection', 'navCandidates', 'navCharts', 'navTiming', 'navCompare', 'navQuantum'];
  var current = '';
  sections.forEach(function(id) {
    var el = document.getElementById(id);
    if (el) {
      var rect = el.getBoundingClientRect();
      if (rect.top <= 150) current = id;
    }
  });
  document.querySelectorAll('.nav-btn').forEach(function(btn) {
    btn.classList.remove('active');
    if (btn.getAttribute('data-target') === current) btn.classList.add('active');
  });
});

$('#backToTop')?.addEventListener('click', function() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ── Interactive How-It-Works Cards ─────────── */
function initInteractiveSteps() {
  var cards = Array.from(document.querySelectorAll('.step-card'));
  var inspectorTitle = $('#stepInspectorTitle');
  var inspectorText = $('#stepInspectorText');
  var inspectorTech = $('#stepInspectorTech');
  if (!cards.length) return;

  function activateCard(card) {
    cards.forEach(function(c) {
      c.classList.remove('active');
      c.setAttribute('aria-pressed', 'false');
    });
    card.classList.add('active');
    card.setAttribute('aria-pressed', 'true');
    if (inspectorTitle) inspectorTitle.textContent = card.getAttribute('data-step-title') || 'Step';
    if (inspectorText) inspectorText.textContent = card.getAttribute('data-step-detail') || '';
    if (inspectorTech) inspectorTech.textContent = card.getAttribute('data-step-tech') || 'Pipeline';
  }

  cards.forEach(function(card, idx) {
    card.addEventListener('click', function() { activateCard(card); });
    card.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        activateCard(card);
      }
    });

    card.addEventListener('pointermove', function(e) {
      var rect = card.getBoundingClientRect();
      var px = (e.clientX - rect.left) / rect.width;
      var py = (e.clientY - rect.top) / rect.height;
      var rx = (0.5 - py) * 7;
      var ry = (px - 0.5) * 9;
      card.style.setProperty('--mx', (px * 100).toFixed(2) + '%');
      card.style.setProperty('--my', (py * 100).toFixed(2) + '%');
      card.style.setProperty('--rx', rx.toFixed(2) + 'deg');
      card.style.setProperty('--ry', ry.toFixed(2) + 'deg');
    });

    card.addEventListener('pointerleave', function() {
      card.style.setProperty('--mx', '50%');
      card.style.setProperty('--my', '50%');
      card.style.setProperty('--rx', '0deg');
      card.style.setProperty('--ry', '0deg');
    });

    if (idx === 0) activateCard(card);
  });
}
initInteractiveSteps();

/* ── Section Navigation Click Handler ─────────── */
document.querySelectorAll('.nav-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var targetId = this.getAttribute('data-target');
    var targetEl = document.getElementById(targetId);
    if (targetEl) {
         targetEl.scrollIntoView({ behavior: 'auto', block: 'start' });
      document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
      this.classList.add('active');
    }
  });
});

/* ── Device Badge ─────────────────────────────── */
fetch('/api/device').then(r => r.json()).then(d => {
  $('#deviceBadge').textContent = d.device || 'cpu';
}).catch(() => { $('#deviceBadge').textContent = 'cpu'; });

/* ── Upload Logic ─────────────────────────────── */
let sceneFile = null, targetFile = null;

function setupUpload(inputSel, cardSel, previewSel, onFile) {
  const input = $(inputSel), card = $(cardSel), preview = $(previewSel);
  card.addEventListener('click', () => input.click());
  card.addEventListener('dragover', e => { e.preventDefault(); card.style.borderColor = 'var(--accent)'; });
  card.addEventListener('dragleave', () => { card.style.borderColor = ''; });
  card.addEventListener('drop', e => {
    e.preventDefault(); card.style.borderColor = '';
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });
  function handleFile(file) {
    onFile(file);
    preview.src = URL.createObjectURL(file);
    card.classList.add('has-file');
    updateButton();
  }
}
setupUpload('#sceneInput',  '#sceneCard',  '#scenePreview',  f => sceneFile = f);
setupUpload('#targetInput', '#targetCard', '#targetPreview', f => targetFile = f);

function updateButton() { $('#analyzeBtn').disabled = !(sceneFile && targetFile); }

/* ── Loading Steps ─────────────────────────────── */
var loadingInterval = null;
var loadingStart = 0;
var loadingSteps = [
  { t: 0,    text: 'Uploading images...' },
  { t: 1500, text: 'Running YOLO detection...' },
  { t: 3500, text: 'Computing CLIP features...' },
  { t: 6000, text: 'Calculating similarity...' },
  { t: 8500, text: 'Running Grover\u2019s algorithm...' },
  { t: 11000,text: 'Generating visualizations...' },
];
function startLoadingSteps() {
  loadingStart = Date.now();
  $('#loadingSteps').classList.add('visible');
  loadingInterval = setInterval(function() {
    var elapsed = Date.now() - loadingStart;
    $('#loadingTimer').textContent = (elapsed / 1000).toFixed(1) + 's';
    for (var i = loadingSteps.length - 1; i >= 0; i--) {
      if (elapsed >= loadingSteps[i].t) {
        $('#loadingText').textContent = loadingSteps[i].text;
        break;
      }
    }
  }, 100);
}
function stopLoadingSteps() {
  clearInterval(loadingInterval);
  $('#loadingSteps').classList.remove('visible');
}

/* ── Analyze ──────────────────────────────────── */
$('#analyzeBtn').addEventListener('click', async () => {
  if (!sceneFile || !targetFile) return;
  const btn = $('#analyzeBtn');
  btn.disabled = true; btn.classList.add('loading'); btn.textContent = 'Analyzing\u2026';
  hide('#errorCard'); hide('#resultsSection');
  $('#resultsSection').classList.remove('visible');
  startLoadingSteps();

  const form = new FormData();
  form.append('scene', sceneFile);
  form.append('target', targetFile);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      showErrorCard(data);
      showToast(data.error, 'error');
    } else if (!res.ok) {
      throw new Error(data.detail || 'Analysis failed');
    } else {
      var elapsed = ((Date.now() - loadingStart) / 1000).toFixed(1);
      renderResults(data);
      showToast('Analysis complete in ' + elapsed + 's \u2014 ' + data.n_matches + ' match(es) found', 'success');
    }
  } catch (err) {
    $('#errorMessage').textContent = err.message;
    show('#errorCard');
    showToast('Analysis failed', 'error');
  } finally {
    stopLoadingSteps();
    btn.disabled = false; btn.classList.remove('loading'); btn.textContent = 'Analyze with Quantum';
  }
});

/* ── Error Card (pattern-absent) ──────────────── */
function showErrorCard(data) {
  var html = '<h4>' + data.error + '</h4>';
  if (data.rejection_reasons && data.rejection_reasons.length) {
    html += '<ul class="error-reasons">';
    for (var i = 0; i < data.rejection_reasons.length; i++) {
      html += '<li>' + data.rejection_reasons[i] + '</li>';
    }
    html += '</ul>';
  }
  if (data.diagnostics) {
    var d = data.diagnostics;
    html += '<div class="diag-mini"><h5>Diagnostics</h5><div class="diag-mini-grid">';
    var keys = Object.keys(d);
    for (var k = 0; k < keys.length; k++) {
      var label = keys[k].replace(/_/g, ' ');
      html += '<div class="diag-mini-item"><div class="diag-mini-label">' + label + '</div><div class="diag-mini-val">' + d[keys[k]] + '</div></div>';
    }
    html += '</div></div>';
  }
  $('#errorCard').innerHTML = html;
  show('#errorCard');
}

/* ── Reset ────────────────────────────────────── */
$('#resetBtn').addEventListener('click', function() {
  sceneFile = null; targetFile = null;
  $('#sceneInput').value = ''; $('#targetInput').value = '';
  $('#sceneCard').classList.remove('has-file'); $('#targetCard').classList.remove('has-file');
  hide('#resultsSection'); hide('#errorCard');
  $('#resultsSection').classList.remove('visible');
  $('#sectionNav').classList.remove('visible');
  updateButton();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  //showToast('Ready for new analysis', 'success');
});

/* ── Render Results ───────────────────────────── */
function renderResults(data) {
  const isMatch = data.n_matches > 0;

  // Banner
  $('#resultBanner').className = isMatch ? 'result-banner success' : 'result-banner warning';
  $('#resultBanner').innerHTML =
    '<div class="result-icon">' + (isMatch ? '\u2713' : '\u26A0') + '</div>' +
    '<div class="result-text">' +
      '<h2>' + (isMatch ? 'Pattern Match Found!' : 'No Strong Match') + '</h2>' +
      '<p>' + data.n_matches + ' match(es) detected using ' + data.detection_method + '</p>' +
    '</div>' +
    '<div class="result-meta">' +
      '<span class="result-tag">' + data.n_qubits + ' Qubits</span>' +
      '<span class="result-tag">' + (data.best_score * 100).toFixed(1) + '% Best</span>' +
    '</div>';

  // Stats
  $('#statsGrid').innerHTML =
    '<div class="stat-card"><div class="stat-value">' + data.n_candidates + '</div><div class="stat-label">Candidates Found</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + data.n_matches + '</div><div class="stat-label">Matches Detected</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + (data.best_score * 100).toFixed(1) + '%</div><div class="stat-label">Best Similarity</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + data.n_qubits + '</div><div class="stat-label">Grover Qubits</div></div>';

  // Confidence
  if (data.confidence !== undefined) {
    var pct = data.confidence * 100;
    var cls = pct >= 65 ? 'high' : pct >= 40 ? 'medium' : 'low';
    $('#confidenceSection').innerHTML =
      '<div class="confidence-wrap">' +
        '<div class="confidence-header">' +
          '<h4>Match Confidence</h4>' +
          '<span class="confidence-pct ' + cls + '">' + pct.toFixed(1) + '%</span>' +
        '</div>' +
        '<div class="confidence-track">' +
          '<div class="confidence-fill ' + cls + '" style="width:' + pct + '%"></div>' +
        '</div>' +
      '</div>';
  }

  // Gauges
  var gaugeR = 60;
  var circ = 2 * Math.PI * gaugeR;
  function gauge(pctVal, label, gcls) {
    var off = circ - (pctVal / 100) * circ;
    return '<div class="gauge-card"><div class="gauge-ring">' +
      '<svg viewBox="0 0 140 140">' +
        '<circle class="gauge-bg" cx="70" cy="70" r="' + gaugeR + '"/>' +
        '<circle class="gauge-fill ' + gcls + '" cx="70" cy="70" r="' + gaugeR + '" style="stroke-dasharray:' + circ + ';stroke-dashoffset:' + off + '"/>' +
      '</svg>' +
      '<div class="gauge-center">' +
        '<span class="gauge-percent">' + pctVal.toFixed(1) + '%</span>' +
        '<span class="gauge-sublabel">' + label + '</span>' +
      '</div></div></div>';
  }
  var g = gauge(data.best_score * 100, 'CLIP Similarity', 'green');
  if (data.edge_similarity !== undefined) g += gauge(data.edge_similarity * 100, 'Edge Structure', 'purple');
  if (data.color_similarity !== undefined) g += gauge(data.color_similarity * 100, 'Color Histogram', 'blue');
  $('#gaugesGrid').innerHTML = g;

  // Output
  $('#outputImg').src = data.output_image;
  var bestLabel = data.best_label ? data.best_label : (data.target_object_label ? data.target_object_label : 'object');
  $('#outputCaption').textContent = data.n_matches + ' match(es) found \u00B7 Label: ' + bestLabel + ' \u00B7 Detection: ' + data.detection_method;

  // Thumbnails
  var th = '';
  for (var i = 0; i < data.candidate_thumbs.length; i++) {
    var t = data.candidate_thumbs[i];
    var m = data.matched_indices.indexOf(t.index) !== -1;
    var labelText = t.label ? t.label : (data.target_object_label ? data.target_object_label : 'object');
    th += '<div class="thumb-card ' + (m ? 'match' : '') + '">' +
      '<img src="' + t.image + '" alt="C' + t.index + '" />' +
      '<div class="thumb-info"><div class="thumb-rank">#' + t.rank + '</div>' +
      '<div class="thumb-score">' + (t.score * 100).toFixed(1) + '%</div>' +
      '<div class="thumb-score" style="font-size:.72rem;opacity:.85;text-transform:capitalize">' + labelText + '</div></div></div>';
  }
  $('#thumbsRow').innerHTML = th;

  // Charts: show section only when chart images are provided by backend.
  var chartsTitle = $('#navCharts');
  var chartsNavBtn = document.querySelector('.nav-btn[data-target="navCharts"]');
  if (data.chart_grover && data.chart_similarity) {
    if (chartsTitle) chartsTitle.style.display = '';
    if (chartsNavBtn) chartsNavBtn.style.display = '';
    $('#chartsGrid').style.display = '';
    $('#chartsGrid').innerHTML =
      '<div class="chart-card"><h4>Grover Measurement Distribution</h4><img src="' + data.chart_grover + '" alt="Grover" /></div>' +
      '<div class="chart-card"><h4>CLIP Similarity Scores</h4><img src="' + data.chart_similarity + '" alt="Similarity" /></div>';
  } else {
    if (chartsTitle) chartsTitle.style.display = 'none';
    if (chartsNavBtn) chartsNavBtn.style.display = 'none';
    $('#chartsGrid').innerHTML = '';
    $('#chartsGrid').style.display = 'none';
  }

  // Timing
  if (data.timing) {
    var ti = data.timing;
    var fmt = function(ms) { return ms >= 1000 ? (ms/1000).toFixed(2)+'s' : Math.round(ms)+'ms'; };
    $('#timingGrid').innerHTML =
      '<div class="timing-card"><div class="timing-val">' + fmt(ti.detection_ms) + '</div><div class="timing-label">Detection</div></div>' +
      '<div class="timing-card"><div class="timing-val">' + fmt(ti.similarity_ms) + '</div><div class="timing-label">Similarity</div></div>' +
      '<div class="timing-card"><div class="timing-val">' + fmt(ti.quantum_ms) + '</div><div class="timing-label">Quantum</div></div>' +
      '<div class="timing-card"><div class="timing-val">' + fmt(ti.total_ms) + '</div><div class="timing-label">Total</div></div>';
  }

  // YOLO vs Grover Comparison
  var comparisonHtml = '';
  var cmp = data.comparison || null;
  var classicalSearchMs = cmp && cmp.classical_search_time_ms !== undefined
    ? cmp.classical_search_time_ms
    : (cmp && cmp.estimated_classical_search_ms !== undefined ? cmp.estimated_classical_search_ms : (data.timing ? data.timing.similarity_ms : null));
  var quantumSearchMs = cmp && cmp.quantum_search_time_ms !== undefined
    ? cmp.quantum_search_time_ms
    : (cmp && cmp.measured_grover_search_ms !== undefined ? cmp.measured_grover_search_ms : (data.timing ? data.timing.quantum_ms : null));
  var winner = cmp && cmp.search_winner
    ? cmp.search_winner
    : (classicalSearchMs !== null && quantumSearchMs !== null && quantumSearchMs <= classicalSearchMs ? 'Grover' : 'Classical');
  var winnerClass = winner === 'Grover' ? 'grover' : 'yolo';
  var fasterTag = '<span class="compare-pill grover">Grover shows quadratic speedup in search phase</span>';
  var measuredGapMs = cmp && cmp.measured_gap_ms !== undefined
    ? cmp.measured_gap_ms
    : (classicalSearchMs !== null && quantumSearchMs !== null ? Math.abs(classicalSearchMs - quantumSearchMs) : 0);
  var measuredRatioRaw = cmp && cmp.measured_speedup_ratio_classical_over_grover !== undefined
    ? cmp.measured_speedup_ratio_classical_over_grover
    : (cmp && cmp.measured_speedup_ratio_yolo_over_grover !== undefined
      ? cmp.measured_speedup_ratio_yolo_over_grover
      : (quantumSearchMs && quantumSearchMs > 0 && classicalSearchMs !== null ? (classicalSearchMs / quantumSearchMs) : null));
  var measuredRatio = measuredRatioRaw === null || measuredRatioRaw === undefined
    ? 'N/A'
    : Number(measuredRatioRaw).toFixed(3);
  var measuredGapText = Math.round(measuredGapMs) + 'ms';
  var classicalSearchTimeText = classicalSearchMs !== null ? fmt(classicalSearchMs) : 'N/A';
  var quantumSearchTimeText = quantumSearchMs !== null ? fmt(quantumSearchMs) : 'N/A';
  var runCandidates = (data.quantum_info && data.quantum_info.n_candidates !== undefined)
    ? data.quantum_info.n_candidates
    : (data.n_candidates !== undefined ? data.n_candidates : null);
  var classicalSteps = (cmp && cmp.classical_steps !== undefined)
    ? cmp.classical_steps
    : (runCandidates !== null ? runCandidates : 0);
  var groverSteps = (data.quantum_info && data.quantum_info.iterations !== undefined)
    ? data.quantum_info.iterations
    : (data.grover_iterations !== undefined
      ? data.grover_iterations
      : (cmp && cmp.grover_iterations !== undefined
        ? cmp.grover_iterations
        : (cmp && cmp.grover_steps !== undefined ? cmp.grover_steps : 0)));
  var speedupMsg = 'Grover shows quadratic speedup in search phase';

  comparisonHtml +=
    '<div class="compare-header">' +
      '<span class="compare-pill grover">' + speedupMsg + '</span>' +
      '<p class="compare-note">Measured Gap: ' + measuredGapText + ' | Speedup (Classical/Grover): ' + measuredRatio + 'x</p>' +
    '</div>' +
    '<table class="compare-table">' +
      '<thead><tr><th>ASPECT</th><th>Classical Search</th><th>Grover</th></tr></thead>' +
      '<tbody>' +
        '<tr><td>Primary Goal</td><td>Feature matching in search space</td><td>Feature matching using quantum amplitude amplification</td></tr>' +
        '<tr><td>Time Complexity</td><td>O(N)</td><td>O(sqrt(N))</td></tr>' +
        '<tr><td>Search Type</td><td>Sequential / Linear search</td><td>Quantum search</td></tr>' +
        '<tr><td>Iterations (Steps)</td><td>' + classicalSteps + '</td><td>' + groverSteps + '</td></tr>' +
        '<tr><td>Search time</td><td>' + classicalSearchTimeText + '</td><td>' + quantumSearchTimeText + '</td></tr>' +
        '<tr><td>Pipeline Role</td><td>Post-detection matching</td><td>Accelerated matching</td></tr>' +
        '<tr><td>Scalability</td><td>Slower for large datasets</td><td>Faster for large datasets</td></tr>' +
        '<tr><td>Efficiency</td><td>Low for large N</td><td>High due to quadratic speedup</td></tr>' +
        '<tr><td>Hardware</td><td>Classical computers</td><td>Quantum / Quantum simulator</td></tr>' +
        '<tr><td>Accuracy</td><td>Deterministic</td><td>Probabilistic (high success rate)</td></tr>' +
        '<tr><td>Use Case</td><td>Small-scale search</td><td>Large-scale pattern matching</td></tr>' +
        '<tr><td>Estimated Time</td><td>Higher</td><td>Lower</td></tr>' +
      '</tbody>' +
    '</table>';
  $('#compareSection').innerHTML = comparisonHtml;

  // Image Info
  if (data.image_info) {
    var ii = data.image_info;
    var ss = sceneFile ? URL.createObjectURL(sceneFile) : '';
    var ts = targetFile ? URL.createObjectURL(targetFile) : '';
    var fp = function(n) { return n >= 1e6 ? (n/1e6).toFixed(2)+' MP' : n.toLocaleString()+' px'; };
    $('#imageInfoGrid').innerHTML =
      '<div class="image-info-card">' +
        (ss ? '<img class="image-info-thumb" src="' + ss + '" alt="Scene"/>' : '') +
        '<div class="image-info-details"><h4>Scene Image</h4><p>' + ii.scene_width + ' \u00D7 ' + ii.scene_height + '<br/>' + fp(ii.scene_pixels) + '</p></div>' +
      '</div>' +
      '<div class="image-info-card">' +
        (ts ? '<img class="image-info-thumb" src="' + ts + '" alt="Target"/>' : '') +
        '<div class="image-info-details"><h4>Target Pattern</h4><p>' + ii.target_width + ' \u00D7 ' + ii.target_height + '<br/>' + fp(ii.target_pixels) + '</p></div>' +
      '</div>';
  }

  // Score Table
  if (data.similarity_scores && data.similarity_scores.length) {
    var sc = data.similarity_scores, mx = Math.max.apply(null, sc);
    var rk = sc.map(function(s, idx) { return {index: idx, score: s}; }).sort(function(a, b) { return b.score - a.score; });
    var rows = '';
    for (var r = 0; r < rk.length; r++) {
      var item = rk[r];
      var isM = data.matched_indices.indexOf(item.index) !== -1;
      var bp = (item.score / mx * 100).toFixed(1);
      rows += '<tr class="' + (isM ? 'match-row' : '') + '">' +
        '<td>' + (r + 1) + '</td>' +
        '<td>C' + item.index + (isM ? '<span class="match-badge">MATCH</span>' : '') + '</td>' +
        '<td style="font-family:\'JetBrains Mono\',monospace">' + (item.score * 100).toFixed(3) + '%</td>' +
        '<td style="width:30%"><div class="score-bar"><div class="score-bar-fill ' + (isM ? 'match' : '') + '" style="width:' + bp + '%"></div></div></td>' +
      '</tr>';
    }
    $('#scoreTableWrap').innerHTML = '<table class="score-table">' +
      '<thead><tr><th>#</th><th>Candidate</th><th>Score</th><th>Relative</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table>';
  }

  // Circuit
  if (data.chart_circuit) {
    $('#circuitCard').innerHTML = '<img src="' + data.chart_circuit + '" alt="Quantum Circuit"/>';
    show('#circuitSection');
  }

  // Quantum Info Badges
  var qi = data.quantum_info;
  $('#quantumInfo').innerHTML =
    '<div class="quantum-info">' +
      '<div class="quantum-badge"><div class="quantum-value">' + qi.n_candidates + '</div><div class="quantum-label">Candidates</div></div>' +
      '<div class="quantum-badge"><div class="quantum-value">' + qi.n_qubits + '</div><div class="quantum-label">Qubits</div></div>' +
      '<div class="quantum-badge"><div class="quantum-value">' + qi.state_space + '</div><div class="quantum-label">State Space</div></div>' +
      '<div class="quantum-badge"><div class="quantum-value" style="font-size:1rem">|' + qi.marked_state + '\u27E9</div><div class="quantum-label">Marked State</div></div>' +
      '<div class="quantum-badge"><div class="quantum-value">' + qi.iterations + '</div><div class="quantum-label">Iterations</div></div>' +
      '<div class="quantum-badge"><div class="quantum-value">' + qi.shots + '</div><div class="quantum-label">Shots</div></div>' +
    '</div>';

  // Accordions
  var diag = data.diagnostics;
  var topStatesHtml = '';
  for (var si = 0; si < qi.top_states.length; si++) {
    var st = qi.top_states[si];
    topStatesHtml += '<code>|' + st.state + '\u27E9</code>: ' + st.count + ' counts';
    if (si < qi.top_states.length - 1) topStatesHtml += '<br/>';
  }
  var diagHtml = '';
  var diagKeys = Object.keys(diag);
  for (var di = 0; di < diagKeys.length; di++) {
    var dk = diagKeys[di];
    var dv = diag[dk];
    if (dk === 'all_scores') {
      diagHtml += '<div class="qi-item"><span class="qi-label">' + dk + '</span><span class="qi-value">' + dv.map(function(s) { return s.toFixed(4); }).join(', ') + '</span></div>';
    } else {
      var dlabel = dk.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
      diagHtml += '<div class="qi-item"><span class="qi-label">' + dlabel + '</span><span class="qi-value">' + (typeof dv === 'number' ? dv.toFixed(4) : dv) + '</span></div>';
    }
  }
  $('#accordions').innerHTML =
    '<div class="accordion">' +
      '<div class="accordion-header" onclick="this.parentElement.classList.toggle(\'open\')">' +
        '<span>\uD83E\uDDEA Quantum State Details</span><span class="accordion-arrow">\u25BC</span>' +
      '</div>' +
      '<div class="accordion-body"><div class="accordion-body-inner">' +
        '<p style="font-weight:600;margin-bottom:10px;">Top Measured States</p>' +
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:.85rem;color:var(--muted);">' + topStatesHtml + '</p>' +
      '</div></div>' +
    '</div>' +
    '<div class="accordion">' +
      '<div class="accordion-header" onclick="this.parentElement.classList.toggle(\'open\')">' +
        '<span>\uD83D\uDD0D Pattern Diagnostics</span><span class="accordion-arrow">\u25BC</span>' +
      '</div>' +
      '<div class="accordion-body"><div class="accordion-body-inner">' +
        '<div class="qi-grid">' + diagHtml + '</div>' +
      '</div></div>' +
    '</div>';

  show('#resultsSection');
  $('#resultsSection').classList.add('visible');
  $('#resultsSection').scrollIntoView({ behavior: 'auto', block: 'start' });
  
  // Show section navigation
  $('#sectionNav').classList.add('visible');
}