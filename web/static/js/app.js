// ── State ─────────────────────────────────────────────────────
let currentDataset = null;
let figureData = null;          // original Plotly figure JSON
let originalNodeColors = [];    // to restore after clear
let selectedNodes = [];
let ctrlDown = false;
let currentRotationDeg = 0;     // cumulative rotation angle

const SELECTION_COLOR = '#ff6b35';

// Chemical filter state
let allChemicals = [];
let activeFilters = [];
let filterMode = 'any';   // 'any' = OR logic, 'all' = AND logic
let filterDebounceTimer = null;

// Timeline state
let timelineDataMin = null;
let timelineDataMax = null;
let timelineMinYear = null;
let timelineMaxYear = null;
let timelineDebounceTimer = null;

const SUBCATEGORY_ORDER = [
    "True Substance", "Substitute", "Adulterant", "Diluent",
    "Impurity", "Contaminants", "Fatty acids", "Unknown"
];

// ── Ctrl key tracking ─────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (e.key === 'Control') ctrlDown = true;
});
document.addEventListener('keyup', e => {
    if (e.key === 'Control') ctrlDown = false;
});

// ── Analysis presets ──────────────────────────────────────────
const PRESETS = {
    survey: {
        cosine: 0.15, jaccard: 0.55, euclidean: 0.15, tfidf: 0.15, pearson: 0.00,
        threshold: 0.45,
    },
    batch: {
        cosine: 0.35, jaccard: 0.15, euclidean: 0.35, tfidf: 0.15, pearson: 0.00,
        threshold: 0.70,
    },
    trace: {
        cosine: 0.10, jaccard: 0.20, euclidean: 0.10, tfidf: 0.60, pearson: 0.00,
        threshold: 0.50,
    },
    equal: {
        cosine: 0.25, jaccard: 0.25, euclidean: 0.25, tfidf: 0.25, pearson: 0.00,
        threshold: 0.50,
    },
};

function applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    _setSlider('w-cosine',    'w-cosine-val',    p.cosine);
    _setSlider('w-jaccard',   'w-jaccard-val',   p.jaccard);
    _setSlider('w-euclidean', 'w-euclidean-val', p.euclidean);
    _setSlider('w-tfidf',     'w-tfidf-val',     p.tfidf);
    _setSlider('w-pearson',   'w-pearson-val',   p.pearson ?? 0);
    _setSlider('threshold',   'threshold-val',   p.threshold);
    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('preset-' + key);
    if (btn) btn.classList.add('active');
    computeGraph();
}

function clearActivePreset() {
    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
}

function _setSlider(id, valId, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value;
    document.getElementById(valId).textContent = value.toFixed(2);
}

// ── Slider labels ─────────────────────────────────────────────
function updateSliderValue(input, spanId) {
    document.getElementById(spanId).textContent = parseFloat(input.value).toFixed(2);
}

// ── Loading overlay ───────────────────────────────────────────
function showLoading() {
    const container = document.getElementById('graph-container');
    let ov = document.getElementById('loading-overlay');
    if (!ov) {
        ov = document.createElement('div');
        ov.id = 'loading-overlay';
        ov.className = 'loading-overlay';
        ov.innerHTML = '<div class="spinner"></div>';
        container.parentElement.appendChild(ov);
    }
    ov.style.display = 'flex';
}

function hideLoading() {
    const ov = document.getElementById('loading-overlay');
    if (ov) ov.style.display = 'none';
}

// ── Attach click handler (must re-attach after every Plotly op) ──
function attachClickHandler() {
    const el = document.getElementById('plotly-graph');
    if (!el) return;
    el.removeAllListeners('plotly_click');
    el.on('plotly_click', handleGraphClick);
}

// ── Load dataset ──────────────────────────────────────────────
async function loadDataset() {
    const dataset = document.getElementById('dataset-select').value;
    if (!dataset) return;

    showLoading();
    closeNodePanel();
    clearSelectionSilent();
    clearFiltersSilent();
    resetTimeline();

    try {
        const resp = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        currentDataset = dataset;
        document.getElementById('header-dataset').textContent = dataset;
        document.getElementById('controls-section').style.display = 'block';

        renderGraph(data.graph);
        updateStats(data.stats);
        loadChemicals();
        if (data.year_range) {
            initTimeline(data.year_range.min, data.year_range.max);
        }
    } catch (err) {
        alert('Failed to load dataset: ' + err.message);
    } finally {
        hideLoading();
    }
}

// ── Compute graph ─────────────────────────────────────────────
async function computeGraph() {
    if (!currentDataset) return;

    showLoading();
    closeNodePanel();
    clearSelectionSilent();

    const isFullRange = timelineDataMin === null
        || (timelineMinYear === timelineDataMin && timelineMaxYear === timelineDataMax);

    const payload = {
        weight_cosine:    parseFloat(document.getElementById('w-cosine').value),
        weight_jaccard:   parseFloat(document.getElementById('w-jaccard').value),
        weight_euclidean: parseFloat(document.getElementById('w-euclidean').value),
        weight_tfidf:     parseFloat(document.getElementById('w-tfidf').value),
        weight_pearson:   parseFloat(document.getElementById('w-pearson').value),
        threshold:        parseFloat(document.getElementById('threshold').value),
        sqrt_pretreatment: document.getElementById('sqrt-pretreatment').checked,
        filter_chemicals: [...activeFilters],
        filter_mode:      filterMode,
        year_min:         isFullRange ? null : timelineMinYear,
        year_max:         isFullRange ? null : timelineMaxYear,
    };

    try {
        const resp = await fetch('/api/compute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        renderGraph(data.graph);
        updateStats(data.stats);
    } catch (err) {
        alert('Compute failed: ' + err.message);
    } finally {
        hideLoading();
    }
}

// ── Render Plotly graph ───────────────────────────────────────
function renderGraph(figure) {
    figureData = figure;
    currentRotationDeg = 0;

    const container = document.getElementById('graph-container');
    container.innerHTML = '<div id="plotly-graph" style="width:100%;height:100%"></div>';

    // Store original node colours for restoring after clear
    const nodeTrace = figure.data[1];
    if (nodeTrace && nodeTrace.marker) {
        originalNodeColors = [...(nodeTrace.marker.color || [])];
    }

    const rotateLeftBtn = {
        name: 'Rotate left 15°',
        title: 'Rotate left',
        icon: {
            width: 24, height: 24,
            path: 'M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z',
        },
        click: () => rotateGraph(-15),
    };

    const rotateRightBtn = {
        name: 'Rotate right 15°',
        title: 'Rotate right',
        icon: {
            width: 24, height: 24,
            path: 'M12 4V1l4 4-4 4V6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z',
        },
        click: () => rotateGraph(15),
    };

    Plotly.newPlot('plotly-graph', figure.data, figure.layout, {
        responsive: true,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        modeBarButtonsToAdd: [rotateLeftBtn, rotateRightBtn],
        displaylogo: false,
    });

    attachClickHandler();
}

// ── Graph click handler ───────────────────────────────────────
function handleGraphClick(eventData) {
    if (!eventData.points || !eventData.points.length) return;
    const point = eventData.points[0];

    // Only handle node trace (index 1, not edge trace index 0)
    if (point.curveNumber !== 1) return;

    const nodeName = point.customdata;
    if (!nodeName) return;

    if (ctrlDown) {
        toggleSelection(nodeName);
    } else {
        clearSelectionSilent();
        showNodePanel(nodeName);
    }
}

// ── Node detail panel ─────────────────────────────────────────
async function showNodePanel(nodeName) {
    try {
        const resp = await fetch(`/api/sample/${encodeURIComponent(nodeName)}`);
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        const content = document.getElementById('node-panel-content');
        content.innerHTML = buildPanelHTML(data);

        document.getElementById('node-panel').classList.add('open');

        Plotly.newPlot('substance-chart', data.chart.data, {
            ...data.chart.layout,
            paper_bgcolor: 'white',
            plot_bgcolor: '#f8f9fa',
            font: { size: 11 },
        }, { displayModeBar: false, responsive: true });

    } catch (err) {
        console.error('Failed to load sample details:', err);
    }
}

function buildPanelHTML(data) {
    const meta = data.metadata || {};

    const metaRows = Object.entries(meta)
        .filter(([, v]) => v && v !== 'nan' && v !== '')
        .map(([k, v]) => `<div><span>${formatKey(k)}:</span> ${v}</div>`)
        .join('');

    return `
        <div class="panel-header">
            <h3>${data.name}</h3>
        </div>

        ${metaRows ? `<div class="panel-meta">${metaRows}</div>` : ''}

        <div class="panel-stats">
            <div class="panel-stat">
                <div class="stat-val">${data.substance_count}</div>
                <div class="stat-lbl">Substances</div>
            </div>
            <div class="panel-stat">
                <div class="stat-val">${data.connections}</div>
                <div class="stat-lbl">Connections</div>
            </div>
        </div>

        <div class="panel-section-title">Chemical Profile</div>
        <div id="substance-chart"></div>
    `;
}

function formatKey(k) {
    return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function closeNodePanel() {
    document.getElementById('node-panel').classList.remove('open');
}

// ── Multi-node selection ──────────────────────────────────────
function toggleSelection(nodeName) {
    const idx = selectedNodes.indexOf(nodeName);
    if (idx >= 0) {
        selectedNodes.splice(idx, 1);
    } else {
        selectedNodes.push(nodeName);
    }

    highlightSelectedNodes();

    if (selectedNodes.length >= 2) {
        fetchSelectionSummary();
    } else {
        document.getElementById('selection-drawer').classList.remove('open');
    }
}

function highlightSelectedNodes() {
    if (!figureData || !figureData.data[1]) return;

    const samples = figureData.data[1].customdata || [];
    const newColors = samples.map((name, i) =>
        selectedNodes.includes(name) ? SELECTION_COLOR : originalNodeColors[i] || '#4a7fd4'
    );
    const origSizes = figureData.data[1].marker.size;
    const newSizes = origSizes.map((s, i) =>
        selectedNodes.includes(samples[i]) ? s + 6 : s
    );

    Plotly.restyle('plotly-graph', {
        'marker.color': [newColors],
        'marker.size': [newSizes],
    }, [1]).then(attachClickHandler);
}

async function fetchSelectionSummary() {
    try {
        const resp = await fetch('/api/selection-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: selectedNodes }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        document.getElementById('drawer-title').textContent =
            `${data.n} selected: ${data.selected.join(', ')}`;
        document.getElementById('drawer-body').innerHTML = buildSummaryHTML(data);
        document.getElementById('selection-drawer').classList.add('open');
    } catch (err) {
        console.error('Selection summary failed:', err);
    }
}

function buildSummaryHTML(data) {
    const renderGroup = (title, items, cls, renderFn) => {
        if (!items || !items.length) return '';
        return `
            <div class="summary-group ${cls}">
                <div class="summary-group-title">${title} — ${items.length}</div>
                <div class="substance-tags">${items.map(renderFn).join('')}</div>
            </div>`;
    };

    const valBadges = (values) => values.map(v =>
        `<span class="sample-val"><span class="sample-label" style="background:${SELECTION_COLOR}">${v.sample}</span> ${v.value}%</span>`
    ).join('');

    const allHTML = renderGroup(
        `In ALL ${data.n} selected samples`,
        data.all,
        'group-all',
        sub => `<span class="sub-tag"><span class="sub-name">${sub.name}</span>${valBadges(sub.values)}</span>`
    );

    const someHTML = renderGroup(
        'In SOME selected samples',
        data.some,
        'group-some',
        sub => `<span class="sub-tag"><span class="sub-name">${sub.name}</span>${valBadges(sub.values)}</span>`
    );

    const absentHTML = renderGroup(
        'Not in any selected (common elsewhere)',
        data.notable_absences,
        'group-absent',
        sub => `<span class="sub-tag">${sub.name}<small>${sub.freq_in_dataset} samples</small></span>`
    );

    return allHTML + someHTML + absentHTML;
}

// Silent clear — doesn't call Plotly (used before a full re-render)
function clearSelectionSilent() {
    selectedNodes = [];
    document.getElementById('selection-drawer').classList.remove('open');
}

function clearSelection() {
    selectedNodes = [];
    document.getElementById('selection-drawer').classList.remove('open');

    if (figureData && figureData.data[1] && originalNodeColors.length) {
        const origSizes = figureData.data[1].marker.size;
        Plotly.restyle('plotly-graph', {
            'marker.color': [originalNodeColors],
            'marker.size': [origSizes],
        }, [1]).then(attachClickHandler);
    }
}

// ── Rotate graph ─────────────────────────────────────────────
function rotateGraph(deltaDeg) {
    if (!figureData) return;

    currentRotationDeg += deltaDeg;
    const rad = (currentRotationDeg * Math.PI) / 180;
    const cosA = Math.cos(rad);
    const sinA = Math.sin(rad);

    const nodeTrace = figureData.data[1];
    const edgeTrace = figureData.data[0];

    // Store originals once
    if (!nodeTrace._baseX) {
        nodeTrace._baseX = [...nodeTrace.x];
        nodeTrace._baseY = [...nodeTrace.y];
    }
    if (!edgeTrace._baseX) {
        edgeTrace._baseX = [...edgeTrace.x];
        edgeTrace._baseY = [...edgeTrace.y];
    }

    const newNodeX = nodeTrace._baseX.map((x, i) => x * cosA - nodeTrace._baseY[i] * sinA);
    const newNodeY = nodeTrace._baseX.map((x, i) => x * sinA + nodeTrace._baseY[i] * cosA);

    const newEdgeX = edgeTrace._baseX.map((x, i) =>
        x === null ? null : x * cosA - edgeTrace._baseY[i] * sinA
    );
    const newEdgeY = edgeTrace._baseX.map((x, i) =>
        x === null ? null : x * sinA + edgeTrace._baseY[i] * cosA
    );

    // Update both traces then re-fit and re-attach click handler
    Plotly.restyle('plotly-graph', { x: [newEdgeX], y: [newEdgeY] }, [0])
        .then(() => Plotly.restyle('plotly-graph', { x: [newNodeX], y: [newNodeY] }, [1]))
        .then(() => Plotly.relayout('plotly-graph', { 'xaxis.autorange': true, 'yaxis.autorange': true }))
        .then(attachClickHandler);
}

// ── Chemical filter ───────────────────────────────────────────
async function loadChemicals() {
    try {
        const resp = await fetch('/api/chemicals');
        if (!resp.ok) return;
        const data = await resp.json();
        allChemicals = data.chemicals;
        document.getElementById('filter-section').style.display = 'block';
        document.getElementById('chem-search').value = '';
        renderChemList('');
    } catch (err) {
        console.error('Failed to load chemicals:', err);
    }
}

function renderChemList(search) {
    const q = search.toLowerCase();
    const filtered = q
        ? allChemicals.filter(c => c.name.toLowerCase().includes(q))
        : allChemicals;

    // Group by role, sorted alphabetically within each group
    const groups = new Map();
    for (const c of filtered) {
        const role = c.role || 'Unknown';
        if (!groups.has(role)) groups.set(role, []);
        groups.get(role).push(c);
    }
    for (const items of groups.values()) {
        items.sort((a, b) => a.name.localeCompare(b.name));
    }

    // Render in defined subcategory order, skip empty groups
    const parts = [];
    for (const subcategory of SUBCATEGORY_ORDER) {
        const items = groups.get(subcategory);
        if (!items || !items.length) continue;
        parts.push(`<div class="chem-subcategory-header">${htmlEsc(subcategory)}</div>`);
        for (const c of items) {
            const active = activeFilters.includes(c.name);
            parts.push(`<div class="chem-item${active ? ' active' : ''}" data-chem="${htmlEsc(c.name)}">
                <div class="chem-bar-fill" style="width:${Math.min(c.pct, 100)}%"></div>
                <input type="checkbox" class="chem-check"${active ? ' checked' : ''}>
                <span class="chem-name">${htmlEsc(c.name)}</span>
                <span class="chem-pct">${c.pct}%</span>
            </div>`);
        }
    }
    // Any roles not in the defined order (edge case)
    for (const [role, items] of groups) {
        if (!SUBCATEGORY_ORDER.includes(role) && items.length) {
            parts.push(`<div class="chem-subcategory-header">${htmlEsc(role)}</div>`);
            for (const c of items) {
                const active = activeFilters.includes(c.name);
                parts.push(`<div class="chem-item${active ? ' active' : ''}" data-chem="${htmlEsc(c.name)}">
                    <div class="chem-bar-fill" style="width:${Math.min(c.pct, 100)}%"></div>
                    <input type="checkbox" class="chem-check"${active ? ' checked' : ''}>
                    <span class="chem-name">${htmlEsc(c.name)}</span>
                    <span class="chem-pct">${c.pct}%</span>
                </div>`);
            }
        }
    }

    const list = document.getElementById('chem-list');
    list.innerHTML = parts.join('');

    list.querySelectorAll('.chem-item').forEach(el => {
        el.addEventListener('click', () => toggleChemFilter(el.dataset.chem));
    });
}

function toggleChemFilter(name) {
    const idx = activeFilters.indexOf(name);
    if (idx >= 0) activeFilters.splice(idx, 1);
    else activeFilters.push(name);

    renderFilterChips();
    renderChemList(document.getElementById('chem-search').value);
    updateFilterCount();

    clearTimeout(filterDebounceTimer);
    filterDebounceTimer = setTimeout(computeGraph, 500);
}

function renderFilterChips() {
    document.getElementById('filter-chips').innerHTML = activeFilters.map(name =>
        `<span class="filter-chip">${htmlEsc(name)}<button data-chem="${htmlEsc(name)}" class="chip-remove">×</button></span>`
    ).join('');

    document.querySelectorAll('.chip-remove').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            toggleChemFilter(btn.dataset.chem);
        });
    });
}

function updateFilterCount() {
    const el = document.getElementById('filter-count');
    el.textContent = activeFilters.length ? `(${activeFilters.length} active)` : '';
}

function setFilterMode(mode) {
    filterMode = mode;
    document.getElementById('mode-any').classList.toggle('active', mode === 'any');
    document.getElementById('mode-all').classList.toggle('active', mode === 'all');
    if (activeFilters.length > 0) {
        clearTimeout(filterDebounceTimer);
        filterDebounceTimer = setTimeout(computeGraph, 300);
    }
}

function clearFiltersSilent() {
    activeFilters = [];
    allChemicals = [];
    filterMode = 'any';
    document.getElementById('filter-section').style.display = 'none';
    document.getElementById('filter-chips').innerHTML = '';
    document.getElementById('chem-list').innerHTML = '';
    document.getElementById('filter-count').textContent = '';
    const anyBtn = document.getElementById('mode-any');
    const allBtn = document.getElementById('mode-all');
    if (anyBtn) { anyBtn.classList.add('active'); }
    if (allBtn) { allBtn.classList.remove('active'); }
}

// ── Timeline ──────────────────────────────────────────────────
function initTimeline(dataMin, dataMax) {
    timelineDataMin = dataMin;
    timelineDataMax = dataMax;
    timelineMinYear = dataMin;
    timelineMaxYear = dataMax;

    const minEl = document.getElementById('timeline-min');
    const maxEl = document.getElementById('timeline-max');
    minEl.min = dataMin; minEl.max = dataMax; minEl.value = dataMin;
    maxEl.min = dataMin; maxEl.max = dataMax; maxEl.value = dataMax;

    buildTimelineTicks(dataMin, dataMax);
    updateTimelineFill();
    updateTimelineLabel();
    document.getElementById('timeline-bar').style.display = 'block';
}

function buildTimelineTicks(dataMin, dataMax) {
    const container = document.getElementById('timeline-ticks');
    container.innerHTML = '';
    for (let y = dataMin; y <= dataMax; y++) {
        const span = document.createElement('span');
        span.className = 'timeline-tick';
        span.textContent = y;
        span.dataset.year = y;
        container.appendChild(span);
    }
    updateTickHighlights();
}

function updateTickHighlights() {
    document.querySelectorAll('.timeline-tick').forEach(el => {
        const y = parseInt(el.dataset.year);
        el.classList.toggle('active', y >= timelineMinYear && y <= timelineMaxYear);
    });
}

function onTimelineMinChange() {
    let val = parseInt(document.getElementById('timeline-min').value);
    if (val > timelineMaxYear) { val = timelineMaxYear; document.getElementById('timeline-min').value = val; }
    timelineMinYear = val;
    updateTimelineFill();
    updateTimelineLabel();
    updateTickHighlights();
    clearTimeout(timelineDebounceTimer);
    timelineDebounceTimer = setTimeout(computeGraph, 500);
}

function onTimelineMaxChange() {
    let val = parseInt(document.getElementById('timeline-max').value);
    if (val < timelineMinYear) { val = timelineMinYear; document.getElementById('timeline-max').value = val; }
    timelineMaxYear = val;
    updateTimelineFill();
    updateTimelineLabel();
    updateTickHighlights();
    clearTimeout(timelineDebounceTimer);
    timelineDebounceTimer = setTimeout(computeGraph, 500);
}

function updateTimelineFill() {
    if (timelineDataMin === null || timelineDataMax === timelineDataMin) return;
    const range = timelineDataMax - timelineDataMin;
    const pctMin = (timelineMinYear - timelineDataMin) / range * 100;
    const pctMax = (timelineMaxYear - timelineDataMin) / range * 100;
    const fill = document.getElementById('timeline-fill');
    fill.style.left  = pctMin + '%';
    fill.style.width = (pctMax - pctMin) + '%';
}

function updateTimelineLabel() {
    const label = document.getElementById('timeline-range-label');
    if (timelineMinYear === timelineDataMin && timelineMaxYear === timelineDataMax) {
        label.textContent = `All years  (${timelineDataMin}–${timelineDataMax})`;
    } else if (timelineMinYear === timelineMaxYear) {
        label.textContent = String(timelineMinYear);
    } else {
        label.textContent = `${timelineMinYear} – ${timelineMaxYear}`;
    }
}

function resetTimeline() {
    timelineDataMin = null; timelineDataMax = null;
    timelineMinYear = null; timelineMaxYear = null;
    document.getElementById('timeline-bar').style.display = 'none';
    document.getElementById('timeline-ticks').innerHTML = '';
}

function htmlEsc(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Export GraphML ────────────────────────────────────────────
function exportGraphML() {
    if (!currentDataset) { alert('Load a dataset first.'); return; }
    window.location.href = '/api/export/graphml';
}

// ── Network Analysis ──────────────────────────────────────────
const ANALYSIS_COLORS = [
    '#4a7fd4', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#e91e63', '#00bcd4', '#8bc34a',
    '#ff5722', '#607d8b', '#795548', '#ff9800', '#3f51b5',
];

async function runNetworkAnalysis() {
    if (!currentDataset) { alert('Load a dataset first.'); return; }

    const modal = document.getElementById('analysis-modal');
    const body  = document.getElementById('analysis-modal-body');
    body.innerHTML = '<div style="display:flex;justify-content:center;padding:40px"><div class="spinner"></div></div>';
    modal.classList.add('open');

    try {
        const resp = await fetch('/api/network-analysis');
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        body.innerHTML = buildAnalysisHTML(data);
    } catch (err) {
        body.innerHTML = `<div class="analysis-error">Analysis failed: ${htmlEsc(err.message)}</div>`;
    }
}

function closeAnalysisModal(event) {
    if (event.target === event.currentTarget) {
        event.currentTarget.classList.remove('open');
    }
}

function buildAnalysisHTML(data) {
    let html = `<div style="font-size:12px;color:#556e8a;margin-bottom:16px">
        Graph: <strong>${data.n_nodes}</strong> nodes &nbsp;·&nbsp; <strong>${data.n_edges}</strong> edges
    </div>`;

    // Centrality
    html += `<div class="analysis-section">
        <div class="analysis-section-title">Centrality Measures</div>
        <div class="centrality-grid">
            ${buildCentralityCol('Degree', data.centrality.degree)}
            ${buildCentralityCol('Betweenness', data.centrality.betweenness)}
            ${buildCentralityCol('Eigenvector', data.centrality.eigenvector)}
        </div>
    </div>`;

    // Community detection tabs
    const algos = Object.values(data.communities);
    const tabs = algos.map((a, i) => {
        const label = (a.n_communities != null) ? `${a.name} (${a.n_communities})` : a.name;
        return `<button class="community-tab${i === 0 ? ' active' : ''}" onclick="switchCommunityTab(${i})">${htmlEsc(label)}</button>`;
    }).join('');
    const panels = algos.map((a, i) =>
        `<div class="community-panel${i === 0 ? ' active' : ''}" id="comm-panel-${i}">${buildCommunityPanel(a)}</div>`
    ).join('');

    html += `<div class="analysis-section">
        <div class="analysis-section-title">Community Detection</div>
        <div class="community-tabs">${tabs}</div>
        ${panels}
    </div>`;

    return html;
}

function buildCentralityCol(title, entries) {
    const rows = entries.map((e, i) => `
        <div class="centrality-row">
            <span class="centrality-rank">${i + 1}</span>
            <span class="centrality-node" title="${htmlEsc(e.node)}">${htmlEsc(e.node)}</span>
            <span class="centrality-val">${e.value.toFixed(4)}</span>
        </div>`).join('');
    return `<div class="centrality-col"><h4>${title}</h4>${rows}</div>`;
}

function buildCommunityPanel(algo) {
    if (algo.error)   return `<div class="analysis-error">Error: ${htmlEsc(algo.error)}</div>`;
    if (algo.skipped) return `<div class="analysis-skipped">${htmlEsc(algo.reason)}</div>`;

    let html = '';
    if (algo.description) html += `<div class="community-desc">${htmlEsc(algo.description)}</div>`;
    if (algo.modularity != null) {
        html += `<div style="font-size:11px;color:#556e8a;margin-bottom:10px">
            Modularity: <span class="modularity-badge">${algo.modularity}</span>
        </div>`;
    }

    html += '<div class="community-list">';
    algo.communities.forEach((members, i) => {
        const color = ANALYSIS_COLORS[i % ANALYSIS_COLORS.length];
        const tags = members.map(m => `<span class="community-member-tag">${htmlEsc(m)}</span>`).join('');
        html += `<div class="community-item">
            <div class="community-item-header" style="color:${color}">
                Community ${i + 1} — ${members.length} member${members.length !== 1 ? 's' : ''}
            </div>
            <div class="community-members">${tags}</div>
        </div>`;
    });
    html += '</div>';
    return html;
}

function switchCommunityTab(idx) {
    document.querySelectorAll('.community-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
    document.querySelectorAll('.community-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
}

// ── Stats ─────────────────────────────────────────────────────
function updateStats(stats) {
    document.getElementById('stats-block').innerHTML = `
        <div class="stat-row"><span>Samples</span><strong>${stats.n_samples}</strong></div>
        <div class="stat-row"><span>Edges</span><strong>${stats.n_edges}</strong></div>
        <div class="stat-row"><span>Clusters</span><strong>${stats.n_clusters}</strong></div>
        <div class="stat-row"><span>Isolated</span><strong>${stats.n_isolates}</strong></div>
    `;
}

// ── Analytics Modal ───────────────────────────────────────────

let analyticsLoaded = { metrics: false, sweep: false, clusters: false, timeline: false };

function openAnalytics() {
    document.getElementById('analytics-modal').classList.add('open');
    analyticsLoaded = { metrics: false, sweep: false, clusters: false, timeline: false };
    // Always reload the active tab on open
    showAnalyticsTab('metrics', document.querySelector('.atab'));
}

function closeAnalyticsModal(event) {
    if (event.target.id === 'analytics-modal')
        document.getElementById('analytics-modal').classList.remove('open');
}

function showAnalyticsTab(tab, btn) {
    // Update tab buttons
    document.querySelectorAll('.atab').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    // Show correct panel
    ['metrics','sweep','clusters','timeline'].forEach(t => {
        const el = document.getElementById('atab-' + t);
        el.style.display = t === tab ? 'block' : 'none';
        el.classList.toggle('active', t === tab);
    });
    // Load content if not yet loaded
    if (!analyticsLoaded[tab]) {
        analyticsLoaded[tab] = true;
        if (tab === 'metrics')  loadMetricComparison();
        if (tab === 'sweep')    loadThresholdSweep();
        if (tab === 'clusters') loadClusterProfiles();
        if (tab === 'timeline') loadTimeline();
    }
}

// ── Metric Comparison Tab ─────────────────────────────────────

async function loadMetricComparison() {
    const el = document.getElementById('atab-metrics');
    el.innerHTML = '<p class="analytics-note">Computing…</p>';
    try {
        const res = await fetch('/api/analytics/metric-comparison');
        if (!res.ok) { el.innerHTML = `<p class="analysis-error">${(await res.json()).detail}</p>`; return; }
        const data = await res.json();
        renderMetricComparison(data, el);
    } catch (e) {
        el.innerHTML = `<p class="analysis-error">Error: ${e.message}</p>`;
    }
}

function renderMetricComparison(data, el) {
    const cols = ['metric','edges','clusters','isolates','largest','largest_pct'];
    const hdrs = ['Metric','Edges','Clusters','Isolates','Largest (n)','Largest (%)'];
    let html = `<div class="analytics-section-title">Metric Comparison at threshold ${data.threshold.toFixed(2)}</div>`;
    html += `<p class="analytics-note">${data.n_samples} samples — each metric run independently at the current threshold. "Joint (equal)" uses equal weights across all 5 metrics.</p>`;
    html += '<table class="metric-table"><thead><tr>';
    hdrs.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    data.rows.forEach(row => {
        const isJoint = row.metric === 'Joint (equal)';
        html += `<tr class="${isJoint ? 'metric-highlight' : ''}">`;
        html += `<td>${row.metric}</td>`;
        html += `<td class="num">${row.edges}</td>`;
        html += `<td class="num">${row.clusters}</td>`;
        html += `<td class="num">${row.isolates}</td>`;
        html += `<td class="num">${row.largest}</td>`;
        html += `<td class="num">${row.largest_pct}%</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
}

// ── Threshold Sweep Tab ───────────────────────────────────────

async function loadThresholdSweep() {
    const el = document.getElementById('atab-sweep');
    el.innerHTML = '<p class="analytics-note">Computing sweep…</p>';
    try {
        const res = await fetch('/api/analytics/threshold-sweep');
        if (!res.ok) { el.innerHTML = `<p class="analysis-error">${(await res.json()).detail}</p>`; return; }
        const data = await res.json();
        renderThresholdSweep(data, el);
    } catch (e) {
        el.innerHTML = `<p class="analysis-error">Error: ${e.message}</p>`;
    }
}

function renderThresholdSweep(data, el) {
    el.innerHTML = `
        <div class="analytics-section-title">Threshold Sweep — Joint Similarity</div>
        <p class="analytics-note">Shows how cluster structure changes across thresholds. Blue = current weights. Grey = equal weights baseline.</p>
        <div id="sweep-chart"></div>`;

    const thresholds = data.thresholds;
    const mkTrace = (rows, name, color, dash) => ({
        x: thresholds,
        y: rows.map(r => r.largest_pct),
        name,
        type: 'scatter', mode: 'lines+markers',
        line: { color, width: 2, dash: dash || 'solid' },
        marker: { size: 5 }
    });

    const traceClustersCur = {
        x: thresholds, y: data.current_weights.map(r => r.clusters),
        name: 'Clusters (current)', type: 'scatter', mode: 'lines+markers',
        yaxis: 'y2', line: { color: '#4a7fd4', width: 2, dash: 'dot' }, marker: { size: 5 }
    };
    const traceClustersEq = {
        x: thresholds, y: data.equal_weights.map(r => r.clusters),
        name: 'Clusters (equal)', type: 'scatter', mode: 'lines+markers',
        yaxis: 'y2', line: { color: '#aaaaaa', width: 2, dash: 'dot' }, marker: { size: 5 }
    };

    const traces = [
        mkTrace(data.current_weights, 'Largest cluster % (current)', '#4a7fd4'),
        mkTrace(data.equal_weights,   'Largest cluster % (equal)',   '#aaaaaa'),
        traceClustersCur, traceClustersEq,
    ];

    const layout = {
        margin: { t: 20, b: 50, l: 55, r: 55 },
        xaxis: { title: 'Threshold', dtick: 0.05 },
        yaxis:  { title: 'Largest cluster (%)', range: [0, 105] },
        yaxis2: { title: 'Cluster count', overlaying: 'y', side: 'right', rangemode: 'tozero' },
        legend: { orientation: 'h', y: -0.25 },
        paper_bgcolor: '#fff', plot_bgcolor: '#f8fafc',
        shapes: [{
            type: 'line',
            x0: parseFloat(document.getElementById('threshold')?.value || 0.5),
            x1: parseFloat(document.getElementById('threshold')?.value || 0.5),
            y0: 0, y1: 1, yref: 'paper',
            line: { color: '#e74c3c', width: 1.5, dash: 'dash' }
        }],
        annotations: [{
            x: parseFloat(document.getElementById('threshold')?.value || 0.5),
            y: 1, yref: 'paper', text: 'current', showarrow: false,
            font: { size: 10, color: '#e74c3c' }, yanchor: 'bottom'
        }]
    };

    Plotly.newPlot('sweep-chart', traces, layout, { responsive: true, displayModeBar: false });
}

// ── Cluster Profiles Tab ──────────────────────────────────────

async function loadClusterProfiles() {
    const el = document.getElementById('atab-clusters');
    el.innerHTML = '<p class="analytics-note">Loading cluster profiles…</p>';
    try {
        const res = await fetch('/api/analytics/cluster-profiles');
        if (!res.ok) { el.innerHTML = `<p class="analysis-error">${(await res.json()).detail}</p>`; return; }
        const data = await res.json();
        renderClusterProfiles(data, el);
    } catch (e) {
        el.innerHTML = `<p class="analysis-error">Error: ${e.message}</p>`;
    }
}

function renderClusterProfiles(data, el) {
    const CLUSTER_COLORS = [
        '#4a7fd4','#e74c3c','#2ecc71','#f39c12','#9b59b6',
        '#1abc9c','#e67e22','#e91e63','#00bcd4','#8bc34a'
    ];

    let html = `<div class="analytics-section-title">Cluster Chemical Profiles</div>`;
    html += `<p class="analytics-note">${data.n_clusters} cluster${data.n_clusters !== 1 ? 's' : ''}, ${data.n_isolates} isolate${data.n_isolates !== 1 ? 's' : ''} — top substances by mean proportion.</p>`;

    // Main clusters
    const clusters = data.clusters.filter(c => !c.is_isolate);
    if (clusters.length) {
        html += '<div class="cluster-grid">';
        clusters.forEach((c, i) => {
            const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
            const yrs = Object.entries(c.year_dist || {}).sort((a,b) => a[0]-b[0])
                              .map(([y,n]) => `${y}:${n}`).join(' · ');
            const simStr = c.within_sim !== null ? `within-sim ${c.within_sim}` : '';
            html += `<div class="cluster-card">
                <div class="cluster-card-header" style="color:${color}">
                    Cluster ${c.cluster_id} &nbsp;·&nbsp; ${c.size} sample${c.size !== 1 ? 's' : ''}
                </div>
                <div class="cluster-card-meta">${[simStr, yrs].filter(Boolean).join(' &nbsp;|&nbsp; ')}</div>`;
            c.top_substances.forEach(s => {
                const pct = Math.min(s.mean_pct, 100);
                html += `<div class="cluster-sub-bar">
                    <div class="cluster-sub-label">
                        <span>${s.name}</span><span>${s.mean_pct.toFixed(1)}%</span>
                    </div>
                    <div class="cluster-sub-track">
                        <div class="cluster-sub-fill" style="width:${pct}%;background:${color}"></div>
                    </div>
                </div>`;
            });
            html += '</div>';
        });
        html += '</div>';
    }

    // Isolates section
    const isolates = data.clusters.filter(c => c.is_isolate);
    if (isolates.length) {
        html += `<div class="analytics-section-title" style="margin-top:20px">Isolates (${isolates.length})</div>`;
        html += '<div class="cluster-grid">';
        isolates.forEach(c => {
            const yrs = Object.entries(c.year_dist || {}).sort((a,b) => a[0]-b[0])
                              .map(([y,n]) => `${y}:${n}`).join(' · ');
            html += `<div class="cluster-card isolate-card">
                <div class="cluster-card-header" style="color:#778899">
                    ${c.members[0]} &nbsp;·&nbsp; isolate${yrs ? ' · ' + yrs : ''}
                </div>`;
            c.top_substances.forEach(s => {
                const pct = Math.min(s.mean_pct, 100);
                html += `<div class="cluster-sub-bar">
                    <div class="cluster-sub-label">
                        <span>${s.name}</span><span>${s.mean_pct.toFixed(1)}%</span>
                    </div>
                    <div class="cluster-sub-track">
                        <div class="cluster-sub-fill" style="width:${pct}%;background:#aab8c8"></div>
                    </div>
                </div>`;
            });
            html += '</div>';
        });
        html += '</div>';
    }

    el.innerHTML = html;
}

// ── Timeline Tab ──────────────────────────────────────────────

async function loadTimeline() {
    const el = document.getElementById('atab-timeline');
    el.innerHTML = '<p class="analytics-note">Computing temporal analysis…</p>';
    try {
        const res = await fetch('/api/analytics/temporal-split');
        if (!res.ok) {
            const err = await res.json();
            el.innerHTML = `<p class="analysis-error">${err.detail}</p>`; return;
        }
        const data = await res.json();
        renderTimeline(data, el);
    } catch (e) {
        el.innerHTML = `<p class="analysis-error">Error: ${e.message}</p>`;
    }
}

function renderTimeline(data, el) {
    let html = `<div class="analytics-section-title">Within-Year Statistics</div>`;
    html += `<p class="analytics-note">Mean pairwise joint similarity, cluster count, and top substances per year. Higher within-year similarity = more homogeneous batch within that period.</p>`;

    html += '<div class="timeline-year-grid">';
    data.year_stats.forEach(ys => {
        const sim = ys.mean_sim !== null ? ys.mean_sim.toFixed(4) : 'n/a';
        const cl  = ys.clusters !== null ? ys.clusters : '—';
        const iso = ys.isolates !== null ? ys.isolates : '—';
        html += `<div class="timeline-year-card">
            <div class="timeline-year-header">${ys.year} &nbsp;<small style="color:#778899;font-weight:400">(n=${ys.n})</small></div>
            <div class="timeline-year-stat">Mean similarity: <strong>${sim}</strong></div>
            <div class="timeline-year-stat">Clusters: <strong>${cl}</strong> · Isolates: <strong>${iso}</strong></div>`;
        if (ys.std_sim !== null) {
            html += `<div class="timeline-year-stat" style="color:#aab">Std dev: ${ys.std_sim.toFixed(4)}</div>`;
        }
        html += '</div>';
    });
    html += '</div>';

    // Cross-year similarity table
    if (data.cross_year.length) {
        html += `<div class="analytics-section-title" style="margin-top:20px">Cross-Year Mean Similarity</div>`;
        html += `<p class="analytics-note">Mean pairwise joint similarity between all pairs from year A and year B. Falling cross-year similarity indicates compositional divergence over time.</p>`;
        html += '<table class="cross-year-table"><thead><tr><th>Year A</th><th>Year B</th><th>Mean similarity</th><th>Pairs</th></tr></thead><tbody>';
        data.cross_year.forEach(row => {
            const bar = '█'.repeat(Math.round(row.mean_sim * 10));
            html += `<tr><td>${row.year_a}</td><td>${row.year_b}</td><td><strong>${row.mean_sim.toFixed(4)}</strong> <span style="color:#4a7fd4;font-size:10px">${bar}</span></td><td>${row.n_pairs}</td></tr>`;
        });
        html += '</tbody></table>';
    }

    // Composition chart by year
    if (Object.keys(data.year_composition).length) {
        html += `<div class="analytics-section-title" style="margin-top:20px">Top Substance Proportions by Year</div>`;
        html += '<div id="timeline-comp-chart"></div>';
    }

    el.innerHTML = html;

    // Render composition Plotly chart
    if (Object.keys(data.year_composition).length) {
        const years = data.all_years.map(String);
        // Collect all unique substances across all years
        const allSubs = new Set();
        Object.values(data.year_composition).forEach(subs => subs.forEach(s => allSubs.add(s.name)));
        const subList = Array.from(allSubs);

        const traces = subList.slice(0, 8).map((sub, i) => {
            const COLORS = ['#4a7fd4','#e74c3c','#2ecc71','#f39c12','#9b59b6',
                            '#1abc9c','#e67e22','#607d8b'];
            return {
                name: sub,
                x: years,
                y: years.map(yr => {
                    const subs = data.year_composition[yr] || [];
                    const found = subs.find(s => s.name === sub);
                    return found ? found.mean_pct : 0;
                }),
                type: 'bar',
                marker: { color: COLORS[i % COLORS.length] },
            };
        });

        Plotly.newPlot('timeline-comp-chart', traces, {
            barmode: 'stack',
            margin: { t: 10, b: 50, l: 55, r: 20 },
            xaxis: { title: 'Year' },
            yaxis: { title: 'Mean proportion (%)', rangemode: 'tozero' },
            legend: { orientation: 'h', y: -0.3 },
            paper_bgcolor: '#fff', plot_bgcolor: '#f8fafc',
        }, { responsive: true, displayModeBar: false });
    }
}
