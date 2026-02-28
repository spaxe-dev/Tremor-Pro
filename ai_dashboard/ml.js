/**
 * ML Classification Monitor — ml.js
 *
 * Connects to ESP32 SSE, sends each window to the backend
 * /ml/classify endpoint, and renders results in a live log.
 */

const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let sse = null;
let stats = { total: 0, tremor: 0, noTremor: 0 };
let maxLogEntries = 200;

// ──────────────────────────────────────────────
// DOM refs
// ──────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ──────────────────────────────────────────────
// Check model status on load
// ──────────────────────────────────────────────
async function checkModelStatus() {
    try {
        const res = await fetch(`${BACKEND}/ml/status`);
        const data = await res.json();
        if (data.available && data.model_loaded) {
            $('modelStatus').textContent = '✓ Ready';
            $('modelStatus').style.color = '#34d399';
            $('modelTrees').textContent = data.n_estimators;
            $('modelFeatures').textContent = data.n_features;
        } else if (data.available) {
            $('modelStatus').textContent = 'Not trained';
            $('modelStatus').style.color = '#fbbf24';
        } else {
            $('modelStatus').textContent = '✗ Unavailable';
            $('modelStatus').style.color = '#f87171';
        }
    } catch (e) {
        $('modelStatus').textContent = '✗ Backend offline';
        $('modelStatus').style.color = '#f87171';
    }
}

// ──────────────────────────────────────────────
// Format helpers
// ──────────────────────────────────────────────
function timeStr() {
    const d = new Date();
    return d.toLocaleTimeString('en-GB', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

function badgeClass(pred) {
    const map = { no_tremor: 'badge-no', parkinsonian: 'badge-pk', essential: 'badge-es', physiological: 'badge-ph' };
    return map[pred] || 'badge-no';
}

function predLabel(pred) {
    const map = { no_tremor: 'No Tremor', parkinsonian: 'Parkinson', essential: 'Essential', physiological: 'Physio.' };
    return map[pred] || pred;
}

// ──────────────────────────────────────────────
// Update live summary cards
// ──────────────────────────────────────────────
function updateSummaryCards(result, b1, b2, b3) {
    // Last prediction
    $('lastPrediction').textContent = predLabel(result.prediction);
    const isTremor = result.is_tremor;
    $('lastPrediction').style.color = isTremor ? '#f9a8d4' : '#6ee7b7';

    // Confidence
    $('lastConfidence').textContent = (result.confidence * 100).toFixed(1) + '%';

    // Band powers
    $('lastBands').textContent = `${b1.toFixed(3)} / ${b2.toFixed(3)} / ${b3.toFixed(3)}`;

    // Is tremor
    const el = $('isTremor');
    if (isTremor) {
        el.textContent = 'YES';
        el.style.color = '#fb7185';
    } else {
        el.textContent = 'NO';
        el.style.color = '#34d399';
    }

    // Stats
    stats.total++;
    if (isTremor) stats.tremor++;
    else stats.noTremor++;

    $('totalWindows').textContent = stats.total;
    $('tremorCount').textContent = stats.tremor;
    $('noTremorCount').textContent = stats.noTremor;
    $('tremorPct').textContent = stats.total > 0
        ? ((stats.tremor / stats.total) * 100).toFixed(1) + '%'
        : '0%';
}

// ──────────────────────────────────────────────
// Add log entry
// ──────────────────────────────────────────────
function addLogEntry(result, b1, b2, b3) {
    // Remove placeholder
    const ph = $('logPlaceholder');
    if (ph) ph.remove();

    const log = $('classificationLog');
    const entry = document.createElement('div');
    entry.className = `ml-log-entry ${result.is_tremor ? 'is-tremor' : 'no-tremor'}`;

    const probs = result.probabilities || {};
    const pNo = (probs.no_tremor || 0) * 100;
    const pPk = (probs.parkinsonian || 0) * 100;
    const pEs = (probs.essential || 0) * 100;
    const pPh = (probs.physiological || 0) * 100;

    entry.innerHTML = `
    <span class="font-mono text-[11px] text-slate-500">${timeStr()}</span>
    <div>
      <div class="prob-bar-wrap">
        <div class="prob-no" style="width:${pNo}%" title="No Tremor: ${pNo.toFixed(1)}%"></div>
        <div class="prob-pk" style="width:${pPk}%" title="Parkinsonian: ${pPk.toFixed(1)}%"></div>
        <div class="prob-es" style="width:${pEs}%" title="Essential: ${pEs.toFixed(1)}%"></div>
        <div class="prob-ph" style="width:${pPh}%" title="Physiological: ${pPh.toFixed(1)}%"></div>
      </div>
      <div class="flex gap-3 mt-1 text-[10px] text-slate-600">
        <span><span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1"></span>${pNo.toFixed(0)}%</span>
        <span><span class="inline-block w-1.5 h-1.5 rounded-full bg-pink-400 mr-1"></span>${pPk.toFixed(0)}%</span>
        <span><span class="inline-block w-1.5 h-1.5 rounded-full bg-violet-400 mr-1"></span>${pEs.toFixed(0)}%</span>
        <span><span class="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 mr-1"></span>${pPh.toFixed(0)}%</span>
      </div>
    </div>
    <span class="badge ${badgeClass(result.prediction)}">${predLabel(result.prediction)}</span>
    <span class="font-mono text-[11px] ${result.confidence > 0.8 ? 'text-emerald-400' : result.confidence > 0.5 ? 'text-amber-400' : 'text-rose-400'}">${(result.confidence * 100).toFixed(1)}%</span>
    <span class="font-mono text-[11px] font-bold ${result.is_tremor ? 'text-rose-400' : 'text-emerald-400'}">${result.is_tremor ? '⚠ YES' : '✓ NO'}</span>
  `;

    // Insert at top
    log.prepend(entry);

    // Trim old entries
    while (log.children.length > maxLogEntries) {
        log.removeChild(log.lastChild);
    }

    $('logCount').textContent = `${Math.min(stats.total, maxLogEntries)} entries`;
}

// ──────────────────────────────────────────────
// Classify a window via backend
// ──────────────────────────────────────────────
async function classifyWindow(b1, b2, b3, meanNorm) {
    try {
        const res = await fetch(`${BACKEND}/ml/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ b1, b2, b3, meanNorm }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.warn('[ML] classify error:', e);
        return null;
    }
}

// ──────────────────────────────────────────────
// ESP32 SSE Connection
// ──────────────────────────────────────────────
function connectESP() {
    const ip = $('espIP').value.trim();
    if (!ip) return;

    // Persist IP
    localStorage.setItem('espIP', ip);

    if (sse) { sse.close(); sse = null; }

    const url = `http://${ip}/events`;
    setStatus('disconnected', 'Connecting…');

    try { sse = new EventSource(url); }
    catch (e) {
        setStatus('disconnected', 'Failed');
        localStorage.setItem('espConnected', 'false');
        return;
    }

    sse.onopen = () => {
        setStatus('connected', 'Connected');
        localStorage.setItem('espConnected', 'true');
        $('liveTag').querySelector('span:first-child').style.background = '#34d399';
        $('liveTag').querySelector('span:first-child').classList.add('animate-pulse');
        $('liveText').textContent = 'LIVE';
        $('liveText').style.color = '#34d399';
    };

    sse.onerror = () => {
        setStatus('disconnected', 'Disconnected');
        localStorage.setItem('espConnected', 'false');
        $('liveTag').querySelector('span:first-child').style.background = '#64748b';
        $('liveTag').querySelector('span:first-child').classList.remove('animate-pulse');
        $('liveText').textContent = 'OFFLINE';
        $('liveText').style.color = '#64748b';
    };

    // Listen for band data → classify
    sse.addEventListener('bands', async (e) => {
        const j = JSON.parse(e.data);
        const { b1, b2, b3, meanNorm } = j;

        const result = await classifyWindow(b1, b2, b3, meanNorm || 0);
        if (!result) return;

        updateSummaryCards(result, b1, b2, b3);
        addLogEntry(result, b1, b2, b3);
    });
}

// ──────────────────────────────────────────────
// Status helpers
// ──────────────────────────────────────────────
function setStatus(state, text) {
    const pill = $('statusPill');
    const dot = $('statusDot');
    const txt = $('statusText');
    pill.className = `status-chip ${state}`;
    txt.textContent = text;
}

// ──────────────────────────────────────────────
// Clear log
// ──────────────────────────────────────────────
function clearLog() {
    $('classificationLog').innerHTML = `
    <div id="logPlaceholder" class="flex flex-col items-center justify-center gap-3 py-12 text-center">
      <div class="w-16 h-16 rounded-full border border-teal-400/20 flex items-center justify-center"
        style="background:linear-gradient(135deg,rgba(45,212,191,0.1),rgba(139,92,246,0.06))">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(45,212,191,0.5)" stroke-width="1.5">
          <rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>
          <rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>
        </svg>
      </div>
      <p class="text-[13px] text-slate-500 max-w-[280px]">Log cleared. Connect ESP32 to continue.</p>
    </div>
  `;
    stats = { total: 0, tremor: 0, noTremor: 0 };
    $('totalWindows').textContent = '0';
    $('tremorCount').textContent = '0';
    $('noTremorCount').textContent = '0';
    $('tremorPct').textContent = '0%';
    $('logCount').textContent = '0 entries';
}

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();

    // ESP32 IP Persistence
    const savedIP = localStorage.getItem('espIP');
    const ipInput = $('espIP');
    if (savedIP && ipInput) ipInput.value = savedIP;

    if (ipInput) {
        ipInput.addEventListener('input', (e) => localStorage.setItem('espIP', e.target.value.trim()));
    }

    $('btnConnect').addEventListener('click', connectESP);
    $('btnClearLog').addEventListener('click', clearLog);

    // Auto-connect if it was connected previously
    if (localStorage.getItem('espConnected') === 'true') {
        setTimeout(connectESP, 500); // slight delay for DOM mount
    }

    // Make code blocks copyable
    document.querySelectorAll('.guide-code').forEach(el => {
        el.title = 'Click to copy';
        el.addEventListener('click', () => {
            const text = el.textContent.replace(/\\\n\s*/g, ' ').trim();
            navigator.clipboard.writeText(text).then(() => {
                const orig = el.style.borderColor;
                el.style.borderColor = 'rgba(52,211,153,0.5)';
                setTimeout(() => { el.style.borderColor = orig; }, 600);
            });
        });
    });
});
