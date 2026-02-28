import Chart from 'chart.js/auto';

const BACKEND_URL = import.meta.env?.VITE_BACKEND_URL || 'http://localhost:8000';

// ──────────────────────────────────────────────
// Chart instances (destroyed before re-init)
// ──────────────────────────────────────────────
let trendChart = null;
let distChart = null;

// ──────────────────────────────────────────────
// Data cache (populated after first fetch)
// ──────────────────────────────────────────────
let allSessions = [];      // StoredSession[]
let detailCache = {};       // session_id → full detail object
let activeMetric = 'mean_score';

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function parseTs(ts) {
  if (!ts) return new Date();
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? new Date() : d;
}

function bandLabel(band) {
  if (!band) return 'unknown';
  return band.replace('4_6_hz', '4–6 Hz').replace('6_8_hz', '6–8 Hz').replace('8_12_hz', '8–12 Hz');
}

function fmtDate(ts) {
  const d = parseTs(ts);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

function fmtDuration(minutes) {
  if (!minutes || minutes <= 0) return '—';
  if (minutes < 1) return `${(minutes * 60).toFixed(0)}s`;
  return `${minutes.toFixed(1)} min`;
}

// ──────────────────────────────────────────────
// Aggregate computation (mean + weekly slope)
// ──────────────────────────────────────────────
function computeAggregate(sessions) {
  if (!sessions.length) return { mean: null, slope: 0 };
  const scores = sessions.map(s => s.mean_score);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  if (sessions.length < 2) return { mean, slope: 0 };

  const xs = sessions.map(s => parseTs(s.timestamp).getTime());
  const xm = xs.reduce((a, b) => a + b, 0) / xs.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - xm) * (sessions[i].mean_score - mean);
    den += (xs[i] - xm) ** 2;
  }
  const weekMs = 7 * 24 * 60 * 60 * 1000;
  return { mean, slope: den ? (num / den) * weekMs : 0 };
}

// ──────────────────────────────────────────────
// Sidebar summary cards
// ──────────────────────────────────────────────
function updateSummaryCards(sessions) {
  document.getElementById('profileSessionCount').textContent = String(sessions.length);

  if (!sessions.length) {
    document.getElementById('profileMeanScore').textContent = '—';
    document.getElementById('profileTrendLabel').textContent = 'No data yet';
    document.getElementById('trendBadgeProfile').textContent = '—';
    document.getElementById('trendBadgeProfile').style.color = '#64748b';
    document.getElementById('profileAvgDuration').textContent = '—';
    document.getElementById('profileLastSession').textContent = '—';
    return;
  }

  const { mean, slope } = computeAggregate(sessions);
  document.getElementById('profileMeanScore').textContent = mean.toFixed(2);

  const dir = slope >= 0 ? '+' : '';
  const label = `${dir}${slope.toFixed(2)} / week`;
  const badge = document.getElementById('trendBadgeProfile');
  document.getElementById('profileTrendLabel').textContent =
    slope > 0.1 ? 'Worsening' : slope < -0.1 ? 'Improving' : 'Stable';
  badge.textContent = label;
  badge.style.color = slope > 0.1 ? '#ef4444' : slope < -0.1 ? '#22c55e' : '#64748b';

  // Avg duration
  const avgDur = sessions.reduce((s, x) => s + x.duration_minutes, 0) / sessions.length;
  document.getElementById('profileAvgDuration').textContent = fmtDuration(avgDur);

  // Last session
  const sorted = sessions.slice().sort((a, b) => parseTs(b.timestamp) - parseTs(a.timestamp));
  document.getElementById('profileLastSession').textContent =
    parseTs(sorted[0].timestamp).toLocaleDateString();
}

// ──────────────────────────────────────────────
// Statistics overview cards (needs detail data)
// ──────────────────────────────────────────────
function updateStatsCards(sessions, details) {
  const scores = sessions.map(s => s.mean_score);
  document.getElementById('statBestScore').textContent =
    scores.length ? Math.min(...scores).toFixed(2) : '—';
  document.getElementById('statWorstScore').textContent =
    scores.length ? Math.max(...scores).toFixed(2) : '—';

  // Total time
  const totalMin = sessions.reduce((s, x) => s + (x.duration_minutes || 0), 0);
  document.getElementById('statTotalTime').textContent = fmtDuration(totalMin);

  // RMS & stability from detail data
  const rmsValues = [];
  const stabValues = [];
  Object.values(details).forEach(d => {
    if (d.raw_summary?.intensity_profile?.rms_mean != null)
      rmsValues.push(d.raw_summary.intensity_profile.rms_mean);
    if (d.raw_summary?.variability_profile?.stability_index != null)
      stabValues.push(d.raw_summary.variability_profile.stability_index);
  });

  document.getElementById('statAvgRms').textContent =
    rmsValues.length ? (rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length).toFixed(3) : '—';
  document.getElementById('statAvgStability').textContent =
    stabValues.length ? (stabValues.reduce((a, b) => a + b, 0) / stabValues.length).toFixed(3) : '—';
}

// ──────────────────────────────────────────────
// Main Overtime Chart (multi-metric)
// ──────────────────────────────────────────────
const METRIC_CFG = {
  mean_score: {
    label: 'Session Severity Trend',
    color: '#2dd4bf', fill: 'rgba(45,212,191,0.15)',
    extractor: (d) => d.raw_summary?.intensity_profile?.tremor_score?.mean ?? d.mean_score,
    yLabel: 'Score (0–10)', min: 0, max: 10,
  },
  rms: {
    label: 'RMS Intensity Over Time',
    color: '#a855f7', fill: 'rgba(168,85,247,0.15)',
    extractor: (d) => d.raw_summary?.intensity_profile?.rms_mean ?? null,
    yLabel: 'RMS',
  },
  stability: {
    label: 'Stability Index Over Time',
    color: '#22c55e', fill: 'rgba(34,197,94,0.15)',
    extractor: (d) => d.raw_summary?.variability_profile?.stability_index ?? null,
    yLabel: 'Stability Index',
  },
  entropy: {
    label: 'Spectral Entropy Over Time',
    color: '#fb7185', fill: 'rgba(251,113,133,0.15)',
    extractor: (d) => d.raw_summary?.variability_profile?.spectral_entropy ?? null,
    yLabel: 'Entropy',
  },
  cv: {
    label: 'Coefficient of Variation Over Time',
    color: '#fbbf24', fill: 'rgba(251,191,36,0.15)',
    extractor: (d) => d.raw_summary?.variability_profile?.coefficient_of_variation ?? null,
    yLabel: 'CoV',
  },
  band: {
    label: 'Dominant Band Over Time',
    color: '#a855f7', fill: 'rgba(168,85,247,0.18)',
    extractor: (d) => {
      const b = d.raw_summary?.frequency_profile?.dominant_band ?? d.dominant_band;
      switch (b) { case '4_6_hz': return 1; case '6_8_hz': return 2; case '8_12_hz': return 3; default: return 0; }
    },
    yLabel: 'Band',
    stepped: true,
    min: 0, max: 3,
    tickCallback: (v) => ['—', '4–6 Hz', '6–8 Hz', '8–12 Hz'][v] || '',
    tooltipLabel: (ctx) => {
      const names = ['Unknown', '4–6 Hz', '6–8 Hz', '8–12 Hz'];
      return `Dominant band: ${names[ctx.parsed.y] || 'Unknown'}`;
    },
  },
};

function renderOvertimeChart(metric) {
  const cfg = METRIC_CFG[metric];
  if (!cfg) return;

  document.getElementById('overtimeChartLabel').textContent = cfg.label;

  // Build data from detail cache (ordered by the session list)
  const labels = [];
  const data = [];
  allSessions.forEach(s => {
    const detail = detailCache[s.session_id];
    if (!detail) return;
    labels.push(parseTs(s.timestamp).toLocaleDateString());
    data.push(cfg.extractor(detail));
  });

  const ctx = document.getElementById('profileTrendChart').getContext('2d');

  if (trendChart) {
    trendChart.destroy();
    trendChart = null;
  }

  const pointColors = metric === 'band'
    ? data.map(v => ['#64748b', '#fb7185', '#fbbf24', '#2dd4bf'][v] || '#64748b')
    : cfg.color;

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: cfg.color,
        backgroundColor: cfg.fill,
        fill: true,
        tension: cfg.stepped ? 0 : 0.35,
        stepped: cfg.stepped || false,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: pointColors,
        pointBorderColor: 'rgba(0,0,0,0.3)',
        pointBorderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(6,6,16,0.9)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#94a3b8',
          bodyColor: '#e2e8f0',
          callbacks: {
            label: cfg.tooltipLabel || (ctx => `${cfg.yLabel}: ${ctx.parsed.y != null ? ctx.parsed.y.toFixed(3) : '—'}`),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: 'rgba(148,163,184,0.8)', font: { size: 10 } },
          grid: { color: 'rgba(30,41,59,0.6)' },
        },
        y: {
          ticks: {
            color: 'rgba(148,163,184,0.8)',
            font: { size: 10 },
            callback: cfg.tickCallback || undefined,
          },
          grid: { color: 'rgba(30,41,59,0.6)' },
          ...(cfg.min != null ? { min: cfg.min } : {}),
          ...(cfg.max != null ? { max: cfg.max } : {}),
        },
      },
      onClick: (evt, elements) => {
        if (elements.length) {
          const idx = elements[0].index;
          const s = allSessions[idx];
          if (s) openSessionDetail(s.session_id);
        }
      },
    },
  });
}

// ──────────────────────────────────────────────
// Intensity Distribution (stacked area chart)
// ──────────────────────────────────────────────
function renderDistributionChart() {
  const labels = [];
  const low = [], mod = [], high = [], vhigh = [];

  allSessions.forEach(s => {
    const d = detailCache[s.session_id];
    if (!d || !d.raw_summary?.intensity_distribution) return;
    labels.push(parseTs(s.timestamp).toLocaleDateString());
    const dist = d.raw_summary.intensity_distribution;
    low.push((dist.low_fraction || 0) * 100);
    mod.push((dist.moderate_fraction || 0) * 100);
    high.push((dist.high_fraction || 0) * 100);
    vhigh.push((dist.very_high_fraction || 0) * 100);
  });

  const ctx = document.getElementById('profileDistChart').getContext('2d');
  if (distChart) { distChart.destroy(); distChart = null; }

  distChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Low', data: low, fill: true,
          backgroundColor: 'rgba(34,197,94,0.25)', borderColor: '#22c55e',
          tension: 0.4, pointRadius: 0,
        },
        {
          label: 'Moderate', data: mod, fill: true,
          backgroundColor: 'rgba(251,191,36,0.25)', borderColor: '#fbbf24',
          tension: 0.4, pointRadius: 0,
        },
        {
          label: 'High', data: high, fill: true,
          backgroundColor: 'rgba(251,113,133,0.25)', borderColor: '#fb7185',
          tension: 0.4, pointRadius: 0,
        },
        {
          label: 'Very High', data: vhigh, fill: true,
          backgroundColor: 'rgba(239,68,68,0.3)', borderColor: '#ef4444',
          tension: 0.4, pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: {
          display: true, position: 'top',
          labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 10, padding: 12 },
        },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: 'rgba(6,6,16,0.9)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#94a3b8', bodyColor: '#e2e8f0',
          callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` },
        },
      },
      scales: {
        x: {
          ticks: { color: 'rgba(148,163,184,0.8)', font: { size: 10 } },
          grid: { color: 'rgba(30,41,59,0.6)' },
        },
        y: {
          stacked: false,
          ticks: { color: 'rgba(148,163,184,0.8)', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(30,41,59,0.6)' },
          min: 0,
        },
      },
    },
  });
}

// ──────────────────────────────────────────────
// Session list rendering
// ──────────────────────────────────────────────
function renderSessionList(sessions) {
  const empty = document.getElementById('sessionListEmpty');
  const wrapper = document.getElementById('sessionList');
  const body = document.getElementById('sessionListBody');
  const countLabel = document.getElementById('sessionCountLabel');

  countLabel.textContent = `${sessions.length} SESSIONS`;

  if (!sessions.length) {
    empty.style.display = 'block';
    wrapper.style.display = 'none';
    return;
  }

  empty.style.display = 'none';
  wrapper.style.display = 'block';
  body.innerHTML = '';

  sessions
    .slice()
    .sort((a, b) => parseTs(b.timestamp) - parseTs(a.timestamp))
    .forEach(s => {
      const tr = document.createElement('tr');
      tr.className = 'border-b border-white/[0.04] hover:bg-violet-500/20 transition-colors cursor-pointer group';
      tr.dataset.sessionId = s.session_id;

      const d = parseTs(s.timestamp);
      const dateStr = `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

      const confCls = s.confidence_level === 'High'
        ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/40'
        : s.confidence_level === 'Low'
          ? 'bg-rose-500/10 text-rose-300 border border-rose-500/40'
          : 'bg-amber-500/10 text-amber-300 border border-amber-500/40';

      tr.innerHTML = `
        <td class="py-2.5 pl-4 pr-3 text-slate-300">${dateStr}</td>
        <td class="py-2.5 pr-3 font-mono text-[11px] text-slate-100">${s.mean_score.toFixed(2)}</td>
        <td class="py-2.5 pr-3 text-[11px] text-slate-300">${bandLabel(s.dominant_band)}</td>
        <td class="py-2.5 pr-4 text-[11px]">
          <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold ${confCls}">
            ${s.confidence_level}
          </span>
        </td>
      `;

      tr.addEventListener('click', (e) => {
        // Prevent event from bubbling to document which might close it redundantly or cause race conditions
        e.stopPropagation();

        openSessionDetail(s.session_id);
        const menu = document.getElementById('sessionDropdownMenu');
        const icon = document.getElementById('dropdownIcon');
        if (menu) {
          menu.style.opacity = '0';
          menu.style.pointerEvents = 'none';
          menu.style.transform = 'scale(0.95)';
        }
        if (icon) icon.style.transform = 'rotate(0deg)';
        document.getElementById('selectedSessionLabel').textContent = `${dateStr} — Score: ${s.mean_score.toFixed(2)}`;
      });
      body.appendChild(tr);
    });
}

// ──────────────────────────────────────────────
// Session detail panel
// ──────────────────────────────────────────────
async function openSessionDetail(sessionId) {
  const panel = document.getElementById('sessionDetailPanel');
  const placeholder = document.getElementById('sessionDetailPlaceholder');

  if (placeholder) placeholder.style.display = 'none';

  panel.style.display = 'flex';
  panel.classList.add('flex-col');
  panel.style.opacity = '1';
  panel.style.transform = 'translateY(0)';

  // Highlight active row
  document.querySelectorAll('#sessionListBody tr').forEach(tr => {
    tr.classList.toggle('bg-violet-500/[0.1]', tr.dataset.sessionId === sessionId);
  });

  let detail = detailCache[sessionId];
  if (!detail) {
    try {
      const resp = await fetch(`${BACKEND_URL}/profile/sessions/${sessionId}`);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      detail = await resp.json();
      detailCache[sessionId] = detail;
    } catch (err) {
      console.error('Failed to load session detail:', err);
      document.getElementById('detailReport').innerHTML =
        `<span class="text-rose-400">Failed to load session detail.</span>`;
      return;
    }
  }

  // Metadata strip
  const metaEl = document.getElementById('detailMeta');
  const raw = detail.raw_summary || {};
  const meta = raw.metadata || {};
  metaEl.innerHTML = [
    chip('Session', detail.session_id, 'text-violet-400'),
    chip('Date', fmtDate(detail.timestamp), 'text-slate-300'),
    chip('Duration', fmtDuration(detail.duration_minutes), 'text-teal-400'),
    chip('Condition', (meta.condition || 'rest').toUpperCase(), 'text-amber-400'),
    chip('Medication', meta.medication_status || 'unknown', 'text-slate-400'),
  ].join('');

  // Biomarker cards
  const bio = document.getElementById('detailBiomarkers');
  const ts = raw.intensity_profile?.tremor_score || {};
  const fp = raw.frequency_profile || {};
  const vp = raw.variability_profile || {};
  const ip = raw.intensity_profile || {};
  bio.innerHTML = [
    bioCard('Mean Score', (ts.mean ?? detail.mean_score)?.toFixed(2), 'text-teal-400'),
    bioCard('Std Dev', ts.std?.toFixed(3) ?? '—', 'text-slate-300'),
    bioCard('Max Score', ts.max?.toFixed(2) ?? '—', 'text-rose-400'),
    bioCard('RMS', ip.rms_mean?.toFixed(3) ?? '—', 'text-violet-400'),
    bioCard('Stability', vp.stability_index?.toFixed(3) ?? '—', 'text-emerald-400'),
    bioCard('Entropy', vp.spectral_entropy?.toFixed(3) ?? '—', 'text-amber-400'),
    bioCard('CoV', vp.coefficient_of_variation?.toFixed(3) ?? '—', 'text-slate-300'),
    bioCard('Dom. Band', bandLabel(fp.dominant_band || detail.dominant_band), 'text-violet-400'),
    bioCard('Band Switches', String(fp.band_switch_count ?? '—'), 'text-slate-300'),
  ].join('');

  // Clinical summary
  const reportEl = document.getElementById('detailReport');
  reportEl.innerHTML = renderMarkdown(detail.clinical_summary || 'No report available.');

  // Advisory
  document.getElementById('detailAdvisory').textContent = detail.advisory_note || '';

  // Smooth scroll into view
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function chip(label, value, colorCls) {
  return `<div class="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5 flex flex-col">
    <span class="text-[9px] text-slate-600 font-bold uppercase tracking-widest">${label}</span>
    <span class="text-[11px] font-semibold ${colorCls} font-mono">${value}</span>
  </div>`;
}

function bioCard(label, value, colorCls) {
  return `<div class="bg-white/[0.025] border border-white/[0.05] rounded-lg px-3 py-2 text-center">
    <div class="text-[9px] text-slate-600 font-bold uppercase tracking-wider mb-0.5">${label}</div>
    <div class="font-mono text-sm font-bold ${colorCls}">${value}</div>
  </div>`;
}

function renderMarkdown(text) {
  // Ultra-light markdown → HTML (## headings, **bold**, newlines)
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^## (.+)$/gm, '<h3 class="text-violet-400 text-[11px] font-bold uppercase tracking-widest mt-3.5 mb-1.5">$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-slate-200">$1</strong>')
    .replace(/\n{2,}/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// ──────────────────────────────────────────────
// Metric tab switching
// ──────────────────────────────────────────────
function initMetricTabs() {
  const tabs = document.querySelectorAll('#metricTabs button');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeMetric = tab.dataset.metric;
      renderOvertimeChart(activeMetric);
    });
  });
}

// ──────────────────────────────────────────────
// Close detail panel
// ──────────────────────────────────────────────
function initCloseDetail() {
  document.getElementById('closeDetailBtn').addEventListener('click', () => {
    document.getElementById('sessionDetailPanel').style.display = 'none';

    const placeholder = document.getElementById('sessionDetailPlaceholder');
    if (placeholder) placeholder.style.display = 'flex';

    const label = document.getElementById('selectedSessionLabel');
    if (label) label.textContent = 'Select a past session...';

    document.querySelectorAll('#sessionListBody tr').forEach(tr =>
      tr.classList.remove('bg-violet-500/[0.1]'));
  });
}

// ──────────────────────────────────────────────
// Dropdown toggle
// ──────────────────────────────────────────────
function initDropdown() {
  const btn = document.getElementById('sessionDropdownBtn');
  const menu = document.getElementById('sessionDropdownMenu');
  const icon = document.getElementById('dropdownIcon');

  if (btn && menu) {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = menu.style.opacity === '1';
      if (isOpen) {
        menu.style.opacity = '0';
        menu.style.pointerEvents = 'none';
        menu.style.transform = 'scale(0.95)';
        if (icon) icon.style.transform = 'rotate(0deg)';
      } else {
        menu.style.opacity = '1';
        menu.style.pointerEvents = 'auto';
        menu.style.transform = 'scale(1)';
        if (icon) icon.style.transform = 'rotate(180deg)';
      }
    });

    document.addEventListener('click', (e) => {
      // Close dropdown when clicking anywhere outside of the menu
      if (!menu.contains(e.target) && menu.style.opacity === '1') {
        menu.style.opacity = '0';
        menu.style.pointerEvents = 'none';
        menu.style.transform = 'scale(0.95)';
        if (icon) icon.style.transform = 'rotate(0deg)';
      }
    });
  }
}

// ──────────────────────────────────────────────
// Data loading
// ──────────────────────────────────────────────
async function loadSessions() {
  try {
    const resp = await fetch(BACKEND_URL + '/profile/sessions');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    allSessions = await resp.json();

    updateSummaryCards(allSessions);
    renderSessionList(allSessions);

    // Fetch detail for ALL sessions (in parallel, max 10 at a time)
    const chunks = [];
    for (let i = 0; i < allSessions.length; i += 10) {
      chunks.push(allSessions.slice(i, i + 10));
    }

    for (const chunk of chunks) {
      await Promise.all(chunk.map(async (s) => {
        if (detailCache[s.session_id]) return;
        try {
          const r = await fetch(`${BACKEND_URL}/profile/sessions/${s.session_id}`);
          if (r.ok) {
            detailCache[s.session_id] = await r.json();
          }
        } catch { /* ignore individual errors */ }
      }));
    }

    // Now render charts & stats that depend on details
    updateStatsCards(allSessions, detailCache);
    renderOvertimeChart(activeMetric);
    renderDistributionChart();
  } catch (err) {
    console.error('Failed to load sessions:', err);
  }
}

// ──────────────────────────────────────────────
// Fade-in animation (simple)
// ──────────────────────────────────────────────
function animateFadeItems() {
  document.querySelectorAll('.fade-item').forEach((el, i) => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, 60 * i);
  });
}

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  animateFadeItems();
  initMetricTabs();
  initCloseDetail();
  initDropdown();
  loadSessions();
});
