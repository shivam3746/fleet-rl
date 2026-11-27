const $ = (id) => document.getElementById(id);
const recorded = document.querySelector('meta[name="fleetlab-mode"]').content === 'recorded';
const pages = {overview: 'Overview', replay: recorded ? 'Recorded replay' : 'Live replay', benchmarks: 'Benchmarks', training: 'Training'};
const environments = {synthetic: '50-zone grid', transfer: '72-zone transfer', manhattan: 'Manhattan replay'};
const random = 'Hungarian (random reposition)';
const names = {
  'GraphSAGE': 'GraphSAGE + PPO', 'MLP': 'MLP + PPO', 'Greedy (stay)': 'Greedy · stay',
  'Hungarian (stay)': 'Hungarian · stay', [random]: 'Hungarian · random reposition',
};
const fmt = (value, digits = 2) => value == null ? '—' : value.toLocaleString('en-US', {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});
const pct = (value) => value == null ? '—' : `${fmt(100 * value)}%`;
const signed = (value, digits = 2) => `${value < 0 ? '−' : '+'}${fmt(Math.abs(value), digits)}`;
const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));

const state = {reports: {}, replay: null, suite: 'synthetic', selectedZone: null, playing: false, busy: false, timer: null};
let frames = [];
let frameIndex = 0;

async function request(path, data) {
  if (recorded) {
    if (path === '/api/reports') return readAsset('./data/reports.json');
    return recordedReplay(data);
  }
  const options = data === undefined ? undefined : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data),
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The local dashboard request failed.');
  return result;
}

async function readAsset(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error('The saved experiment asset is unavailable.');
  return response.json();
}

async function recordedReplay(data) {
  if (data?.action === 'advance') {
    frameIndex = Math.min(frameIndex + data.minutes, frames.length - 1);
  } else {
    const seed = data?.seed ?? 1007;
    if (!Number.isInteger(seed) || seed < 1000 || seed > 1019) throw new Error('Choose a saved seed from 1000 to 1019.');
    frames = await readAsset(`./data/replay-${seed}.json`);
    frameIndex = data ? 0 : 5;
  }
  return frames[frameIndex];
}

function metric(label, value, unit, note, tone = '') {
  return `<article class="metric"><div class="metric-label">${escape(label)}</div>` +
    `<strong>${escape(value)}${unit ? ` <small>${escape(unit)}</small>` : ''}</strong>` +
    `<span class="metric-note ${tone}">${escape(note)}</span></article>`;
}

function ordered(summary) {
  return Object.entries(summary).sort((a, b) => (b[1].service_rate ?? 0) - (a[1].service_rate ?? 0));
}

function showOverview() {
  const suites = state.reports.benchmarks || {};
  const suite = suites.synthetic;
  if (suite?.summary.GraphSAGE && suite.summary[random]) {
    const graph = suite.summary.GraphSAGE;
    const reference = suite.summary[random];
    $('overview-context').textContent = `${suite.episodes} shared traces · 50 zones`;
    $('overview-metrics').innerHTML = [
      metric('Service rate', pct(graph.service_rate), '', `${signed((graph.service_rate - reference.service_rate) * 100)} pp vs random repositioning`, 'good'),
      metric('Mean pickup wait', fmt(graph.mean_pickup_wait), 'min', `${signed(graph.mean_pickup_wait - reference.mean_pickup_wait)} min vs random repositioning`, 'good'),
      metric('P90 pickup wait', fmt(graph.p90_pickup_wait), 'min', `${signed(graph.p90_pickup_wait - reference.p90_pickup_wait)} min vs random repositioning`, 'good'),
      metric('Empty travel', fmt(graph.mean_empty_travel), 'veh-min/ep', `${signed(graph.mean_empty_travel - reference.mean_empty_travel)} vs random repositioning`, 'cost'),
    ].join('');
    $('overview-bars').innerHTML = '<div class="bar-axis"><span>90%</span><span>95%</span><span>100%</span></div>' +
      ordered(suite.summary).map(([name, row]) => {
        const percent = Math.max(0, Math.min(100, (row.service_rate - .90) * 1000));
        return `<div class="bar-row"><div class="bar-caption"><span>${escape(names[name] || name)}</span>` +
          `<strong>${pct(row.service_rate)}</strong></div><div class="bar-track"><div class="bar-fill ${name === 'GraphSAGE' ? 'lead' : ''}" style="width:${percent}%"></div></div></div>`;
      }).join('');
  } else {
    $('overview-metrics').innerHTML = '<p class="muted">No saved synthetic report was found.</p>';
    $('overview-bars').innerHTML = '<p class="muted">Run the evaluation to populate this comparison.</p>';
  }
  $('environment-results').innerHTML = Object.entries(environments).map(([key, label]) => {
    const summary = suites[key]?.summary;
    const lead = summary?.GraphSAGE?.service_rate;
    const comparison = summary?.[random]?.service_rate;
    return `<div class="environment-cell"><span>${label}</span><strong>${pct(lead)}</strong>` +
      `<small class="good">${lead == null ? 'Report unavailable' : `${signed((lead - comparison) * 100)} pp vs random repositioning`}</small></div>`;
  }).join('');
}

function mapMarkup(replay, selectable) {
  const tag = selectable ? 'button' : 'div';
  const zones = replay.zones.map((zone) => {
    const label = `Zone ${zone.id}, ${zone.idle} idle, ${zone.waiting} awaiting pickup, ${zone.inbound} inbound`;
    const control = selectable ? `type="button" data-zone="${zone.id}" aria-pressed="${state.selectedZone === zone.id}" aria-label="${label}"` : '';
    const idle = zone.idle ? `<span class="zone-stat" aria-hidden="true">${zone.idle}</span>` : '';
    const wait = zone.waiting ? `<span class="zone-stat wait" aria-hidden="true">${zone.waiting}</span>` : '';
    return `<${tag} class="zone ${zone.idle ? 'has-idle' : ''} ${zone.waiting ? 'has-wait' : ''}" ${control} title="${label}">` +
      `<span>${zone.id}</span><span class="zone-counts">${idle}${wait}</span></${tag}>`;
  }).join('');
  return '<div class="map-surface">' +
    `<div class="zone-grid" style="--columns:${replay.grid.columns}">${zones}</div>` +
    '<div class="map-legend"><span><i class="legend-dot"></i>Idle vehicles</span><span><i class="legend-dot wait"></i>Awaiting pickup</span></div></div>';
}

function renderZoneDetails() {
  const zone = state.replay?.zones.find(({id}) => id === state.selectedZone);
  $('zone-detail').textContent = zone ?
    `Zone ${zone.id} · ${zone.idle} idle · ${zone.waiting} awaiting pickup (${zone.unassigned} unassigned) · ${zone.inbound} idle within 5 min` :
    'Select a zone to view details.';
}

function renderReplay() {
  const replay = state.replay;
  if (!replay) return;
  $('preview-minute').textContent = `Minute ${replay.minute}`;
  $('preview-context').textContent = `${replay.label} · ${replay.idle} idle vehicles`;
  $('preview-map').innerHTML = mapMarkup(replay, false);
  $('live-map').innerHTML = mapMarkup(replay, true);
  $('live-minute').textContent = `Minute ${replay.minute}`;
  $('replay-state').textContent = replay.complete ? 'Replay complete' : `${replay.label} · ${state.playing ? 'Playing' : 'Paused'}`;
  $('replay-metrics').innerHTML = [
    metric('Requests served', fmt(replay.served, 0), '', `of ${replay.arrivals} arrivals`),
    metric('Awaiting pickup', fmt(replay.waiting, 0), '', `${replay.expired} expired so far`),
    metric('Mean pickup wait', fmt(replay.mean_wait), 'min', 'Among requests picked up'),
    metric('Empty travel', fmt(replay.empty_travel, 0), 'veh-min', `${replay.idle} idle vehicles now`),
  ].join('');
  renderZoneDetails();
  const rows = replay.assignments.slice(0, 3).map((row) => ({
    kind: 'Dispatch', description: `Vehicle ${row.vehicle} · request ${row.request}`, detail: `${row.pickup_minutes} min to pickup`,
  }));
  rows.push(...replay.moves.slice(0, 3).map((row) => ({
    kind: 'Move', description: `Vehicle ${row.vehicle} · zone ${row.origin} to ${row.destination}`, detail: row.choice,
  })));
  $('decision-list').innerHTML = rows.length ? rows.map((row) =>
    `<div class="decision-row"><span class="decision-kind ${row.kind === 'Move' ? 'move' : ''}">${row.kind}</span>` +
    `<span>${escape(row.description)}</span><small>${escape(row.detail)}</small></div>`).join('') :
    '<p class="muted">No assignments or repositioning recorded at this boundary.</p>';
  $('evidence').textContent = replay.explanation;
  renderControls();
}

function renderControls() {
  $('step-one').disabled = state.busy || state.playing || !state.replay || state.replay.complete;
  $('step-five').disabled = $('step-one').disabled;
  $('finish').disabled = $('step-one').disabled;
  $('play').disabled = !state.replay || state.replay.complete || (state.busy && !state.playing);
  $('play').textContent = state.playing ? 'Pause' : 'Play';
  $('play').setAttribute('aria-pressed', state.playing);
  $('replay-form').querySelector('button').disabled = state.busy;
}

function pause() {
  state.playing = false;
  window.clearTimeout(state.timer);
  renderControls();
  if (state.replay) {
    $('replay-state').textContent = state.replay.complete ? 'Replay complete' : `${state.replay.label} · Paused`;
  }
}

async function replayAction(data) {
  if (state.busy) return false;
  state.busy = true;
  $('replay-error').hidden = true;
  renderControls();
  try {
    state.replay = await request('/api/replay', data);
    if (state.replay.complete) pause();
    return true;
  } catch (error) {
    $('replay-error').textContent = error.message;
    $('replay-error').hidden = false;
    if (!state.replay) {
      $('preview-minute').textContent = 'Replay unavailable';
      $('preview-map').textContent = error.message;
      $('replay-state').textContent = 'Replay unavailable';
    }
    pause();
    return false;
  } finally {
    state.busy = false;
    renderReplay();
    renderControls();
  }
}

async function tick() {
  if (!state.playing) return;
  await replayAction({action: 'advance', minutes: 1});
  if (state.playing) state.timer = window.setTimeout(tick, 300);
}

function ciText(delta, unit) {
  if (!delta || delta.difference == null) return '—';
  const interval = delta.ci_95 ? ` (${delta.ci_95.map((value) => signed(value)).join(' to ')})` : '';
  return `${signed(delta.difference)} ${unit}${interval}`;
}

function showBenchmark() {
  const suite = state.reports.benchmarks?.[state.suite];
  if (!suite) {
    $('suite-content').innerHTML = '<p class="notice">No saved report for this environment was found.</p>';
    return;
  }
  const graph = suite.summary.GraphSAGE;
  const graphCards = graph ? [
    metric('GraphSAGE service', pct(graph.service_rate), '', `${fmt(graph.served_requests, 0)} of ${fmt(graph.requests, 0)} requests`),
    metric('Mean pickup wait', fmt(graph.mean_pickup_wait), 'min', 'Among served requests'),
    metric('P90 pickup wait', fmt(graph.p90_pickup_wait), 'min', 'Among served requests'),
    metric('Empty travel', fmt(graph.mean_empty_travel), 'veh-min/ep', `Average over ${suite.episodes} episodes`),
  ].join('') : '';
  const rows = ordered(suite.summary).map(([name, row]) => `<tr class="${name === 'GraphSAGE' ? 'highlight' : ''}">` +
    `<td>${escape(names[name] || name)}</td><td>${fmt(row.served_requests, 0)} / ${fmt(row.requests, 0)}</td>` +
    `<td>${pct(row.service_rate)}</td><td>${fmt(row.mean_pickup_wait)} min</td><td>${fmt(row.p90_pickup_wait)} min</td>` +
    `<td>${fmt(row.mean_empty_travel)}</td></tr>`).join('');
  const intervals = Object.entries(suite.paired || {}).map(([name, delta]) => `<tr><td>${escape(names[name] || name)}</td>` +
    `<td>${ciText(delta.service_rate_pp, 'pp')}</td><td>${ciText(delta.mean_pickup_wait, 'min')}</td>` +
    `<td>${ciText(delta.mean_empty_travel, 'veh-min')}</td></tr>`).join('');
  const notes = {synthetic: '20 reserved synthetic demand traces.', transfer: 'GraphSAGE transferred without retraining.',
    manhattan: 'Simulated replay of metered trips. Meter-on time stands in for request arrival.'};
  $('suite-content').innerHTML = `<div class="metric-grid">${graphCards}</div>` +
    `<article class="card"><div class="panel-header"><div><h2>${environments[state.suite]}</h2><p>${notes[state.suite]}</p></div></div>` +
    '<div class="table-scroll"><table><thead><tr><th>Policy</th><th>Served / requests</th><th>Service rate</th>' +
    `<th>Mean wait</th><th>P90 wait</th><th>Empty veh-min/ep</th></tr></thead><tbody>${rows}</tbody></table></div></article>` +
    `<article class="card environment-card"><div class="panel-header"><div><h2>GraphSAGE minus each reference</h2>` +
    '<p>Point differences and paired 95% intervals across demand traces</p></div></div><div class="table-scroll"><table>' +
    '<thead><tr><th>Reference policy</th><th>Service</th><th>Mean wait</th><th>Empty travel per episode</th></tr></thead>' +
    `<tbody>${intervals}</tbody></table></div><p class="table-note">Learned policies were evaluated from one training seed.</p></article>`;
}

const plots = {
  mean_return: {title: 'Mean episode return', label: (n) => fmt(n, 0)},
  service_rate: {title: 'Service rate', label: (n) => `${fmt(100 * n, 0)}%`},
  mean_pickup_wait: {title: 'Mean pickup wait (min)', label: (n) => fmt(n, 1)},
  mean_empty_travel: {title: 'Empty travel (vehicle-min per episode)', label: (n) => fmt(n, 0)},
};

function showTraining() {
  const training = state.reports.training || {};
  const records = Object.entries(training);
  if (!records.length) {
    $('training-summary').innerHTML = '<p class="notice">No local training log was found.</p>';
    return;
  }
  $('training-summary').innerHTML = records.map(([name, rows]) => {
    const best = rows.filter((row) => row.transitions > 0).reduce((a, b) =>
      !a || b.validation.mean_return > a.validation.mean_return ? b : a, null);
    const policy = name === 'graph' ? 'GraphSAGE' : 'MLP';
    return best ? metric(`${policy} best return`, fmt(best.validation.mean_return), '', `At ${fmt(best.transitions, 0)} transitions`) :
      metric(`${policy} initial return`, fmt(rows[0].validation.mean_return), '', 'Awaiting the first trained checkpoint');
  }).join('');
  $('training-charts').innerHTML = Object.entries(plots).map(([key, chart]) =>
    `<article class="card"><div class="panel-header"><h2>${chart.title}</h2></div><div class="chart-key"><span>GraphSAGE</span><span class="mlp">MLP</span></div>` +
    `<div class="plot"><canvas data-plot="${key}" role="img" aria-label="${chart.title} over PPO transitions"></canvas></div></article>`).join('');
  window.requestAnimationFrame(drawPlots);
}

function drawPlots() {
  if ($('page-training').hidden) return;
  const records = Object.entries(state.reports.training || {});
  for (const canvas of document.querySelectorAll('canvas[data-plot]')) {
    const key = canvas.dataset.plot;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width) continue;
    const dpi = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpi);
    canvas.height = Math.round(height * dpi);
    const ctx = canvas.getContext('2d');
    ctx.scale(dpi, dpi);
    const bounds = {left: 47, right: width - 14, top: 10, bottom: height - 27};
    const rows = records.flatMap(([, values]) => values);
    const values = rows.map((row) => row.validation[key]).filter(Number.isFinite);
    if (!values.length) continue;
    const maxStep = Math.max(1, ...rows.map((row) => row.transitions));
    const padding = (Math.max(...values) - Math.min(...values)) * .09 || 1;
    const low = Math.min(...values) - padding;
    const high = Math.max(...values) + padding;
    const x = (n) => bounds.left + (n / maxStep) * (bounds.right - bounds.left);
    const y = (n) => bounds.bottom - ((n - low) / (high - low)) * (bounds.bottom - bounds.top);
    ctx.font = '12px Segoe UI, system-ui, sans-serif';
    ctx.fillStyle = '#536478';
    ctx.strokeStyle = '#dce5ef';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const value = low + (i / 4) * (high - low);
      ctx.beginPath(); ctx.moveTo(bounds.left, y(value)); ctx.lineTo(bounds.right, y(value)); ctx.stroke();
      ctx.fillText(plots[key].label(value), bounds.left - 7, y(value));
    }
    ctx.textAlign = 'center';
    for (let i = 0; i <= 3; i++) {
      const step = (i / 3) * maxStep;
      ctx.fillText(i ? `${fmt(step / 1000, 1)}k` : '0', x(step), height - 14);
    }
    for (const [name, series] of records) {
      const color = name === 'graph' ? '#2c59dd' : '#ac5717';
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (const row of series) {
        if (!Number.isFinite(row.validation[key])) continue;
        if (started) ctx.lineTo(x(row.transitions), y(row.validation[key]));
        else ctx.moveTo(x(row.transitions), y(row.validation[key]));
        started = true;
      }
      ctx.stroke();
    }
  }
}

function navigate() {
  const hash = location.hash.slice(1);
  const page = Object.hasOwn(pages, hash) ? hash : 'overview';
  for (const key of Object.keys(pages)) $('page-' + key).hidden = key !== page;
  for (const link of document.querySelectorAll('nav [data-page]')) {
    if (link.dataset.page === page) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  $('breadcrumb').textContent = pages[page];
  if (page === 'training') window.requestAnimationFrame(drawPlots);
  if (page !== 'replay') pause();
  window.scrollTo(0, 0);
}

$('replay-form').addEventListener('submit', (event) => {
  event.preventDefault(); pause(); state.selectedZone = null;
  replayAction({action: 'start', seed: Number($('seed').value)});
});
$('step-one').addEventListener('click', () => replayAction({action: 'advance', minutes: 1}));
$('step-five').addEventListener('click', () => replayAction({action: 'advance', minutes: 5}));
$('finish').addEventListener('click', () => replayAction({action: 'advance', minutes: 150}));
$('play').addEventListener('click', () => {
  if (state.playing) { pause(); return; }
  state.playing = true; renderControls(); tick();
});
$('live-map').addEventListener('click', (event) => {
  const button = event.target.closest('[data-zone]');
  if (!button) return;
  state.selectedZone = Number(button.dataset.zone);
  for (const zone of $('live-map').querySelectorAll('[data-zone]')) zone.setAttribute('aria-pressed', zone === button);
  renderZoneDetails();
});
for (const button of document.querySelectorAll('[data-suite]')) button.addEventListener('click', () => {
  state.suite = button.dataset.suite;
  for (const other of document.querySelectorAll('[data-suite]')) other.setAttribute('aria-pressed', other === button);
  showBenchmark();
});
$('operator-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button');
  button.disabled = true;
  $('operator-answer').hidden = false;
  $('operator-answer').textContent = 'Waiting for the local operator…';
  try {
    const result = await request('/api/operator', {message: $('question').value.trim()});
    $('operator-answer').textContent = result.answer;
  } catch (error) {
    $('operator-answer').textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
window.addEventListener('hashchange', navigate);
window.addEventListener('resize', drawPlots);
if (recorded) {
  document.querySelector('nav [data-page="replay"]').lastChild.textContent = 'Recorded replay';
  document.querySelector('.intro a[href="#replay"]').textContent = 'Open replay';
  $('replay-title').textContent = 'Recorded replay';
  $('replay-description').textContent = 'Play back saved GraphSAGE decisions for held-out seeds 1000–1019.';
  $('simulation-source').textContent = 'Recorded simulation';
  $('seed').min = '1000';
  $('seed').max = '1019';
  $('operator-form').closest('.operator-card').hidden = true;
}
navigate();
replayAction();
request('/api/reports').then((reports) => {
  state.reports = reports; showOverview(); showBenchmark(); showTraining();
}).catch((error) => {
  for (const id of ['overview-metrics', 'suite-content', 'training-summary']) $(id).textContent = error.message;
});
