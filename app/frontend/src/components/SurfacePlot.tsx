import { useEffect, useRef, useState, useCallback } from 'react';
import type { SurfaceData, Variant } from '../types';
import type { PathPoint } from '../App';

// -- constants ----------------------------------------------------------------

const COLORSCALES: Record<Variant, [number, string][]> = {
  binary: [
    [0, '#09090b'], [0.15, '#064e3b'], [0.35, '#10b981'],
    [0.55, '#38bdf8'], [0.75, '#a78bfa'], [1, '#fbbf24'],
  ],
  real: [
    [0, '#09090b'], [0.15, '#7c2d12'], [0.35, '#f97316'],
    [0.55, '#fbbf24'], [0.75, '#10b981'], [1, '#38bdf8'],
  ],
};
const ACCENT: Record<Variant, string> = { binary: '#10b981', real: '#f97316' };
const INITIAL_EYE = { x: 1.5, y: 1.5, z: 0.8 };
const CONFIG = { responsive: true, scrollZoom: true };
const SPEEDS = [0.25, 0.5, 1, 2, 4];

type Vec3 = { x: number; y: number; z: number };
type EpochPop = { pos: number[][]; fit: number[] };

function rosenbrock(x: number, y: number) {
  return 100 * (y - x * x) ** 2 + (1 - x) ** 2;
}

function makeLayout(eye: Vec3) {
  return {
    paper_bgcolor: '#18181b', plot_bgcolor: '#18181b',
    margin: { l: 0, r: 0, t: 8, b: 8 },
    scene: {
      xaxis: { title: 'x\u2081', color: '#52525b', gridcolor: '#252528', backgroundcolor: '#151518' },
      yaxis: { title: 'x\u2082', color: '#52525b', gridcolor: '#252528', backgroundcolor: '#151518' },
      zaxis: { title: 'f(x)', color: '#52525b', gridcolor: '#252528', backgroundcolor: '#151518' },
      camera: { eye },
    },
    font: { family: 'IBM Plex Sans, system-ui', color: '#555', size: 11 },
  };
}

function toSpherical(e: Vec3) {
  const r = Math.sqrt(e.x ** 2 + e.y ** 2 + e.z ** 2);
  return { r, theta: Math.atan2(e.y, e.x), phi: Math.acos(Math.max(-1, Math.min(1, e.z / r))) };
}
function toCartesian(r: number, theta: number, phi: number): Vec3 {
  return { x: r * Math.sin(phi) * Math.cos(theta), y: r * Math.sin(phi) * Math.sin(theta), z: r * Math.cos(phi) };
}

function buildWireframe(a: number, b: number) {
  const n = 40;
  const xs: number[] = [], ys: number[] = [], zs: number[] = [];
  function edge(x0: number, y0: number, x1: number, y1: number) {
    for (let i = 0; i <= n; i++) {
      const t = i / n, x = x0 + t * (x1 - x0), y = y0 + t * (y1 - y0);
      xs.push(x); ys.push(y); zs.push(rosenbrock(x, y) + 0.5);
    }
    xs.push(NaN); ys.push(NaN); zs.push(NaN);
  }
  edge(a, a, b, a); edge(b, a, b, b); edge(b, b, a, b); edge(a, b, a, a);
  return { x: xs, y: ys, z: zs, type: 'scatter3d', mode: 'lines', line: { color: '#ffffff', width: 3, dash: 'dot' }, showlegend: false, hoverinfo: 'skip' };
}

/** Deduplicate consecutive identical positions for a clean trail */
function cleanTrail(points: PathPoint[]): PathPoint[] {
  if (points.length === 0) return [];
  const out = [points[0]];
  for (let i = 1; i < points.length; i++) {
    if (points[i].x !== points[i - 1].x || points[i].y !== points[i - 1].y) {
      out.push(points[i]);
    }
  }
  return out;
}

// -- component ----------------------------------------------------------------

interface Props {
  data: SurfaceData | null;
  variant: Variant;
  bounds: { a: number; b: number };
  bestPoint: PathPoint | null;
  convergencePath: PathPoint[] | null;
  populationHistory: EpochPop[] | null;
  bestPerEpoch: PathPoint[] | null;
  maximize: boolean;
}

export default function SurfacePlot({
  data, variant, bounds, bestPoint, convergencePath,
  populationHistory, bestPerEpoch, maximize,
}: Props) {
  const plotEl = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<object | null>(null);
  const wireRef = useRef<object | null>(null);
  const cameraEye = useRef({ ...INITIAL_EYE });
  const [focused, setFocused] = useState(false);

  // Playback
  const [epoch, setEpoch] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(2);
  const playTimer = useRef<number>(0);
  const totalEpochs = populationHistory?.length ?? 0;
  const isLastEpoch = epoch >= totalEpochs - 1;

  // Reset on new results
  const prevHistRef = useRef<EpochPop[] | null>(null);
  useEffect(() => {
    if (populationHistory && populationHistory !== prevHistRef.current) {
      prevHistRef.current = populationHistory;
      setEpoch(0);
      setPlaying(true);
    }
  }, [populationHistory]);

  // -- WASD camera ----------------------------------------------------------
  const keysDown = useRef(new Set<string>());
  const vel = useRef({ dTheta: 0, dPhi: 0, dR: 0 });
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!focused) { keysDown.current.clear(); return; }
    const dn = (e: KeyboardEvent) => { const k = e.key.toLowerCase(); if ('wasdqe'.includes(k)) { e.preventDefault(); keysDown.current.add(k); } };
    const up = (e: KeyboardEvent) => keysDown.current.delete(e.key.toLowerCase());
    window.addEventListener('keydown', dn); window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', dn); window.removeEventListener('keyup', up); keysDown.current.clear(); };
  }, [focused]);

  useEffect(() => {
    if (!focused) { cancelAnimationFrame(rafRef.current); return; }
    const a = 0.004, d = 0.90, m = 0.05;
    function tick() {
      const v = vel.current, k = keysDown.current;
      if (k.has('a')) v.dTheta -= a; if (k.has('d')) v.dTheta += a;
      if (k.has('w')) v.dPhi -= a; if (k.has('s')) v.dPhi += a;
      if (k.has('e')) v.dR -= a * 2; if (k.has('q')) v.dR += a * 2;
      v.dTheta = Math.max(-m, Math.min(m, v.dTheta));
      v.dPhi = Math.max(-m, Math.min(m, v.dPhi));
      v.dR = Math.max(-m * 2, Math.min(m * 2, v.dR));
      if (Math.abs(v.dTheta) > 1e-4 || Math.abs(v.dPhi) > 1e-4 || Math.abs(v.dR) > 1e-4) {
        const s = toSpherical(cameraEye.current);
        const ne = toCartesian(Math.max(0.4, s.r + v.dR), s.theta + v.dTheta, Math.max(0.1, Math.min(Math.PI - 0.1, s.phi + v.dPhi)));
        cameraEye.current = ne;
        if (plotEl.current) Plotly.relayout(plotEl.current, { 'scene.camera.eye': ne });
      }
      v.dTheta *= d; v.dPhi *= d; v.dR *= d;
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [focused]);

  // -- Render surface + wireframe -------------------------------------------
  useEffect(() => {
    if (!plotEl.current || !data) return;
    const el = plotEl.current;
    const surface = {
      x: data.x, y: data.y, z: data.z, type: 'surface',
      colorscale: COLORSCALES[variant],
      opacity: 0.85,
      contours: { z: { show: true, usecolormap: true, highlightcolor: '#fff', project: { z: false } } },
      lighting: { ambient: 0.55, diffuse: 0.6, specular: 0.25, roughness: 0.5 },
      showlegend: false, hoverinfo: 'x+y+z',
    };
    const wire = buildWireframe(bounds.a, bounds.b);
    surfaceRef.current = surface;
    wireRef.current = wire;
    Plotly.react(el, [surface, wire], makeLayout(cameraEye.current), CONFIG).then(() => {
      try {
        (el as unknown as { on: (e: string, h: (d: Record<string, unknown>) => void) => void })
          .on('plotly_relayout', (ev) => { const c = ev['scene.camera'] as { eye?: Vec3 } | undefined; if (c?.eye) cameraEye.current = c.eye; });
      } catch { /* ok */ }
    });
    return () => { if (plotEl.current) Plotly.purge(plotEl.current); };
  }, [data, variant, bounds.a, bounds.b]);

  // -- Render epoch ---------------------------------------------------------
  const renderEpoch = useCallback((ep: number) => {
    if (!plotEl.current || !surfaceRef.current || !wireRef.current || !populationHistory || !bestPerEpoch) return;
    const pop = populationHistory[ep];
    if (!pop) return;

    const accent = ACCENT[variant];

    // Find best index in this epoch using the correct objective direction
    const bestFit = maximize ? Math.max(...pop.fit) : Math.min(...pop.fit);
    const bestIdx = pop.fit.indexOf(bestFit);

    // Population dots — color: good=accent, bad=red
    const popTrace = {
      x: pop.pos.map(p => p[0]), y: pop.pos.map(p => p[1]), z: pop.fit,
      type: 'scatter3d', mode: 'markers',
      marker: {
        size: 4, opacity: 0.75,
        color: pop.fit,
        colorscale: maximize
          ? [[0, '#ef4444'], [0.5, '#eab308'], [1, accent]]
          : [[0, accent], [0.5, '#eab308'], [1, '#ef4444']],
      },
      showlegend: false,
      hovertemplate: 'x\u2081=%{x:.4f}<br>x\u2082=%{y:.4f}<br>f=%{z:.4f}<extra></extra>',
    };

    // Best marker for this epoch
    const bestMarker = {
      x: [pop.pos[bestIdx][0]], y: [pop.pos[bestIdx][1]], z: [pop.fit[bestIdx]],
      type: 'scatter3d', mode: 'markers+text',
      marker: { size: 10, color: accent, line: { color: '#fff', width: 2 } },
      text: [`f = ${pop.fit[bestIdx].toFixed(4)}`],
      textposition: 'top center',
      textfont: { family: 'IBM Plex Mono', size: 14, color: '#ccc' },
      showlegend: false, hoverinfo: 'skip',
    };

    // Trail — use actual best-per-epoch data (from algorithm), deduplicated
    const trailPoints = cleanTrail(bestPerEpoch.slice(0, ep + 1));
    const trail = {
      x: trailPoints.map(p => p.x), y: trailPoints.map(p => p.y), z: trailPoints.map(p => p.z),
      type: 'scatter3d', mode: 'lines',
      line: { color: accent, width: 3 },
      showlegend: false, hoverinfo: 'skip', opacity: 0.6,
    };

    const traces = [surfaceRef.current, wireRef.current, popTrace, bestMarker, trail];

    // On last epoch, add the final optimum from the algorithm result
    if (isLastEpoch && bestPoint) {
      // Offset the label above the surface so it's always visible
      const zMax = Math.max(...pop.fit);
      const zRange = zMax - Math.min(...pop.fit);
      const labelZ = bestPoint.z + Math.max(zRange * 0.3, 1);

      // Marker at actual position
      traces.push({
        x: [bestPoint.x], y: [bestPoint.y], z: [bestPoint.z],
        type: 'scatter3d', mode: 'markers',
        marker: { size: 12, color: '#ffffff', line: { color: accent, width: 3 }, symbol: 'diamond' } as object,
        showlegend: false,
        hovertemplate: `x\u2081=%{x:.6f}<br>x\u2082=%{y:.6f}<br>f(x)=%{z:.6f}<extra>Optimum</extra>`,
      } as object);
      // Floating label above
      traces.push({
        x: [bestPoint.x], y: [bestPoint.y], z: [labelZ],
        type: 'scatter3d', mode: 'text',
        text: [`optimum = ${bestPoint.z.toFixed(6)}`],
        textposition: 'top center',
        textfont: { family: 'IBM Plex Mono', size: 16, color: '#e0e0e0' },
        showlegend: false, hoverinfo: 'skip',
      } as object);
    }

    Plotly.react(plotEl.current, traces, makeLayout(cameraEye.current), CONFIG);
  }, [populationHistory, bestPerEpoch, bestPoint, variant, maximize, isLastEpoch]);

  useEffect(() => { if (populationHistory) renderEpoch(epoch); }, [epoch, renderEpoch, populationHistory]);

  // -- Playback timer -------------------------------------------------------
  useEffect(() => {
    window.clearInterval(playTimer.current);
    if (!playing || !populationHistory) return;
    const interval = 80 / SPEEDS[speedIdx];
    playTimer.current = window.setInterval(() => {
      setEpoch(prev => {
        const next = prev + 1;
        if (next >= totalEpochs) { setPlaying(false); return totalEpochs - 1; }
        return next;
      });
    }, interval);
    return () => window.clearInterval(playTimer.current);
  }, [playing, speedIdx, totalEpochs, populationHistory]);

  // -- Static render (no population data) -----------------------------------
  useEffect(() => {
    if (populationHistory) return;
    if (!bestPoint || !plotEl.current || !surfaceRef.current || !wireRef.current) return;
    const accent = ACCENT[variant];
    Plotly.react(plotEl.current, [
      surfaceRef.current, wireRef.current,
      {
        x: [bestPoint.x], y: [bestPoint.y], z: [bestPoint.z],
        type: 'scatter3d', mode: 'markers+text',
        marker: { size: 8, color: accent, line: { color: '#fff', width: 2 } },
        text: [`f=${bestPoint.z.toFixed(4)}`], textposition: 'top center',
        textfont: { family: 'IBM Plex Mono', size: 14, color: '#999' },
        showlegend: false, hovertemplate: `x\u2081=%{x:.6f}<br>x\u2082=%{y:.6f}<br>f(x)=%{z:.6f}<extra>Best</extra>`,
      },
    ], makeLayout(cameraEye.current), CONFIG);
  }, [bestPoint, variant, populationHistory]);

  // -- JSX ------------------------------------------------------------------
  const currentBestFit = populationHistory && bestPerEpoch?.[epoch]
    ? bestPerEpoch[epoch].z : null;

  return (
    <div
      className={`rounded border bg-[#18181b] overflow-hidden outline-none transition-colors ${focused ? 'border-[#444]' : 'border-[#222]'}`}
      tabIndex={0} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
    >
      <div className="flex items-center justify-between px-4 pt-3 text-[13px]">
        <span className="text-[#666]">Rosenbrock surface</span>
        <div className="flex items-center gap-3 font-mono text-[12px]">
          {populationHistory && <span className="text-[#888]">epoch {epoch + 1}/{totalEpochs}</span>}
          {currentBestFit !== null && <span className="text-[#aaa]">best {currentBestFit.toFixed(6)}</span>}
          <span className="text-[#555]">[{bounds.a}, {bounds.b}]</span>
        </div>
      </div>

      <div ref={plotEl} className="w-full h-[520px]" />

      <div className="px-4 pb-2.5 space-y-1.5">
        {populationHistory && totalEpochs > 0 && (
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => {
              if (isLastEpoch && !playing) { setEpoch(0); setPlaying(true); }
              else setPlaying(p => !p);
            }}
              className="w-8 h-8 rounded border border-[#333] bg-[#1e1e22] text-[#aaa] flex items-center justify-center hover:bg-[#252528] transition-colors cursor-pointer text-[14px]">
              {playing ? '\u23F8' : '\u25B6'}
            </button>
            <input type="range" min={0} max={totalEpochs - 1} value={epoch}
              onChange={e => { setEpoch(Number(e.target.value)); setPlaying(false); }}
              className="flex-1 h-1 accent-[#888] cursor-pointer" />
            <button type="button" onClick={() => setSpeedIdx(i => (i + 1) % SPEEDS.length)}
              className="px-2 h-8 rounded border border-[#333] bg-[#1e1e22] text-[#666] text-[12px] font-mono hover:bg-[#252528] transition-colors cursor-pointer min-w-[44px]">
              {SPEEDS[speedIdx]}x
            </button>
            <span className="text-[12px] text-[#555] font-mono">{populationHistory[epoch]?.pos.length ?? 0}p</span>
          </div>
        )}
        <div className="text-[11px] font-mono text-[#444]">
          {focused ? 'wasd orbit · qe zoom · scroll · drag' : 'click to focus'}
        </div>
      </div>
    </div>
  );
}
