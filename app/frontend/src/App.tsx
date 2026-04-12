import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { fetchMethods, fetchSurface, runAlgorithm, downloadCsv } from './api';
import type { Variant, Methods, AlgorithmResult, SurfaceData } from './types';
import Sidebar from './components/Sidebar';
import SurfacePlot from './components/SurfacePlot';
import Results from './components/Results';

export interface PathPoint { x: number; y: number; z: number }
export interface Range { min: number; max: number }

export default function App() {
  const [variant, setVariant] = useState<Variant>('binary');
  const [methods, setMethods] = useState<Methods | null>(null);
  const [result, setResult] = useState<AlgorithmResult | null>(null);
  const [surfaceData, setSurfaceData] = useState<SurfaceData | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [maximize, setMaximize] = useState(false);
  const [bounds, setBounds] = useState({ a: -2.048, b: 2.048 });
  const [x1Range, setX1Range] = useState<Range>({ min: -4, max: 4 });
  const [x2Range, setX2Range] = useState<Range>({ min: -4, max: 4 });
  const surfaceDebounce = useRef<number>(0);

  useEffect(() => { fetchMethods().then(setMethods); }, []);

  const loadSurface = useCallback(async (x1: Range, x2: Range) => {
    setSurfaceData(await fetchSurface(x1.min, x1.max, x2.min, x2.max));
  }, []);

  useEffect(() => {
    window.clearTimeout(surfaceDebounce.current);
    surfaceDebounce.current = window.setTimeout(() => loadSurface(x1Range, x2Range), 300);
  }, [x1Range, x2Range, loadSurface]);

  useEffect(() => { document.documentElement.setAttribute('data-variant', variant); }, [variant]);

  function handleBoundsChange(a: number, b: number) { if (a < b) setBounds({ a, b }); }
  function handleViewRangeChange(axis: 'x1' | 'x2', min: number, max: number) {
    if (min >= max) return;
    if (axis === 'x1') setX1Range({ min, max }); else setX2Range({ min, max });
  }

  const convergencePath = useMemo<PathPoint[] | null>(() => {
    if (!result?.best_positions_history?.length) return null;
    const raw: PathPoint[] = result.best_positions_history.map((pos, i) => ({
      x: pos[0], y: pos[1], z: result.best_history[i],
    }));
    const path = [raw[0]];
    for (let i = 1; i < raw.length; i++) {
      if (raw[i].x !== raw[i - 1].x || raw[i].y !== raw[i - 1].y) path.push(raw[i]);
    }
    const last = raw[raw.length - 1];
    if (path[path.length - 1] !== last) path.push(last);
    return path;
  }, [result]);

  const bestPoint = result && result.best_variables.length >= 2
    ? { x: result.best_variables[0], y: result.best_variables[1], z: result.best_fitness }
    : null;

  async function handleRun(params: Record<string, unknown>) {
    setIsRunning(true); setProgress(30); setMaximize(!!params.maximize);
    const a = Number(params.a), b = Number(params.b);
    if (a !== bounds.a || b !== bounds.b) setBounds({ a, b });
    try {
      setProgress(60);
      const res = await runAlgorithm(params);
      setProgress(100);
      if ('error' in res && res.error) alert('Error: ' + res.error);
      else setResult(res as AlgorithmResult);
    } catch (e) { alert('Request failed: ' + (e as Error).message); }
    finally { setTimeout(() => { setIsRunning(false); setProgress(0); }, 400); }
  }

  return (
    <div className="grid grid-cols-[360px_1fr] h-screen">
      <Sidebar
        variant={variant} methods={methods} isRunning={isRunning} hasResults={!!result}
        bounds={bounds} x1Range={x1Range} x2Range={x2Range}
        onVariantChange={setVariant} onRun={handleRun}
        onDownload={() => result ? downloadCsv() : alert('Run the algorithm first.')}
        onBoundsChange={handleBoundsChange} onViewRangeChange={handleViewRangeChange}
        progress={progress}
      />
      <main className="overflow-y-auto p-6 space-y-5 bg-[#17171a]">
        <SurfacePlot
          data={surfaceData} variant={variant} bounds={bounds}
          bestPoint={bestPoint} convergencePath={convergencePath}
          populationHistory={result?.population_history ?? null}
          bestPerEpoch={result?.best_positions_history?.map((pos, i) => ({
            x: pos[0], y: pos[1], z: result.best_history[i],
          })) ?? null}
          maximize={maximize}
        />
        {result ? (
          <Results result={result} maximize={maximize} />
        ) : (
          <div className="text-[14px] text-[#555] py-16 text-center rounded-xl border border-dashed border-[#2a2a2e]">
            Run the algorithm to see results.
          </div>
        )}
      </main>
    </div>
  );
}
